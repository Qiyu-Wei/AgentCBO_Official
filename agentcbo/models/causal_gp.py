from __future__ import annotations
from typing import Callable

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean


class CausalRBFKernel(Kernel):
    """K(x,x') = σ²·exp(-r²/2) + √Var_do(x)·√Var_do(x')"""
    has_lengthscale = True

    def __init__(self, variance_adjustment: Callable, ard_num_dims: int = None, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        self.variance_adjustment = variance_adjustment
        self.register_parameter('raw_outputscale',
                                torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint('raw_outputscale', Positive())

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def forward(self, x1, x2, diag=False, **params):
        x1_s = x1.div(self.lengthscale)
        x2_s = x2.div(self.lengthscale)
        if diag:
            sq_dist = ((x1_s - x2_s) ** 2).sum(-1)
        else:
            # Direct squared distance avoids autograd gradient singularity
            # at distance=0 that torch.cdist(p=2)**2 would cause
            sq_dist = (x1_s.unsqueeze(-2) - x2_s.unsqueeze(-3)).pow(2).sum(-1)
        rbf = self.outputscale * torch.exp(-0.5 * sq_dist)

        try:
            v1 = self._var_adj(x1).clamp(1e-6, 100.0)
            v2 = self._var_adj(x2).clamp(1e-6, 100.0)
            s1, s2 = torch.sqrt(v1), torch.sqrt(v2)
            if diag:
                causal = s1 * s2
            else:
                causal = s1.unsqueeze(-1) * s2.unsqueeze(-2)
        except Exception:
            causal = 0.0
        return rbf + causal

    def _var_adj(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1]).detach().cpu().numpy()
        v = self.variance_adjustment(x_2d)
        if isinstance(v, np.ndarray):
            v = v.flatten()
        return torch.tensor(v, dtype=x.dtype, device=x.device).reshape(orig_shape)


class CausalMean(Mean):
    """do-calculus mean function wrapper for GPyTorch."""

    def __init__(self, mean_function: Callable):
        super().__init__()
        self.mean_function = mean_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        try:
            x_2d = x.reshape(-1, x.shape[-1]).detach().cpu().numpy()
            m = self.mean_function(x_2d)
            if isinstance(m, np.ndarray):
                m = m.flatten()
            return torch.tensor(m, dtype=x.dtype, device=x.device).reshape(orig_shape)
        except Exception:
            return torch.zeros(orig_shape, dtype=x.dtype, device=x.device)


class CausalGP(SingleTaskGP):
    """CausalRBFKernel + CausalMean + input Normalize."""

    def __init__(self, train_X, train_Y, variance_adjustment=None,
                 mean_function=None, bounds=None, fixed_noise=None):
        d = train_X.shape[-1]
        if fixed_noise is not None:
            from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
            likelihood = FixedNoiseGaussianLikelihood(noise=fixed_noise)
        else:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2))
        if bounds is not None:
            lo, hi = bounds[0], bounds[1]
            degenerate = (hi - lo).abs() < 1e-8
            if degenerate.any():
                hi = hi.clone()
                hi[degenerate] = lo[degenerate] + 1.0
                bounds = torch.stack([lo, hi])
        input_tf = Normalize(d=d, bounds=bounds) if bounds is not None else Normalize(d=d)
        # Keep Y in the original benchmark scale: the causal prior functions are
        # defined in that space, so adding Standardize here would misalign them.
        super().__init__(
            train_X=train_X, train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_tf, outcome_transform=None)

        if variance_adjustment is not None or mean_function is not None:
            if bounds is not None:
                lo = bounds[0].detach().cpu().numpy()
                hi = bounds[1].detach().cpu().numpy()
            else:
                lo = train_X.min(dim=0).values.detach().cpu().numpy()
                hi = train_X.max(dim=0).values.detach().cpu().numpy()
            rng = np.maximum(hi - lo, 1e-8)
            # BoTorch normalizes train_X internally, so the prior callbacks must
            # be evaluated on the original intervention scale.
            def unnorm(x_np, _lo=lo, _rng=rng):
                return x_np * _rng + _lo

        if variance_adjustment is not None:
            wrapped_var = lambda x, _va=variance_adjustment, _un=unnorm: _va(_un(x))
            self.covar_module = CausalRBFKernel(
                variance_adjustment=wrapped_var, ard_num_dims=d)
        if mean_function is not None:
            wrapped_mean = lambda x, _mf=mean_function, _un=unnorm: _mf(_un(x))
            self.mean_module = CausalMean(mean_function=wrapped_mean)
