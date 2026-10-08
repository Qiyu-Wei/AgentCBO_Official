from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

if TYPE_CHECKING:
    import pandas as pd
    from ..causal.dag import CausalGraph

try:
    from botorch.models import FixedNoiseGP
    _HAS_FIXED_NOISE = True
except ImportError:
    _HAS_FIXED_NOISE = False


class ConditionalGPNetwork:
    """Per-edge conditional GPs: p(V | pa(V)) for do-calculus propagation."""

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.conditional_gps: Dict[str, Optional[SingleTaskGP]] = {}

    def fit(self, obs_data: pd.DataFrame):
        for var in self.graph.variable_names:
            parents = self.graph.get_parents(var)
            if not parents:
                self.conditional_gps[var] = None
                continue
            X = torch.tensor(obs_data[parents].values, dtype=torch.float64)
            Y = torch.tensor(obs_data[var].values, dtype=torch.float64).unsqueeze(-1)
            d = len(parents)
            # Phase 1: wide constraint fit to estimate noise
            wide_lh = GaussianLikelihood(noise_constraint=Interval(1e-6, 1.0))
            probe = SingleTaskGP(
                train_X=X, train_Y=Y, likelihood=wide_lh,
                covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1))
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(probe.likelihood, probe))
            except Exception:
                pass
            learned_noise = probe.likelihood.noise.item()
            if not np.isfinite(learned_noise):
                learned_noise = 0.1
            # Phase 2: re-fit with appropriate noise model
            if learned_noise < 1e-3:
                if _HAS_FIXED_NOISE:
                    model = FixedNoiseGP(
                        train_X=X, train_Y=Y,
                        train_Yvar=torch.full_like(Y, 1e-4),
                        covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
                        input_transform=Normalize(d=d),
                        outcome_transform=Standardize(m=1))
                else:
                    model = SingleTaskGP(
                        train_X=X, train_Y=Y,
                        train_Yvar=torch.full_like(Y, 1e-4),
                        covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
                        input_transform=Normalize(d=d),
                        outcome_transform=Standardize(m=1))
            else:
                lh = GaussianLikelihood(noise_constraint=Interval(1e-4, 1e-2))
                model = SingleTaskGP(
                    train_X=X, train_Y=Y, likelihood=lh,
                    covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
                    input_transform=Normalize(d=d),
                    outcome_transform=Standardize(m=1))
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                pass
            self.conditional_gps[var] = model

    def refit(self, obs_data: pd.DataFrame):
        self.fit(obs_data)

    def predict(self, var: str, parent_values: np.ndarray,
                include_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        model = self.conditional_gps[var]
        if model is None:
            raise ValueError(f"No GP for root node '{var}'")
        model.eval()
        X = torch.tensor(parent_values, dtype=torch.float64)
        if X.dim() == 1:
            X = X.unsqueeze(-1)
        with torch.no_grad():
            posterior = model.posterior(X, observation_noise=include_noise)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            variance = posterior.variance.squeeze(-1).cpu().numpy()
        return mean, variance
