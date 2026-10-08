from __future__ import annotations
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from scipy.stats import norm


class CausalExpectedImprovement(AcquisitionFunction):
    """CBO-style causal EI with cost normalization (BoTorch native)."""

    def __init__(self, model, best_f: float, task: str = 'min',
                 cost_functions: Dict = None, exploration_set: Tuple = None):
        super().__init__(model=model)
        self.best_f = best_f
        self.task = task
        self.cost_functions = cost_functions
        self.exploration_set = exploration_set

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        variance = posterior.variance.clamp_min(1e-6).squeeze(-2).squeeze(-1)
        sigma = torch.sqrt(variance)
        if self.task == 'min':
            u = (self.best_f - mean) / sigma
        else:
            u = (mean - self.best_f) / sigma
        normal = torch.distributions.Normal(0, 1)
        ei = sigma * (u * normal.cdf(u) + torch.exp(normal.log_prob(u)))
        if self.cost_functions and self.exploration_set:
            cost = self._compute_cost(X)
            ei = ei / cost.clamp_min(1e-8)
        return ei

    def _compute_cost(self, X: torch.Tensor) -> torch.Tensor:
        x_np = X.squeeze(1).detach().cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        cost = np.zeros(len(x_np))
        for i, v in enumerate(self.exploration_set):
            cost_fn = self.cost_functions.get(v, lambda val: np.ones_like(val, dtype=float))
            cost += cost_fn(x_np[:, i])
        return torch.tensor(cost, dtype=X.dtype, device=X.device)


class ManualCausalEI:
    """No-GP EI using do-calculus prior only. σ²=1+var_do, grid search."""

    def __init__(self, current_best: float, task: str,
                 mean_function: Callable, variance_function: Callable,
                 cost_functions: Dict = None, exploration_set: Tuple = None):
        self.current_best = current_best
        self.task = task
        self.mean_fn = mean_function
        self.var_fn = variance_function
        self.cost_functions = cost_functions
        self.exploration_set = exploration_set

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        mean = self.mean_fn(x)
        variance = np.clip(1.0 + self.var_fn(x), 0, None)
        std = np.sqrt(np.maximum(variance, 1e-6))
        if self.task == 'min':
            u = (self.current_best - mean) / std
        else:
            u = (mean - self.current_best) / std
        ei = std * (u * norm.cdf(u) + norm.pdf(u))
        if self.cost_functions and self.exploration_set:
            cost = np.zeros(x.shape[0])
            for i, v in enumerate(self.exploration_set):
                cost_fn = self.cost_functions.get(v, lambda val: np.ones_like(val, dtype=float))
                cost += cost_fn(x[:, i])
            ei = ei / np.maximum(cost[:, None], 1e-8)
        return ei
