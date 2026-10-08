from __future__ import annotations

import unittest

import numpy as np
import torch

from agentcbo.models.causal_gp import CausalGP


class TestCausalGP(unittest.TestCase):
    def test_causalgp_unnormalizes_inputs_for_causal_prior(self):
        train_x = torch.tensor([[0.0, 10.0], [2.0, 20.0]], dtype=torch.float64)
        train_y = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 10.0], [2.0, 20.0]], dtype=torch.float64)

        mean_function = lambda x: np.asarray(x)[:, 0] + np.asarray(x)[:, 1] / 10.0
        variance_adjustment = lambda x: np.ones(len(np.asarray(x))) * 4.0

        model = CausalGP(
            train_X=train_x,
            train_Y=train_y,
            variance_adjustment=variance_adjustment,
            mean_function=mean_function,
            bounds=bounds,
        )

        x_norm = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        self.assertAlmostEqual(model.mean_module(x_norm).item(), 2.5, places=8)
        self.assertAlmostEqual(model.covar_module._var_adj(x_norm).item(), 4.0, places=8)
        self.assertIsNone(getattr(model, "outcome_transform", None))

    def test_causal_gp_keeps_y_in_original_scale(self):
        train_X = torch.tensor([[0.0, 0.0], [2.0, 4.0]], dtype=torch.float64)
        train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0], [2.0, 4.0]], dtype=torch.float64)

        def mean_function(x):
            x = np.asarray(x)
            return x[:, 0] + 2.0 * x[:, 1]

        def variance_adjustment(x):
            return np.full(len(np.asarray(x)), 9.0)

        model = CausalGP(
            train_X=train_X,
            train_Y=train_Y,
            variance_adjustment=variance_adjustment,
            mean_function=mean_function,
            bounds=bounds,
        )

        x_norm = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        self.assertFalse(hasattr(model, "outcome_transform"))
        self.assertAlmostEqual(model.mean_module(x_norm).item(), 5.0, places=6)
