from __future__ import annotations
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from ..causal.dag import CausalGraph
from .conditional_gps import ConditionalGPNetwork


class DoCalculusComputer:
    """Compute E[Y|do(ES=x)] via chain propagation along DAG topological order."""

    def __init__(self, graph: CausalGraph, conditional_gps: ConditionalGPNetwork,
                 obs_data: pd.DataFrame, target: str):
        self.graph = graph
        self.gps = conditional_gps
        self.obs_data = obs_data
        self.target = target
        self._cache: Dict[Tuple, Tuple[float, float]] = {}

    def compute_do(self, exploration_set: Tuple[str, ...],
                   intervention_values: np.ndarray) -> Tuple[float, float]:
        iv = np.atleast_1d(intervention_values).astype(float)
        cache_key = (exploration_set, tuple(np.round(iv, 8)))
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_obs = len(self.obs_data)
        node_values = {v: self.obs_data[v].values.copy() for v in self.graph.variable_names}
        node_vars = {}

        for i, v in enumerate(exploration_set):
            node_values[v] = float(iv[i]) * np.ones(n_obs)
            node_vars[v] = 0.0

        for node_idx in self.graph.dag.topological_order():
            v_name = self.graph.variable_names[node_idx]
            if v_name in exploration_set:
                continue
            parents = self.graph.get_parents(v_name)
            if not parents or self.gps.conditional_gps.get(v_name) is None:
                node_vars[v_name] = 0.0
                continue
            parent_data = np.column_stack([node_values[p] for p in parents])
            mean, var = self.gps.predict(v_name, parent_data)
            mean_flat = mean.flatten()
            if np.any(np.isnan(mean_flat)):
                mean_flat = np.nan_to_num(mean_flat, nan=float(self.obs_data[v_name].mean()))
            node_values[v_name] = mean_flat
            var_flat = np.nan_to_num(var.flatten(), nan=0.0, posinf=100.0, neginf=0.0)
            node_vars[v_name] = float(np.mean(var_flat)) + sum(
                node_vars.get(p, 0.0) for p in parents)

        target_parents = self.graph.get_parents(self.target)
        if not target_parents:
            obs_y = self.obs_data[self.target].values
            result = (float(np.mean(obs_y)), float(np.var(obs_y)) + 1.0)
            self._cache[cache_key] = result
            return result

        target_input = np.column_stack([node_values[p] for p in target_parents])
        mean_do, var_do = self.gps.predict(self.target, target_input, include_noise=False)
        var_do = np.nan_to_num(var_do, nan=0.0, posinf=100.0, neginf=0.0)
        propagated_var = sum(node_vars.get(p, 0.0) for p in target_parents)
        result = (float(np.mean(mean_do)), float(np.mean(var_do)) + propagated_var)
        self._cache[cache_key] = result
        return result

    def get_mean_function(self, exploration_set: Tuple[str, ...]) -> Callable:
        def mean_fn(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            results = np.zeros(len(x))
            for i in range(len(x)):
                results[i], _ = self.compute_do(exploration_set, x[i])
            return results[:, np.newaxis]
        return mean_fn

    def get_variance_function(self, exploration_set: Tuple[str, ...]) -> Callable:
        def var_fn(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            results = np.zeros(len(x))
            for i in range(len(x)):
                _, results[i] = self.compute_do(exploration_set, x[i])
            return results[:, np.newaxis]
        return var_fn
