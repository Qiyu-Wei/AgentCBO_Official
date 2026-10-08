from __future__ import annotations
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


class SyntheticBenchmark(CausalBenchmark):
    """Factory for the paper's chain benchmark."""

    def __init__(self, name, sem, noise, dag_parents, var_names, target, manip,
                 ranges, exploration_sets):
        self._name = name
        self._sem = sem
        self._noise = noise
        self._dag_parents = dag_parents
        self._var_names = var_names
        self._target = target
        self._manip = manip
        self._ranges = ranges
        self._es = exploration_sets

    @property
    def target_variable(self) -> str:
        return self._target

    @property
    def manipulative_variables(self) -> List[str]:
        return self._manip

    def get_sem(self) -> OrderedDict:
        return self._sem

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return self._noise

    def get_ground_truth_dag(self) -> CausalGraph:
        return CausalGraph(DAG(self._dag_parents, len(self._var_names)), self._var_names)

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return self._es

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return self._ranges

    @staticmethod
    def chain(n_vars: int = 5) -> SyntheticBenchmark:
        """X0 → X1 → ... → X_{n-2} → Y"""
        names = [f'X{i}' for i in range(n_vars - 1)] + ['Y']

        def make_sem():
            sem = OrderedDict()
            sem[names[0]] = lambda n, pa: n
            for i in range(1, n_vars - 1):
                if i % 2 == 1:
                    sem[names[i]] = (lambda idx: lambda n, pa: 2 * np.sin(pa[names[idx - 1]]) + n)(i)
                else:
                    sem[names[i]] = (lambda idx: lambda n, pa: np.exp(-pa[names[idx - 1]]) + n)(i)
            sem['Y'] = lambda n, pa: np.cos(pa[names[-2]]) - np.exp(-pa[names[-2]] / 20.0) + n
            return sem

        noise = {names[0]: lambda rng: rng.normal(0, 1)}
        for i in range(1, n_vars - 1):
            noise[names[i]] = lambda rng: rng.normal(0, 0.1)
        noise['Y'] = lambda rng: rng.normal(0, 0.1)

        parents = [[]] + [[i] for i in range(n_vars - 1)]
        manip = [names[0]]
        ranges = {names[0]: (-3.0, 3.0)}
        es = [(names[0],)]

        return SyntheticBenchmark('Chain', make_sem(), noise, parents, names, 'Y',
                                   manip, ranges, es)
