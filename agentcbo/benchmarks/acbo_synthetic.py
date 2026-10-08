from __future__ import annotations
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


class ACBOSynthetic(CausalBenchmark):
    """ACBO paper synthetic functions wrapped as causal benchmarks."""

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
    def rosenbrock() -> ACBOSynthetic:
        return ACBOSynthetic(
            'Rosenbrock',
            OrderedDict([
                ('X0', lambda n, pa: n),
                ('X1', lambda n, pa: 0.6 * pa['X0'] + 0.4 * n),
                ('Y', lambda n, pa: -100 * (pa['X1'] - pa['X0'] ** 2) ** 2
                                     - (1 - pa['X0']) ** 2 + 10 + n),
            ]),
            {'X0': lambda rng: rng.uniform(0, 1),
             'X1': lambda rng: rng.uniform(0, 1),
             'Y': lambda rng: rng.normal(0, 0.01)},
            [[], [0], [0, 1]],
            ['X0', 'X1', 'Y'], 'Y', ['X0', 'X1'],
            {'X0': (0.0, 1.0), 'X1': (0.0, 1.0)},
            [('X0',), ('X1',)])

    @staticmethod
    def ackley() -> ACBOSynthetic:
        return ACBOSynthetic(
            'Ackley',
            OrderedDict([
                ('X0', lambda n, pa: n),
                ('X1', lambda n, pa: n),
                ('Y', lambda n, pa: 20 * np.exp(-0.2 * np.sqrt(np.abs(pa['X0'])))
                                     + np.exp(pa['X1']) - 20 - np.e + n),
            ]),
            {'X0': lambda rng: rng.uniform(-2, 2),
             'X1': lambda rng: rng.uniform(-2, 2),
             'Y': lambda rng: rng.normal(0, 0.01)},
            [[], [], [0, 1]],
            ['X0', 'X1', 'Y'], 'Y', ['X0', 'X1'],
            {'X0': (-2.0, 2.0), 'X1': (-2.0, 2.0)},
            [('X0',), ('X1',), ('X0', 'X1')])
