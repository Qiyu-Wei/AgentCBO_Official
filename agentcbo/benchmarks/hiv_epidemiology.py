from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class HIVEpidemiology(CausalBenchmark):
    """
    5-node HIV epidemiology benchmark.
    Variables: B(behavioral risk), T(treatment), L(viral load), R(drug resistance), Y(outcome)
    Manipulative: T
    Target: minimize Y
    """

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['T']

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('B', lambda noise, pa: _sigmoid(3.0 * noise)),
            ('T', lambda noise, pa: _sigmoid(-3.0 + 2.0 * pa['B'] + noise)),
            ('L', lambda noise, pa: _sigmoid(2.0 - 3.0 * pa['T'] + 2.0 * pa['B'] + noise)),
            ('R', lambda noise, pa: _sigmoid(-2.0 + 2.0 * pa['T'] - 1.0 * pa['B'] + noise)),
            ('Y', lambda noise, pa: -1.0 * (np.cos(3.0 * pa['L']) + np.sin(3.0 * pa['R'])
                                            + pa['L'] + pa['R'] - 0.5 * pa['B'] + noise)),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {
            'B': lambda rng: rng.normal(0.0, 1.0),
            'T': lambda rng: rng.normal(0.0, 1.0),
            'L': lambda rng: rng.normal(0.0, 1.0),
            'R': lambda rng: rng.normal(0.0, 1.0),
            'Y': lambda rng: rng.normal(0.0, 0.1),
        }

    def get_ground_truth_dag(self) -> CausalGraph:
        # B→{T,L,R,Y}, T→{L,R}, L→Y, R→Y
        names = ['B', 'T', 'L', 'R', 'Y']
        parent_nodes = [
            [],          # B
            [0],         # T ← B
            [0, 1],      # L ← B, T
            [0, 1],      # R ← B, T
            [0, 2, 3],   # Y ← B, L, R
        ]
        return CausalGraph(DAG(parent_nodes, 5), names)

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [('T',)]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {'T': (0.0, 1.0)}

    @property
    def domain_knowledge(self) -> str:
        return ('HIV treatment benchmark with behavioral risk, treatment intensity, viral dynamics, '
                'drug resistance, and an aggregate health outcome.')
