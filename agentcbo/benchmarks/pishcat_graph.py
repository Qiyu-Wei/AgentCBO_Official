from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class PISHCATGraph(CausalBenchmark):
    """
    Static 6-node PISHCAT cyber-defence benchmark adapted from DCBO.

    Variables:
      P: patch deployment intensity (manipulative, exogenous)
      I: intrusion response intensity (manipulative, exogenous)
      H: system hardening level
      A: attack impact
      C: patch cost
      Y: total defence cost (target, minimize)
    """

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['P', 'I']

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('P', lambda noise, pa: noise),
            ('I', lambda noise, pa: noise),
            ('H', lambda noise, pa: _sigmoid(2.0 * pa['P'] - 1.0) + noise),
            ('A', lambda noise, pa: np.exp(-pa['I']) * (1.0 - _sigmoid(2.0 * pa['P'] - 1.0)) + noise),
            ('C', lambda noise, pa: 0.5 * pa['P'] + 0.3 * _sigmoid(pa['H']) + noise),
            ('Y', lambda noise, pa: pa['C'] + pa['A'] + noise),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {
            'P': lambda rng: rng.uniform(0.0, 1.0),
            'I': lambda rng: rng.uniform(0.0, 3.0),
            'H': lambda rng: rng.normal(0.0, 0.1),
            'A': lambda rng: rng.normal(0.0, 0.1),
            'C': lambda rng: rng.normal(0.0, 0.05),
            'Y': lambda rng: rng.normal(0.0, 0.1),
        }

    def get_ground_truth_dag(self) -> CausalGraph:
        # P→{H,A,C}, I→A, H→C, A→Y, C→Y
        names = ['P', 'I', 'H', 'A', 'C', 'Y']
        parent_nodes = [
            [],        # P
            [],        # I
            [0],       # H ← P
            [0, 1],    # A ← P, I
            [0, 2],    # C ← P, H
            [3, 4],    # Y ← A, C
        ]
        return CausalGraph(DAG(parent_nodes, 6), names)

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        # POMIS from the benchmark specification. Current optimizers recompute
        # exploration sets from the active graph and do not enforce this directly.
        return [('P', 'I')]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'P': (0.0, 1.0),
            'I': (0.0, 3.0),
        }

    @property
    def domain_knowledge(self) -> str:
        return (
            'Cyber defence cost optimization. P represents patch deployment intensity, '
            'I represents intrusion response level. Higher patching hardens systems but '
            'costs more. Stronger intrusion response reduces attack impact.'
        )
