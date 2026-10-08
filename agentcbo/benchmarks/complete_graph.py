from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


class CompleteGraph(CausalBenchmark):
    """
    9-node dense graph with 2 hidden variables (U1, U2).
    Observable: F, A, B, C, D, E, Y
    Manipulative: B, D, E
    """
    hidden_variables = ['U1', 'U2']

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['B', 'D', 'E']

    @property
    def domain_knowledge(self) -> str:
        return (
            'Dense 9-node causal benchmark modeling a two-pathway regulatory network '
            'with latent confounders U1, U2. '
            'Pathway 1: F (exogenous source) -> A (quadratic transform with hidden U1 contribution) -> E '
            '(cosine modulation combined with C). '
            'Pathway 2: B (exogenous source with hidden U2 contribution) -> C (exponential decay of B) '
            '-> D (exponential decay of C). Pathway 2 also feeds E via C. '
            'Target Y receives inputs from D (cosine), E (sine), and both latent confounders U1, U2. '
            'Manipulative variables are B, D, E. Variable names use single letters '
            '(F, A, B, C, D, E, Y) with no domain-specific meaning beyond the described graph structure.'
        )

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('U1', lambda noise, pa: noise),
            ('U2', lambda noise, pa: noise),
            ('F',  lambda noise, pa: noise),
            ('A',  lambda noise, pa: pa['F'] ** 2 + pa['U1'] + noise),
            ('B',  lambda noise, pa: pa['U2'] + noise),
            ('C',  lambda noise, pa: np.exp(-pa['B']) + noise),
            ('D',  lambda noise, pa: np.exp(-pa['C']) / 10.0 + noise),
            ('E',  lambda noise, pa: np.cos(pa['A']) + pa['C'] / 10.0 + noise),
            ('Y',  lambda noise, pa: np.cos(pa['D']) - pa['D'] / 5.0
                                      + np.sin(pa['E']) - pa['E'] / 4.0
                                      + pa['U1'] + np.exp(-pa['U2']) + noise),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {k: lambda rng: rng.normal(0, 1) for k in
                ['U1', 'U2', 'F', 'A', 'B', 'C', 'D', 'E', 'Y']}

    def get_ground_truth_dag(self) -> CausalGraph:
        # Full DAG including hidden: U1→{A,Y}, U2→{B,Y}, F→A, A→E, B→C, C→{D,E}, D→Y, E→Y
        names = ['U1', 'U2', 'F', 'A', 'B', 'C', 'D', 'E', 'Y']
        parent_nodes = [
            [],        # U1
            [],        # U2
            [],        # F
            [2, 0],    # A ← F, U1
            [1],       # B ← U2
            [4],       # C ← B
            [5],       # D ← C
            [3, 5],    # E ← A, C
            [6, 7, 0, 1],  # Y ← D, E, U1, U2
        ]
        return CausalGraph(DAG(parent_nodes, 9), names)

    def get_observable_variable_names(self) -> List[str]:
        return ['F', 'A', 'B', 'C', 'D', 'E', 'Y']

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [('B',), ('D',), ('E',), ('B', 'D'), ('B', 'E'), ('D', 'E')]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {'B': (-5.0, 4.0), 'D': (-5.0, 5.0), 'E': (-6.0, 3.0)}

    def sample_observational(self, n, seed=None):
        df = super().sample_observational(n, seed)
        return df[self.get_observable_variable_names()]
