from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class PSAGraph(CausalBenchmark):
    """
    6-node PSA screening graph.
    Variables: age, bmi, A(treatment), S(screening), cancer, Y(outcome)
    Manipulative: A, S
    Target: minimize Y
    """

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['A', 'S']

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('age',    lambda noise, pa: 55 + 20 * noise),
            ('bmi',    lambda noise, pa: 27.0 - 0.01 * pa['age'] + 0.7 * noise),
            ('A',      lambda noise, pa: _sigmoid(-8 + 0.10 * pa['age'] + 0.03 * pa['bmi']) + noise),
            ('S',      lambda noise, pa: _sigmoid(-13 + 0.10 * pa['age'] + 0.20 * pa['bmi']) + noise),
            ('cancer', lambda noise, pa: _sigmoid(2.2 - 0.05 * pa['age'] + 0.01 * pa['bmi']
                                                   - 0.04 * pa['S'] + 0.02 * pa['A']) + noise),
            ('Y',      lambda noise, pa: 6.8 + 0.04 * pa['age'] - 0.15 * pa['bmi']
                                          - 0.60 * pa['S'] + 0.55 * pa['A'] + 1.00 * pa['cancer']
                                          + 0.4 * noise),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {
            'age':    lambda rng: rng.uniform(0, 1),
            'bmi':    lambda rng: rng.normal(0, 1),
            'A':      lambda rng: rng.normal(0, 0.001),
            'S':      lambda rng: rng.normal(0, 0.001),
            'cancer': lambda rng: rng.normal(0, 0.001),
            'Y':      lambda rng: rng.normal(0, 1),
        }

    def get_ground_truth_dag(self) -> CausalGraph:
        # age→{bmi,A,S,cancer,Y}, bmi→{A,S,cancer,Y}, A→{cancer,Y}, S→{cancer,Y}, cancer→Y
        parent_nodes = [
            [],              # age
            [0],             # bmi ← age
            [0, 1],         # A ← age, bmi
            [0, 1],         # S ← age, bmi
            [0, 1, 2, 3],   # cancer ← age, bmi, A, S
            [0, 1, 2, 3, 4],# Y ← age, bmi, A, S, cancer
        ]
        dag = DAG(parent_nodes, 6)
        return CausalGraph(dag, ['age', 'bmi', 'A', 'S', 'cancer', 'Y'])

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [('A',), ('S',), ('A', 'S')]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {'A': (0.0, 1.0), 'S': (0.0, 1.0)}

    @property
    def domain_knowledge(self) -> str:
        return ('Prostate cancer screening model. age and bmi are exogenous. '
                'A is treatment, S is screening. Both affect cancer risk and outcome Y.')
