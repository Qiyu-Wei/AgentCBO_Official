from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


class ToyGraph(CausalBenchmark):
    """
    3-node chain: X → Z → Y (CBO classic benchmark)
    SEM: X=ε_x(U-5,5), Z=exp(-X)+ε_z(N0,0.1), Y=cos(Z)-exp(-Z/20)+ε_y(N0,0.1)
    Optimal: Z≈-3.20, Y≈-2.17
    """

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['X', 'Z']

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('X', lambda noise, pa: noise),
            ('Z', lambda noise, pa: np.exp(-pa['X']) + noise),
            ('Y', lambda noise, pa: np.cos(pa['Z']) - np.exp(-pa['Z'] / 20.0) + noise),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {
            'X': lambda rng: rng.uniform(-5, 5),
            'Z': lambda rng: rng.normal(0, 0.1),
            'Y': lambda rng: rng.normal(0, 0.1),
        }

    def get_ground_truth_dag(self) -> CausalGraph:
        dag = DAG([[], [0], [1]], 3)
        return CausalGraph(dag, ['X', 'Z', 'Y'])

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [('X',), ('Z',)]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {'X': (-5.0, 5.0), 'Z': (-5.0, 20.0)}

    @property
    def domain_knowledge(self) -> str:
        return 'Simple chain: X causes Z through exponential transform, Z causes Y through trigonometric transform.'
