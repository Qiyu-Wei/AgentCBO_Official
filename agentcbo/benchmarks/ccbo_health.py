from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sample_truncated_standard_normal(rng, low: float, high: float) -> float:
    while True:
        sample = rng.normal(0.0, 1.0)
        if low <= sample <= high:
            return sample


class CCBOHealth(CausalBenchmark):
    """
    10-node health benchmark adapted from the cCBO health/PSA setting.
    Variables: Age, BMR, CI, Height, Weight, BMI, Aspirin, Statin, Cancer, Y
    Manipulative: Aspirin, Statin, CI
    Target: minimize Y
    """

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['Aspirin', 'Statin', 'CI']

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('Age',      lambda noise, pa: 55.0 + 20.0 * noise),
            ('BMR',      lambda noise, pa: 1500.0 + 10.0 * noise),
            ('CI',       lambda noise, pa: -100.0 + 200.0 * noise),
            ('Height',   lambda noise, pa: 175.0 + 10.0 * noise),
            ('Weight',   lambda noise, pa: (pa['BMR'] + 6.8 * pa['Age'] - 5.0 * pa['Height']) / 13.7
                                            + pa['CI'] * 150.0 / 7716.0 + noise),
            ('BMI',      lambda noise, pa: pa['Weight'] / (pa['Height'] / 100.0) ** 2),
            ('Aspirin',  lambda noise, pa: _sigmoid(-8.0 + 0.10 * pa['Age'] + 0.03 * pa['BMI']) + noise),
            ('Statin',   lambda noise, pa: _sigmoid(-13.0 + 0.10 * pa['Age'] + 0.20 * pa['BMI']) + noise),
            ('Cancer',   lambda noise, pa: _sigmoid(2.2 - 0.05 * pa['Age'] + 0.01 * pa['BMI']
                                                    - 0.04 * pa['Statin'] + 0.02 * pa['Aspirin']) + noise),
            ('Y',        lambda noise, pa: 6.8 + 0.04 * pa['Age'] - 0.15 * pa['BMI']
                                            - 0.60 * pa['Statin'] + 0.55 * pa['Aspirin']
                                            + 1.00 * pa['Cancer'] + 0.4 * noise),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {
            'Age':      lambda rng: rng.uniform(0.0, 1.0),
            'BMR':      lambda rng: _sample_truncated_standard_normal(rng, -1.0, 2.0),
            'CI':       lambda rng: rng.uniform(0.0, 1.0),
            'Height':   lambda rng: _sample_truncated_standard_normal(rng, -0.5, 0.5),
            'Weight':   lambda rng: rng.normal(0.0, 1.0),
            'BMI':      lambda rng: 0.0,
            'Aspirin':  lambda rng: rng.normal(0.0, 0.001),
            'Statin':   lambda rng: rng.normal(0.0, 0.001),
            'Cancer':   lambda rng: rng.normal(0.0, 0.001),
            'Y':        lambda rng: rng.normal(0.0, 1.0),
        }

    def get_ground_truth_dag(self) -> CausalGraph:
        # Age→{Weight,BMI,Aspirin,Statin,Cancer,Y}, BMR→Weight, CI→Weight,
        # Height→{Weight,BMI}, Weight→BMI, BMI→{Aspirin,Statin,Cancer,Y},
        # Aspirin→{Cancer,Y}, Statin→{Cancer,Y}, Cancer→Y
        names = ['Age', 'BMR', 'CI', 'Height', 'Weight', 'BMI', 'Aspirin', 'Statin', 'Cancer', 'Y']
        parent_nodes = [
            [],                # Age
            [],                # BMR
            [],                # CI
            [],                # Height
            [1, 2, 3, 0],      # Weight ← BMR, CI, Height, Age
            [4, 3, 0],         # BMI ← Weight, Height, Age
            [0, 5],            # Aspirin ← Age, BMI
            [0, 5],            # Statin ← Age, BMI
            [0, 5, 6, 7],      # Cancer ← Age, BMI, Aspirin, Statin
            [0, 5, 6, 7, 8],   # Y ← Age, BMI, Aspirin, Statin, Cancer
        ]
        return CausalGraph(DAG(parent_nodes, 10), names)

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [
            ('Aspirin',),
            ('Statin',),
            ('CI',),
            ('Aspirin', 'Statin'),
            ('Aspirin', 'CI'),
            ('Statin', 'CI'),
            ('Aspirin', 'Statin', 'CI'),
        ]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'Aspirin': (0.0, 1.0),
            'Statin': (0.0, 1.0),
            'CI': (-400.0, 400.0),
        }

    @property
    def domain_knowledge(self) -> str:
        return ('Medical screening benchmark with demographics, metabolism, calorie intake, '
                'anthropometrics, medication usage, cancer status, and PSA level.')
