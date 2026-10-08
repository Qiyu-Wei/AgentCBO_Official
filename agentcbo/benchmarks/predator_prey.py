from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


def _monod(x, rate: float, half_saturation: float):
    x = np.maximum(x, 0.0)
    return rate * x / (half_saturation + x + 1e-9)


class PredatorPrey(CausalBenchmark):
    """
    Static reduction of the rotifer-algae chemostat benchmark used by DCBO.

    Variables:
      M: nutrient supply in the inflow medium (manipulative)
      N: algae / phytoplankton density (manipulative)
      P: juvenile rotifer density
      Y: negative adult rotifer net density (target, minimize to maximize adults)

    This is not a literal port of the dynamic ODE benchmark. Instead it collapses the
    omitted feedback loop into Monod-style saturation plus dilution, so that the system
    fits the static CausalBenchmark interface.
    """

    beta = 5.0
    delta = 0.55
    r_P = 3.3
    K_P = 4.3
    r_B = 2.25
    K_B = 15.0

    algae_scale = 55.0
    algae_half_saturation = 35.0
    juvenile_scale = 1.3
    target_scale = 10.0

    @property
    def target_variable(self) -> str:
        return 'Y'

    @property
    def manipulative_variables(self) -> List[str]:
        return ['M', 'N']

    def get_sem(self) -> OrderedDict:
        return OrderedDict([
            ('M', lambda noise, pa: noise),
            ('N', lambda noise, pa: np.clip(
                self.algae_scale * pa['M'] / (self.algae_half_saturation + pa['M']) + noise,
                0.0, 60.0,
            )),
            ('P', lambda noise, pa: np.clip(
                self.juvenile_scale * _monod(pa['N'], self.r_P, self.K_P) + noise,
                0.0, 8.0,
            )),
            ('Y', lambda noise, pa: -self.target_scale * (
                self.beta * _monod(pa['P'], self.r_B, self.K_B) - self.delta * pa['P']
            ) + noise),
        ])

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {
            'M': lambda rng: rng.uniform(60.0, 100.0),
            'N': lambda rng: rng.normal(0.0, 3.0),
            'P': lambda rng: rng.normal(0.0, 0.15),
            'Y': lambda rng: rng.normal(0.0, 0.1),
        }

    def get_ground_truth_dag(self) -> CausalGraph:
        # M→N→P→Y
        names = ['M', 'N', 'P', 'Y']
        parent_nodes = [
            [],     # M
            [0],    # N ← M
            [1],    # P ← N
            [2],    # Y ← P
        ]
        return CausalGraph(DAG(parent_nodes, 4), names)

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        # Conceptual POMIS for the static reduction. Note: current optimizers recompute
        # exploration sets from the discovered graph and do not enforce this.
        return [('N',)]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'M': (60.0, 100.0),
            'N': (0.0, 60.0),
        }

    @property
    def domain_knowledge(self) -> str:
        return (
            'Static rotifer-algae chemostat surrogate. M is inflow nutrient supply, N is algae density, '
            'P is juvenile rotifer density, and Y is negative adult rotifer net density. '
            'Higher algae increase juvenile recruitment through a Monod response, while adult performance '
            'is a net balance between saturating recruitment and linear dilution.'
        )
