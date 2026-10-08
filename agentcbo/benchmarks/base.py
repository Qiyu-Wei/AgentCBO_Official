from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..causal.dag import CausalGraph


class CausalBenchmark(ABC):

    @abstractmethod
    def get_sem(self) -> OrderedDict:
        ...

    @abstractmethod
    def get_ground_truth_dag(self) -> CausalGraph:
        ...

    @abstractmethod
    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        ...

    @abstractmethod
    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        ...

    @abstractmethod
    def get_noise_distributions(self) -> Dict[str, Callable]:
        ...

    @property
    @abstractmethod
    def target_variable(self) -> str:
        ...

    @property
    @abstractmethod
    def manipulative_variables(self) -> List[str]:
        ...

    @property
    def domain_knowledge(self) -> str:
        return ''

    def get_cost_functions(self) -> Dict[str, Callable]:
        return {}

    def sample_observational(self, n: int, seed: int = None) -> pd.DataFrame:
        rng = np.random.RandomState(seed if seed is not None else None)
        sem = self.get_sem()
        noise_dists = self.get_noise_distributions()
        data = {v: np.zeros(n) for v in sem}
        dag = self.get_ground_truth_dag()
        for v in sem:
            noise = np.array([noise_dists[v](rng) for _ in range(n)])
            parents = dag.get_parents(v)
            pa = {p: data[p] for p in parents}
            data[v] = sem[v](noise, pa)
        return pd.DataFrame(data)

    def sample_intervention(self, interventions: Dict[str, float],
                            n: int = 1, seed: int = None) -> pd.DataFrame:
        rng = np.random.RandomState(seed if seed is not None else None)
        sem = self.get_sem()
        noise_dists = self.get_noise_distributions()
        dag = self.get_ground_truth_dag()
        data = {v: np.zeros(n) for v in sem}
        for v in sem:
            if v in interventions:
                data[v] = np.full(n, interventions[v])
            else:
                noise = np.array([noise_dists[v](rng) for _ in range(n)])
                parents = dag.get_parents(v)
                pa = {p: data[p] for p in parents}
                data[v] = sem[v](noise, pa)
        return pd.DataFrame(data)

    def evaluate_intervention(self, interventions: Dict[str, float],
                              n_samples: int = 1000, seed: int = None) -> float:
        rng = np.random.RandomState(seed if seed is not None else None)
        sem = self.get_sem()
        noise_dists = self.get_noise_distributions()
        dag = self.get_ground_truth_dag()
        data = {v: np.zeros(n_samples) for v in sem}
        for v in sem:
            if v in interventions:
                data[v] = np.full(n_samples, interventions[v])
            else:
                noise = np.array([noise_dists[v](rng) for _ in range(n_samples)])
                parents = dag.get_parents(v)
                pa = {p: data[p] for p in parents}
                data[v] = sem[v](noise, pa)
        return float(np.mean(data[self.target_variable]))
