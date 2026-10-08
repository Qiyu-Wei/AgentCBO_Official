from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from ..causal.dag import CausalGraph


class CausalDiscoveryEngine(ABC):

    @abstractmethod
    def discover(self, data: np.ndarray, variable_names: List[str],
                 domain_knowledge: str = None) -> Tuple[CausalGraph, float]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
