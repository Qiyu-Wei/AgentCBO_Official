from __future__ import annotations
from typing import List, Tuple

import numpy as np

from ..causal.dag import CausalGraph
from ..causal.pc_algorithm import PCAlgorithm
from .base import CausalDiscoveryEngine


class PCOnlyEngine(CausalDiscoveryEngine):
    """Pure statistical causal discovery (no LLM). Used as AgentCBO baseline."""

    def __init__(self, alpha: float = 0.05):
        self.pc = PCAlgorithm(alpha=alpha)

    @property
    def name(self) -> str:
        return "PC-Only"

    def discover(self, data: np.ndarray, variable_names: List[str],
                 domain_knowledge: str = None) -> Tuple[CausalGraph, float]:
        graph, cpdag_adj, _ = self.pc.run(data, variable_names)
        n = len(variable_names)
        n_edges = graph.dag.n_edges()
        max_edges = n * (n - 1) / 2
        edge_density = n_edges / max(max_edges, 1)
        density_score = 1.0 - abs(edge_density - 0.3) / 0.7
        # oriented ratio
        n_undir = int(np.sum((cpdag_adj == -1) & (cpdag_adj.T == -1)) / 2)
        oriented_ratio = max(0.0, 1.0 - n_undir / max(n_edges, 1)) if n_edges > 0 else 0.0
        data_score = min(len(data) / 100.0, 1.0)
        confidence = float(np.clip(
            0.3 * density_score + 0.3 * oriented_ratio + 0.4 * data_score, 0.1, 0.9))
        return graph, confidence
