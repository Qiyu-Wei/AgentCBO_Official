from __future__ import annotations
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark


class ERGraphBenchmark(CausalBenchmark):

    def __init__(self, n_nodes: int = 10, edge_prob: float = None, seed: int = 0):
        self.n_nodes = int(n_nodes)
        self.edge_prob = 2.0 / self.n_nodes if edge_prob is None else float(edge_prob)
        self.seed = int(seed)
        self._variable_names = [f'X{i}' for i in range(self.n_nodes - 1)] + ['Y']
        self._target = 'Y'
        self._manipulative = self._variable_names[:-1]
        self._dag, self._weights = self._build_graph()

    @property
    def target_variable(self) -> str:
        return self._target

    @property
    def manipulative_variables(self) -> List[str]:
        return list(self._manipulative)

    def _build_graph(self) -> Tuple[DAG, Dict[int, np.ndarray]]:
        rng = np.random.RandomState(self.seed)
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if rng.random() < self.edge_prob:
                    adj[i, j] = 1
        target_idx = self.n_nodes - 1
        if not adj[:, target_idx].any():
            adj[rng.randint(0, target_idx), target_idx] = 1
        dag = DAG.from_adjacency_matrix(adj)
        weights = {}
        for node_idx in range(self.n_nodes):
            n_parents = len(dag.get_parents(node_idx))
            if n_parents == 0:
                continue
            mags = rng.uniform(0.5, 2.0, size=n_parents)
            signs = np.where(rng.random(size=n_parents) < 0.5, -1.0, 1.0)
            weights[node_idx] = mags * signs
        return dag, weights

    def get_sem(self) -> OrderedDict:
        sem = OrderedDict()
        for node_idx, name in enumerate(self._variable_names):
            parents = [self._variable_names[p] for p in self._dag.get_parents(node_idx)]
            if not parents:
                sem[name] = lambda noise, pa: noise
                continue
            w = self._weights[node_idx].copy()
            sem[name] = (
                lambda noise, pa, ps=tuple(parents), ws=w:
                sum(ws[k] * pa[ps[k]] for k in range(len(ps))) + noise
            )
        return sem

    def get_noise_distributions(self) -> Dict[str, Callable]:
        noise = {}
        for node_idx, name in enumerate(self._variable_names):
            scale = 1.0 if not self._dag.get_parents(node_idx) else 0.1
            noise[name] = lambda rng, s=scale: rng.normal(0.0, s)
        return noise

    def get_ground_truth_dag(self) -> CausalGraph:
        return CausalGraph(DAG.from_adjacency_matrix(self._dag.get_adjacency_matrix().copy()),
                           list(self._variable_names))

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [(v,) for v in self._manipulative]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {v: (-3.0, 3.0) for v in self._manipulative}

    @property
    def domain_knowledge(self) -> str:
        return (f'Erdos-Renyi linear ANM with {self.n_nodes} nodes. '
                'Edges point from lower-index to higher-index nodes only.')
