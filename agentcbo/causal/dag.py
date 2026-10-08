from __future__ import annotations
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Bound the number of joint intervention sets while retaining singletons.
ES_FAMILY_BUDGET = 250


def arity_within_budget(n_relevant: int, budget: int = ES_FAMILY_BUDGET) -> int:
    """Largest r with sum_{k<=r} C(n_relevant, k) <= budget (at least 1)."""
    total, r = 0, 0
    while r < n_relevant:
        nxt = total + _comb(n_relevant, r + 1)
        if nxt > budget:
            break
        total, r = nxt, r + 1
    return max(r, 1)


def _comb(n: int, k: int) -> int:
    from math import comb
    return comb(n, k)


class DAG:
    """DAG stored as parent lists, with adj[parent, child] = 1."""

    def __init__(self, parent_nodes: List[List[int]], n_nodes: int):
        self.parent_nodes = [list(p) for p in parent_nodes]
        self.n_nodes = n_nodes
        self._children: Optional[List[List[int]]] = None

    def _build_children(self):
        self._children = [[] for _ in range(self.n_nodes)]
        for child, parents in enumerate(self.parent_nodes):
            for p in parents:
                self._children[p].append(child)

    def get_parents(self, node: int) -> List[int]:
        return self.parent_nodes[node]

    def get_children(self, node: int) -> List[int]:
        if self._children is None:
            self._build_children()
        return self._children[node]

    def topological_order(self) -> List[int]:
        in_deg = [len(p) for p in self.parent_nodes]
        q = deque(i for i, d in enumerate(in_deg) if d == 0)
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for c in self.get_children(n):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        if len(order) != self.n_nodes:
            raise ValueError(f"Graph has a cycle: only {len(order)}/{self.n_nodes} nodes reachable")
        return order

    def get_ancestors(self, node: int) -> Set[int]:
        visited = set()
        stack = list(self.parent_nodes[node])
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.parent_nodes[n])
        return visited

    def get_descendants(self, node: int) -> Set[int]:
        visited = set()
        stack = list(self.get_children(node))
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.get_children(n))
        return visited

    def get_adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        for child, parents in enumerate(self.parent_nodes):
            for p in parents:
                adj[p, child] = 1
        return adj

    def n_edges(self) -> int:
        return sum(len(p) for p in self.parent_nodes)

    @staticmethod
    def from_adjacency_matrix(adj: np.ndarray) -> DAG:
        n = adj.shape[0]
        parent_nodes = []
        for j in range(n):
            parent_nodes.append([i for i in range(n) if adj[i, j] == 1])
        return DAG(parent_nodes, n)


@dataclass
class CausalGraph:
    """A DAG with variable names, edge confidence, and discovery metadata."""
    dag: DAG
    variable_names: List[str]
    confidence_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _idx(self, name: str) -> int:
        return self.variable_names.index(name)

    def get_variable_index(self, var_name: str) -> int:
        return self._idx(var_name)

    def get_parents(self, var_name: str) -> List[str]:
        idx = self._idx(var_name)
        return [self.variable_names[p] for p in self.dag.get_parents(idx)]

    def get_children(self, var_name: str) -> List[str]:
        idx = self._idx(var_name)
        return [self.variable_names[c] for c in self.dag.get_children(idx)]

    def get_ancestors(self, var_name: str) -> Set[str]:
        idx = self._idx(var_name)
        return {self.variable_names[i] for i in self.dag.get_ancestors(idx)}

    def get_markov_blanket(self, var_name: str) -> List[str]:
        idx = self._idx(var_name)
        mb = set(self.dag.get_parents(idx))
        children = self.dag.get_children(idx)
        mb.update(children)
        for c in children:
            mb.update(self.dag.get_parents(c))
        mb.discard(idx)
        return [self.variable_names[i] for i in sorted(mb)]

    def compute_exploration_sets(self, target: str, manipulative_vars: List[str],
                                  max_subset_size: int = None) -> List[Tuple[str, ...]]:
        ancestors = self.get_ancestors(target)
        relevant = sorted(set(manipulative_vars) & ancestors)
        if not relevant:
            relevant = sorted(manipulative_vars)
        max_r = max_subset_size if max_subset_size is not None else len(relevant)
        max_r = min(max_r, arity_within_budget(len(relevant)))
        es = []
        for r in range(1, max_r + 1):
            for subset in itertools.combinations(relevant, r):
                es.append(subset)
        # Ensure every manipulative variable has a singleton ES, even if
        # the discovered graph did not place it among ancestors of target.
        for v in manipulative_vars:
            singleton = (v,)
            if singleton not in es:
                es.append(singleton)
        return es

    def subgraph(self, nodes) -> CausalGraph:
        node_list = list(nodes) if isinstance(nodes, (list, tuple)) else sorted(nodes, key=lambda n: self._idx(n))
        idx_map = {self._idx(n): i for i, n in enumerate(node_list)}
        old_indices = set(idx_map.keys())
        parent_nodes = []
        for n in node_list:
            old_idx = self._idx(n)
            new_parents = [idx_map[p] for p in self.dag.get_parents(old_idx) if p in old_indices]
            parent_nodes.append(new_parents)
        new_conf = {}
        for (i, j), v in self.confidence_scores.items():
            if i in idx_map and j in idx_map:
                new_conf[(idx_map[i], idx_map[j])] = v
        return CausalGraph(DAG(parent_nodes, len(node_list)), node_list, new_conf)

    @staticmethod
    def fully_connected(variable_names: List[str], target: str) -> CausalGraph:
        names = [v for v in variable_names if v != target] + [target]
        n = len(names)
        parent_nodes = [list(range(i)) for i in range(n)]
        return CausalGraph(DAG(parent_nodes, n), names)

    def to_adjacency_matrix(self) -> np.ndarray:
        """causallearn 编码: adj[i,j]=-1, adj[j,i]=1 → i→j"""
        n = self.dag.n_nodes
        adj = np.zeros((n, n), dtype=int)
        for child, parents in enumerate(self.dag.parent_nodes):
            for p in parents:
                adj[p, child] = -1
                adj[child, p] = 1
        return adj

    def to_dict(self) -> Dict:
        return {
            'adjacency_matrix': self.dag.get_adjacency_matrix().tolist(),
            'variable_names': self.variable_names,
            'confidence_scores': {f"{i},{j}": v for (i, j), v in self.confidence_scores.items()},
            'metadata': self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict) -> CausalGraph:
        adj = np.array(d['adjacency_matrix'])
        conf = {}
        for k, v in d.get('confidence_scores', {}).items():
            i, j = map(int, k.split(','))
            conf[(i, j)] = v
        return CausalGraph(DAG.from_adjacency_matrix(adj), d['variable_names'], conf,
                           d.get('metadata', {}))
