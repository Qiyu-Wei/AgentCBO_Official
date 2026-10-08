from __future__ import annotations
import threading
from typing import Any, Dict, List, Tuple

import numpy as np

from .dag import DAG, CausalGraph


class PCAlgorithm:

    def __init__(self, alpha: float = 0.05, indep_test: str = 'fisherz'):
        self.alpha = alpha
        self.indep_test = indep_test

    def run(self, data: np.ndarray, variable_names: List[str],
            background_knowledge=None) -> Tuple[CausalGraph, np.ndarray, Any]:
        from causallearn.search.ConstraintBased.PC import pc

        cg = pc(data, self.alpha, self.indep_test,
                background_knowledge=background_knowledge,
                node_names=variable_names)
        cpdag_adj = cg.G.graph

        dag_adj = self._orient_cpdag(cg)
        confidence_scores = self._extract_confidence(cg, variable_names)
        graph = CausalGraph(DAG.from_adjacency_matrix(dag_adj), variable_names, confidence_scores)
        return graph, cpdag_adj, cg

    def _orient_cpdag(self, cg_result) -> np.ndarray:
        """CPDAG → DAG (binary adj[parent,child]=1). Timeout 5s with greedy fallback."""
        cl_adj = cg_result.G.graph
        n = cl_adj.shape[0]

        # Try pdag2dag with timeout
        dag_graph = self._pdag2dag_with_timeout(cg_result.G, timeout=5)
        if dag_graph is not None:
            return self._cl_to_binary(dag_graph.graph)

        # Greedy fallback
        binary = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if cl_adj[i, j] == -1 and cl_adj[j, i] == 1:
                    binary[i, j] = 1  # i→j already oriented
        # Undirected edges: orient low→high index, check acyclicity
        for i in range(n):
            for j in range(i + 1, n):
                if cl_adj[i, j] == -1 and cl_adj[j, i] == -1:
                    binary[i, j] = 1
                    if self._has_cycle(binary, n):
                        binary[i, j] = 0
                        binary[j, i] = 1
                        if self._has_cycle(binary, n):
                            binary[j, i] = 0  # drop edge
        return binary

    def _pdag2dag_with_timeout(self, general_graph, timeout=5):
        result = [None]

        def _run():
            try:
                from causallearn.utils.PDAG2DAG import pdag2dag
                result[0] = pdag2dag(general_graph)
            except Exception:
                pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        return result[0]

    @staticmethod
    def _cl_to_binary(cl_adj: np.ndarray) -> np.ndarray:
        n = cl_adj.shape[0]
        binary = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if cl_adj[i, j] == -1 and cl_adj[j, i] == 1:
                    binary[i, j] = 1
        return binary

    @staticmethod
    def _has_cycle(adj: np.ndarray, n: int) -> bool:
        in_deg = adj.sum(axis=0)
        q = [i for i in range(n) if in_deg[i] == 0]
        visited = 0
        while q:
            node = q.pop()
            visited += 1
            for j in range(n):
                if adj[node, j]:
                    in_deg[j] -= 1
                    if in_deg[j] == 0:
                        q.append(j)
        return visited < n

    @staticmethod
    def _extract_confidence(cg_result, variable_names: List[str]) -> Dict[Tuple[int, int], float]:
        """Extract edge confidence from PC sepset info."""
        scores = {}
        cl_adj = cg_result.G.graph
        n = len(variable_names)
        for i in range(n):
            for j in range(n):
                if cl_adj[i, j] == -1 and cl_adj[j, i] == 1:
                    scores[(i, j)] = 1.0  # oriented edge, full confidence
                elif cl_adj[i, j] == -1 and cl_adj[j, i] == -1:
                    scores[(i, j)] = 0.5  # undirected, uncertain
        return scores
