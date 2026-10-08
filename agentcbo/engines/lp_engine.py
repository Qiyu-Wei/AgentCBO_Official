from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from ..causal.llm_prior import LLMCausalPrior
from ..causal.pc_algorithm import PCAlgorithm
from ..llm.client import LLMClient
from .base import CausalDiscoveryEngine


class LPEngine(CausalDiscoveryEngine):
    """LLM Prior + PC fusion: PC discovers skeleton, LLM orients edges."""

    def __init__(self, llm_client: Optional[LLMClient],
                 min_samples_for_pc: int = 30,
                 pc_alpha: float = 0.05,
                 llm_confidence_threshold: float = 0.7):
        self.llm_prior = LLMCausalPrior(llm_client) if llm_client is not None else None
        self.llm = llm_client  # expose for metadata collection
        self.pc = PCAlgorithm(alpha=pc_alpha)
        self.min_samples_for_pc = min_samples_for_pc
        self.llm_confidence_threshold = llm_confidence_threshold
        self._edge_cache: Dict[Tuple[frozenset, str], Tuple[Tuple[str, ...], np.ndarray]] = {}

    @property
    def name(self) -> str:
        return "LP"

    def get_edge_confidence_matrix(self, variable_names: List[str],
                                   domain_knowledge: str = None) -> np.ndarray:
        key = (frozenset(variable_names), domain_knowledge or '')
        cached = self._edge_cache.get(key)
        if cached is None:
            if self.llm_prior is None:
                raise RuntimeError('LPEngine has no prior provider')
            prior = self.llm_prior.get_full_prior_matrix(variable_names, domain_knowledge)
            prior = self._resolve_cycles(np.asarray(prior, dtype=float).copy(), variable_names)
            self._edge_cache[key] = (tuple(variable_names), prior.copy())
            return prior.copy()
        order, prior = cached
        if list(order) == list(variable_names):
            return prior.copy()
        idx = [order.index(v) for v in variable_names]
        return prior[np.ix_(idx, idx)].copy()

    def discover(self, data: np.ndarray, variable_names: List[str],
                 domain_knowledge: str = None) -> Tuple[CausalGraph, float]:
        n = len(data)
        llm_prior = self.get_edge_confidence_matrix(variable_names, domain_knowledge)
        low_confidence_pairs = self._low_confidence_pairs(llm_prior, variable_names)

        pc_cg = None
        pc_adj = None
        llm_added = set()
        if n >= self.min_samples_for_pc:
            bk = self._build_background_knowledge(llm_prior, variable_names)
            graph, cpdag_adj, pc_cg = self.pc.run(data, variable_names, background_knowledge=bk)
            pc_adj = graph.dag.get_adjacency_matrix().copy()
            graph, llm_added = self._supplement_llm_edges(graph, cpdag_adj, llm_prior, pc_cg, data, variable_names)
        else:
            graph = self._llm_only_graph(llm_prior, variable_names)

        confidence = self._compute_confidence(graph, llm_prior, pc_cg, n, llm_added)
        conflict_pairs = self._lp_pc_conflict_pairs(llm_prior, pc_adj, variable_names)
        # Instrumentation: record filter-stage statistics in graph metadata
        graph.metadata['lp_stats'] = {
            'n_bk_required': self._count_bk_required(llm_prior, variable_names),
            'n_pc_edges': graph.dag.n_edges() - len(llm_added),
            'n_llm_supplemented': len(llm_added),
            'n_total_edges': graph.dag.n_edges(),
            'confidence': confidence,
        }
        if hasattr(self.llm_prior, 'get_stats'):
            graph.metadata['llm_prior_stats'] = self.llm_prior.get_stats()
        graph.metadata['lp_prior_matrix'] = llm_prior.tolist()
        graph.metadata['lp_low_confidence_pairs'] = low_confidence_pairs
        graph.metadata['lp_conflict_pairs'] = conflict_pairs
        graph.metadata['lp_pc_adj'] = pc_adj.tolist() if pc_adj is not None else None
        graph.metadata['llm_added_edges'] = sorted([(int(i), int(j)) for (i, j) in llm_added])
        return graph, confidence

    def _low_confidence_pairs(self, llm_prior, variable_names):
        pairs = []
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                p_ij = float(llm_prior[i, j])
                p_ji = float(llm_prior[j, i])
                if max(p_ij, p_ji) < 0.8 or abs(p_ij - p_ji) < 0.25:
                    pairs.append((variable_names[i], variable_names[j]))
        return pairs

    def _lp_pc_conflict_pairs(self, llm_prior, pc_adj, variable_names):
        if pc_adj is None:
            return []
        conflicts = []
        thr = self.llm_confidence_threshold
        n = len(variable_names)
        for i in range(n):
            for j in range(i + 1, n):
                p_ij = float(llm_prior[i, j])
                p_ji = float(llm_prior[j, i])
                prior_dir = None
                if p_ij > thr and p_ji < 1 - thr:
                    prior_dir = (i, j)
                elif p_ji > thr and p_ij < 1 - thr:
                    prior_dir = (j, i)
                if prior_dir is None:
                    continue
                pc_dir = None
                if pc_adj[i, j]:
                    pc_dir = (i, j)
                elif pc_adj[j, i]:
                    pc_dir = (j, i)
                if pc_dir != prior_dir:
                    conflicts.append((variable_names[i], variable_names[j]))
        return conflicts

    def _count_bk_required(self, llm_prior, variable_names):
        """Count how many edges passed the BK threshold."""
        thr = self.llm_confidence_threshold
        count = 0
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                p_ij, p_ji = llm_prior[i, j], llm_prior[j, i]
                if (p_ij > thr and p_ji < 1 - thr) or (p_ji > thr and p_ij < 1 - thr):
                    count += 1
        return count

    def _build_background_knowledge(self, llm_prior, variable_names,
                                    confidence_threshold=None,
                                    forbidden_threshold=0.2):
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        bk = BackgroundKnowledge()
        nodes = [GraphNode(name) for name in variable_names]
        thr = self.llm_confidence_threshold if confidence_threshold is None else confidence_threshold
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                p_ij, p_ji = llm_prior[i, j], llm_prior[j, i]
                if p_ij > thr and p_ji < 1 - thr:
                    bk.add_required_by_node(nodes[i], nodes[j])
                    bk.add_forbidden_by_node(nodes[j], nodes[i])
                elif p_ji > thr and p_ij < 1 - thr:
                    bk.add_required_by_node(nodes[j], nodes[i])
                    bk.add_forbidden_by_node(nodes[i], nodes[j])
                elif p_ij < forbidden_threshold and p_ji < forbidden_threshold:
                    bk.add_forbidden_by_node(nodes[i], nodes[j])
                    bk.add_forbidden_by_node(nodes[j], nodes[i])
        return bk

    def _supplement_llm_edges(self, graph, cpdag_adj, llm_prior, pc_cg, data, variable_names,
                              alpha=None, supplement_threshold: float = 0.8):
        n = len(variable_names)
        adj = graph.dag.get_adjacency_matrix().copy()
        skel = (adj | adj.T)
        changed = False
        llm_added = set()
        for i in range(n):
            for j in range(i + 1, n):
                if skel[i, j] or skel[j, i]:
                    continue
                if llm_prior[i, j] > supplement_threshold or llm_prior[j, i] > supplement_threshold:
                    try:
                        from causallearn.utils.cit import CIT
                        ci_test = CIT(data, 'fisherz')
                        sepset = pc_cg.sepset
                        if sepset[i][j] is not None:
                            cond = list(sepset[i][j])
                            p_val = ci_test(i, j, cond)
                            if p_val > (self.pc.alpha if alpha is None else alpha) * 3:
                                continue
                    except Exception:
                        pass
                    if llm_prior[i, j] > llm_prior[j, i]:
                        src, dst = i, j
                    else:
                        src, dst = j, i
                    adj[src, dst] = 1
                    if PCAlgorithm._has_cycle(adj, n):
                        adj[src, dst] = 0
                    else:
                        changed = True
                        llm_added.add((src, dst))

        if changed:
            conf = graph.confidence_scores.copy()
            graph = CausalGraph(DAG.from_adjacency_matrix(adj), variable_names, conf)
        return graph, llm_added

    def _llm_only_graph(self, llm_prior, variable_names):
        n = len(variable_names)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if llm_prior[i, j] > 0.5:
                    edges.append((i, j, llm_prior[i, j]))
                elif llm_prior[j, i] > 0.5:
                    edges.append((j, i, llm_prior[j, i]))
        edges.sort(key=lambda e: -e[2])
        adj = np.zeros((n, n), dtype=int)
        for src, dst, _ in edges:
            adj[src, dst] = 1
            if PCAlgorithm._has_cycle(adj, n):
                adj[src, dst] = 0
        conf = {(i, j): llm_prior[i, j] for i in range(n) for j in range(n) if adj[i, j]}
        return CausalGraph(DAG.from_adjacency_matrix(adj), variable_names, conf)

    def _resolve_cycles(self, llm_prior, variable_names):
        """Detect directed cycles in LLM prior and weaken the weakest edge."""
        n = len(variable_names)
        adj = np.zeros((n, n), dtype=int)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and llm_prior[i, j] > 0.6:
                    edges.append((i, j, llm_prior[i, j]))
        edges.sort(key=lambda e: -e[2])
        for src, dst, _ in edges:
            adj[src, dst] = 1
            if PCAlgorithm._has_cycle(adj, n):
                adj[src, dst] = 0
                llm_prior[src, dst] = 0.4  # weaken
        return llm_prior

    def _compute_confidence(self, graph, llm_prior, pc_cg, n, llm_added=None):
        data_term = n / (n + self.min_samples_for_pc)
        if llm_added is None:
            llm_added = set()
        if pc_cg is not None:
            adj = graph.dag.get_adjacency_matrix()
            nv = len(graph.variable_names)
            agree = total = 0
            for i in range(nv):
                for j in range(i + 1, nv):
                    if (i, j) in llm_added:
                        continue
                    if adj[i, j] or adj[j, i]:
                        total += 1
                        if adj[i, j] and llm_prior[i, j] > 0.5:
                            agree += 1
                        elif adj[j, i] and llm_prior[j, i] > 0.5:
                            agree += 1
            agreement = agree / max(total, 1)
            return float(np.clip(0.5 * data_term + 0.5 * agreement, 0.1, 0.95))
        else:
            avg_conf = float(np.mean([max(llm_prior[i, :]) for i in range(llm_prior.shape[0])]))
            return float(np.clip(0.3 * data_term + 0.3 * avg_conf, 0.1, 0.6))
