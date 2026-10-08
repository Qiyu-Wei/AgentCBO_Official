from __future__ import annotations
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from .dag import CausalGraph


class CausalGraphMetrics:

    @staticmethod
    def shd(pred: CausalGraph, true: CausalGraph) -> int:
        """pcalg 标准: 缺失+多余+反向各计1"""
        p = pred.dag.get_adjacency_matrix()
        t = true.dag.get_adjacency_matrix()
        n = t.shape[0]
        assert p.shape == t.shape
        extra = missing = reversed_e = 0
        for i in range(n):
            for j in range(i + 1, n):
                pe = p[i, j] or p[j, i]
                te = t[i, j] or t[j, i]
                if pe and not te:
                    extra += 1
                elif not pe and te:
                    missing += 1
                elif pe and te:
                    if (p[i, j] != t[i, j]) or (p[j, i] != t[j, i]):
                        reversed_e += 1
        return extra + missing + reversed_e

    @staticmethod
    def nshd(pred: CausalGraph, true: CausalGraph) -> float:
        n = true.dag.n_nodes
        return CausalGraphMetrics.shd(pred, true) / max(n * (n - 1) // 2, 1)

    @staticmethod
    def skeleton_f1(pred: CausalGraph, true: CausalGraph) -> Dict[str, float]:
        p = pred.dag.get_adjacency_matrix()
        t = true.dag.get_adjacency_matrix()
        n = t.shape[0]
        p_skel = (p | p.T).astype(bool)
        t_skel = (t | t.T).astype(bool)
        tp = fp = fn = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p_skel[i, j] and t_skel[i, j]:
                    tp += 1
                elif p_skel[i, j] and not t_skel[i, j]:
                    fp += 1
                elif not p_skel[i, j] and t_skel[i, j]:
                    fn += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return {'precision': prec, 'recall': rec, 'f1': f1}

    @staticmethod
    def arrow_f1(pred: CausalGraph, true: CausalGraph) -> Dict[str, float]:
        p = pred.dag.get_adjacency_matrix()
        t = true.dag.get_adjacency_matrix()
        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return {'precision': prec, 'recall': rec, 'f1': f1}

    @staticmethod
    def direction_accuracy(pred: CausalGraph, true: CausalGraph) -> float:
        p = pred.dag.get_adjacency_matrix()
        t = true.dag.get_adjacency_matrix()
        n = t.shape[0]
        correct = total = 0
        for i in range(n):
            for j in range(i + 1, n):
                p_edge = p[i, j] or p[j, i]
                t_edge = t[i, j] or t[j, i]
                if p_edge and t_edge:
                    total += 1
                    if p[i, j] == t[i, j] and p[j, i] == t[j, i]:
                        correct += 1
        return correct / max(total, 1)
