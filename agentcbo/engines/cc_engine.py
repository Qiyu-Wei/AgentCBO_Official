from __future__ import annotations
import json
from typing import List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from ..causal.pc_algorithm import PCAlgorithm
from ..llm.client import LLMClient
from .base import CausalDiscoveryEngine

_FILTER_PROMPT = """You are an expert in causal discovery algorithm selection.

Given a dataset with the following properties:
- n_samples: {n_samples}
- n_variables: {n_vars}
- Variable names: {var_names}
- Linearity: {linearity}
- Gaussianity: {gaussianity}
Domain: {domain_knowledge}

Available algorithms: PC, GES, DirectLiNGAM
Select the top 2 most appropriate algorithms and explain why.

Respond as JSON: {{"selected": ["algo1", "algo2"], "reasoning": "..."}}"""

_HP_PROMPT = """Configure hyperparameters for {algorithm} on this dataset:
- n_samples: {n_samples}, n_vars: {n_vars}
- Linearity: {linearity}, Gaussianity: {gaussianity}

Respond as JSON with hyperparameter dict for {algorithm}.
For PC: {{"alpha": <float>}}
For GES: {{"score_func": "local_score_BIC"}}
For DirectLiNGAM: {{}}"""

class _PCWrapper:
    def __init__(self, data, variable_names, alpha=0.05, **kw):
        self.data = data
        self.names = variable_names
        self.alpha = alpha

    def run(self):
        pc = PCAlgorithm(alpha=self.alpha)
        graph, cpdag, _ = pc.run(self.data, self.names)
        return graph


class _GESWrapper:
    def __init__(self, data, variable_names, score_func='local_score_BIC', **kw):
        self.data = data
        self.names = variable_names
        self.score_func = score_func

    def run(self):
        from causallearn.search.ScoreBased.GES import ges
        record = ges(self.data, score_func=self.score_func, maxP=None)
        cl_adj = record['G'].graph
        binary = PCAlgorithm._cl_to_binary(cl_adj)
        # Orient remaining undirected
        n = binary.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if cl_adj[i, j] == -1 and cl_adj[j, i] == -1:
                    binary[i, j] = 1
                    if PCAlgorithm._has_cycle(binary, n):
                        binary[i, j] = 0
                        binary[j, i] = 1
                        if PCAlgorithm._has_cycle(binary, n):
                            binary[j, i] = 0  # drop edge
        return CausalGraph(DAG.from_adjacency_matrix(binary), self.names)


class _LiNGAMWrapper:
    def __init__(self, data, variable_names, **kw):
        self.data = data
        self.names = variable_names

    def run(self):
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
        model = DirectLiNGAM()
        model.fit(self.data)
        adj = (np.abs(model.adjacency_matrix_) > 0.1).astype(int).T  # lingam: adj[j,i]=coef means i→j
        return CausalGraph(DAG.from_adjacency_matrix(adj), self.names)


WRAPPER_REGISTRY = {'PC': _PCWrapper, 'GES': _GESWrapper, 'DirectLiNGAM': _LiNGAMWrapper}


class CCEngine(CausalDiscoveryEngine):
    """Causal-Copilot style 7-step pipeline: Filter→HP→Execute→Rerank→PostProcess."""

    def __init__(self, llm_client: LLMClient, n_bootstrap: int = 50,
                 bootstrap_threshold: float = 0.5):
        self.llm = llm_client
        self.n_bootstrap = n_bootstrap
        self.bootstrap_threshold = bootstrap_threshold

    @property
    def name(self) -> str:
        return "CC"

    def discover(self, data: np.ndarray, variable_names: List[str],
                 domain_knowledge: str = None) -> Tuple[CausalGraph, float]:
        n_samples, n_vars = data.shape
        linearity, gaussianity = self._detect_properties(data)

        # 1-2. Filter: LLM selects algorithms
        try:
            candidates = self._filter(variable_names, n_samples, n_vars,
                                       linearity, gaussianity, domain_knowledge)
        except Exception:
            candidates = ['PC', 'GES']

        # 3. Hyperparameters from LLM
        hyperparams = {}
        for algo in candidates:
            try:
                hyperparams[algo] = self._get_hyperparams(
                    algo, n_samples, n_vars, linearity, gaussianity)
            except Exception:
                hyperparams[algo] = {}

        # 4. Execute
        results = {}
        for algo in candidates:
            try:
                hp = hyperparams.get(algo, {})
                wrapper = WRAPPER_REGISTRY[algo](data, variable_names, **hp)
                results[algo] = wrapper.run()
            except Exception:
                continue

        if not results:
            pc = PCAlgorithm()
            graph, _, _ = pc.run(data, variable_names)
            return graph, 0.3

        # 5. Rerank: statistical scoring
        scores = {}
        for algo, graph in results.items():
            scores[algo] = self._score_graph(graph, data, variable_names)
        best_algo = max(scores, key=scores.get)
        best_graph = results[best_algo]

        # 6. PostProcess: bootstrap pruning + DAG enforcement
        best_graph = self._postprocess(best_graph, data, variable_names)

        confidence = float(np.clip(scores[best_algo], 0.1, 0.9))
        return best_graph, confidence

    def _filter(self, var_names, n_samples, n_vars, linearity, gaussianity, dk):
        prompt = _FILTER_PROMPT.format(
            n_samples=n_samples, n_vars=n_vars, var_names=var_names,
            linearity=linearity, gaussianity=gaussianity,
            domain_knowledge=dk or 'None')
        raw = self.llm.query(prompt)
        data = json.loads(raw)
        selected = [a for a in data['selected'] if a in WRAPPER_REGISTRY]
        return selected if selected else ['PC', 'GES']

    def _get_hyperparams(self, algo, n_samples, n_vars, linearity, gaussianity):
        prompt = _HP_PROMPT.format(
            algorithm=algo, n_samples=n_samples, n_vars=n_vars,
            linearity=linearity, gaussianity=gaussianity)
        raw = self.llm.query(prompt)
        return json.loads(raw)

    def _score_graph(self, graph, data, variable_names):
        n = len(variable_names)
        n_edges = graph.dag.n_edges()
        max_edges = n * (n - 1) / 2
        adj = graph.dag.get_adjacency_matrix()

        # 1. Statistical fit (R²): 0.3
        fit_score = 0.5
        try:
            total_r2, count = 0, 0
            for j in range(n):
                parents = [i for i in range(n) if adj[i, j]]
                if not parents:
                    continue
                X = data[:, parents]
                y = data[:, j]
                X_aug = np.column_stack([X, np.ones(len(X))])
                beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                ss_res = np.sum((y - X_aug @ beta) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                total_r2 += max(1 - ss_res / max(ss_tot, 1e-10), 0)
                count += 1
            fit_score = total_r2 / max(count, 1) if count > 0 else 0.5
        except Exception:
            pass

        # 2. Structural: density penalty + acyclicity: 0.2
        density = n_edges / max(max_edges, 1)
        density_score = 1.0 - abs(density - 0.3) / 0.7
        acyclic = 0.0 if PCAlgorithm._has_cycle(adj, n) else 1.0
        structural = 0.5 * density_score + 0.5 * acyclic

        # 3. Information: BIC-based score: 0.2
        info_score = 0.5
        try:
            n_samples = len(data)
            total_bic = 0
            for j in range(n):
                parents = [i for i in range(n) if adj[i, j]]
                y = data[:, j]
                if parents:
                    X = np.column_stack([data[:, p] for p in parents])
                    X_aug = np.column_stack([X, np.ones(n_samples)])
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                    rss = np.sum((y - X_aug @ beta) ** 2)
                else:
                    rss = np.sum((y - np.mean(y)) ** 2)
                k = len(parents) + 1
                bic = n_samples * np.log(max(rss / n_samples, 1e-10)) + k * np.log(n_samples)
                total_bic += bic
            # Normalize: lower BIC is better, map to [0,1] via sigmoid-like transform
            null_bic = sum(len(data) * np.log(max(np.var(data[:, j]), 1e-10)) + np.log(len(data))
                           for j in range(n))
            info_score = float(np.clip(1.0 / (1.0 + np.exp((total_bic - null_bic) / max(abs(null_bic), 1))), 0, 1))
        except Exception:
            pass

        # Fixed stability contribution; bootstrap pruning follows separately.
        stability = 0.5

        # Fixed domain contribution; no domain-knowledge scoring is applied here.
        domain = 0.5

        return 0.3 * fit_score + 0.2 * structural + 0.2 * info_score + 0.2 * stability + 0.1 * domain

    def _run_bootstrap(self, data, variable_names):
        n = len(variable_names)
        rng = np.random.RandomState(42)
        edge_counts = np.zeros((n, n))
        n_success = 0
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(data), len(data), replace=True)
            boot_data = data[idx]
            try:
                pc = PCAlgorithm(alpha=0.05)
                bg, _, _ = pc.run(boot_data, variable_names)
                edge_counts += bg.dag.get_adjacency_matrix()
                n_success += 1
            except Exception:
                continue
        if n_success == 0:
            return edge_counts, 0
        return edge_counts / n_success, n_success

    def _postprocess(self, graph, data, variable_names):
        """Bootstrap pruning + acyclicity enforcement."""
        n = len(variable_names)
        adj = graph.dag.get_adjacency_matrix()
        edge_freq, n_success = self._run_bootstrap(data, variable_names)
        if n_success == 0:
            return graph
        pruned = adj.copy()
        for i in range(n):
            for j in range(n):
                if adj[i, j] and edge_freq[i, j] < self.bootstrap_threshold:
                    pruned[i, j] = 0
        # Enforce acyclicity: remove lowest-frequency edge in each cycle
        while PCAlgorithm._has_cycle(pruned, n):
            min_freq, min_i, min_j = 1.0, -1, -1
            for i in range(n):
                for j in range(n):
                    if pruned[i, j] and edge_freq[i, j] < min_freq:
                        min_freq, min_i, min_j = edge_freq[i, j], i, j
            if min_i < 0:
                break
            pruned[min_i, min_j] = 0
        return CausalGraph(DAG.from_adjacency_matrix(pruned), variable_names)

    @staticmethod
    def _detect_properties(data):
        from scipy import stats
        linearity = 'likely_linear'
        gaussianity = 'likely_gaussian'
        for j in range(data.shape[1]):
            _, p = stats.shapiro(data[:min(500, len(data)), j])
            if p < 0.05:
                gaussianity = 'non_gaussian'
                break
        # Linearity: compare linear R² vs quadratic R² on correlated pairs
        n_vars = data.shape[1]
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                x, y = data[:, i], data[:, j]
                corr = np.corrcoef(x, y)[0, 1]
                if abs(corr) < 0.1:
                    continue
                coeffs1 = np.polyfit(x, y, 1)
                r1 = y - np.polyval(coeffs1, x)
                ss_res1 = np.sum(r1 ** 2)
                coeffs2 = np.polyfit(x, y, 2)
                r2 = y - np.polyval(coeffs2, x)
                ss_res2 = np.sum(r2 ** 2)
                if ss_res1 > 1e-10 and (ss_res1 - ss_res2) / ss_res1 > 0.1:
                    linearity = 'likely_nonlinear'
                    return linearity, gaussianity
        return linearity, gaussianity
