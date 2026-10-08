from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..causal.dag import CausalGraph, DAG
from ..causal.pc_algorithm import PCAlgorithm
from .base import CausalDiscoveryEngine


Pair = Tuple[int, int]
NamedPair = Tuple[str, str]


def _align_adj(graph: CausalGraph, variable_names: Sequence[str]) -> np.ndarray:
    adj = graph.dag.get_adjacency_matrix()
    if graph.variable_names == list(variable_names):
        return adj
    out = np.zeros((len(variable_names), len(variable_names)), dtype=int)
    index = {name: i for i, name in enumerate(graph.variable_names)}
    for i, src in enumerate(variable_names):
        for j, dst in enumerate(variable_names):
            if src in index and dst in index:
                out[i, j] = adj[index[src], index[dst]]
    return out


def _fit_sem(graph: CausalGraph, x: np.ndarray):
    out = []
    for j in range(x.shape[1]):
        pa = list(graph.dag.get_parents(j))
        X = np.column_stack([x[:, pa], np.ones(len(x))]) if pa else np.ones((len(x), 1))
        beta = np.linalg.lstsq(X, x[:, j], rcond=None)[0]
        resid = x[:, j] - X @ beta
        out.append((pa, beta, max(float(np.var(resid)), 1e-6)))
    return out


def _nll(params, x: np.ndarray) -> float:
    total = 0.0
    for j, (pa, beta, sig2) in enumerate(params):
        X = np.column_stack([x[:, pa], np.ones(len(x))]) if pa else np.ones((len(x), 1))
        resid = x[:, j] - X @ beta
        total += 0.5 * float(np.mean(np.log(2 * np.pi * sig2) + resid ** 2 / sig2))
    return total


def _score_graph(
    graph: CausalGraph,
    data: np.ndarray,
    holdout_ratio: float = 0.2,
    lambda_sp: float = 0.03,
) -> Tuple[float, Dict[str, float]]:
    """Deterministic holdout SEM likelihood with a sparsity penalty."""
    n = len(data)
    if n < 4:
        q = float(graph.dag.n_edges())
        return q, {
            'nll_ho': 0.0,
            'r_ci': 0.0,
            'sp': q,
            'n_edges': graph.dag.n_edges(),
            'fast_score': True,
        }
    m = int(round(n * holdout_ratio))
    m = min(max(m, 1), n - 1)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    d_va, d_tr = data[idx[:m]], data[idx[m:]]
    params = _fit_sem(graph, d_tr)
    nll = _nll(params, d_va)
    sp = float(graph.dag.n_edges() * np.log(max(n, 2)) / max(n, 2))
    q = nll + lambda_sp * sp
    return q, {
        'nll_ho': nll,
        'r_ci': 0.0,
        'sp': sp,
        'n_edges': graph.dag.n_edges(),
        'fast_score': True,
    }


class LPControlledRepairEngine(CausalDiscoveryEngine):
    """Calibrate the LP graph, then optionally repair remaining disagreements.

    Edits are restricted to residual pairs and must pass the held-out score,
    coverage, and exploration-set guards.
    """

    def __init__(
        self,
        lp_engine: CausalDiscoveryEngine,
        target_variable: str,
        manipulative_variables: Sequence[str],
        cc_engine: Optional[CausalDiscoveryEngine] = None,
        ra_engine: Optional[CausalDiscoveryEngine] = None,
        variant: str = 'LP-CC',
        holdout_ratio: float = 0.2,
        lambda_sp: float = 0.03,
        min_gain_cc: float = 0.02,
        min_gain_ra: float = 0.05,
        max_cc_edits: int = 3,
        max_ra_edits: int = 1,
        preserve_coverage: bool = True,
        preserve_exploration_sets: bool = True,
    ):
        if variant not in {'LP-CC', 'LP-CC-RA'}:
            raise ValueError(f'Unknown refinement variant: {variant}')
        if cc_engine is None:
            raise ValueError(f'{variant} requires a CC challenger')
        if variant == 'LP-CC-RA' and ra_engine is None:
            raise ValueError(f'{variant} requires an RA challenger')
        self.lp_engine = lp_engine
        self.cc_engine = cc_engine
        self.ra_engine = ra_engine
        self.target_variable = target_variable
        self.manipulative_variables = list(manipulative_variables)
        self.holdout_ratio = holdout_ratio
        self.lambda_sp = lambda_sp
        self.max_ra_edits = max_ra_edits
        self.llm = getattr(lp_engine, 'llm', None)
        self.variant = variant
        self.use_ra = variant == 'LP-CC-RA'
        self.min_gain_cc = min_gain_cc
        self.min_gain_ra = min_gain_ra
        self.max_cc_edits = max_cc_edits
        self.preserve_coverage = preserve_coverage
        self.preserve_exploration_sets = preserve_exploration_sets

    @property
    def name(self) -> str:
        return self.variant

    def discover(self, data: np.ndarray, variable_names: List[str],
                 domain_knowledge: str = None) -> Tuple[CausalGraph, float]:
        g0, c0, base_meta = self._lp_base_stage(data, variable_names, domain_knowledge)
        if g0 is None:
            raise RuntimeError(f'{self.variant} could not obtain a base graph')

        residual_pairs, backbone_pairs = self._build_residual_pairs(g0, variable_names)
        low_pairs = self._to_index_pairs(g0.metadata.get('lp_low_confidence_pairs', []), variable_names)
        conflict_pairs = self._to_index_pairs(g0.metadata.get('lp_conflict_pairs', []), variable_names)
        priority_key = lambda pair: self._priority_sort_key(pair, variable_names, conflict_pairs, low_pairs)

        base_score, base_parts = _score_graph(
            g0, data,
            holdout_ratio=self.holdout_ratio,
            lambda_sp=self.lambda_sp,
        )

        trace = [{
            'stage': 'lp',
            'accepted': True,
            'confidence': c0,
            'score': base_score,
            'score_parts': base_parts,
            'residual_size': len(residual_pairs),
            'source': base_meta['base_source'],
            'error': base_meta['lp_error'],
        }]

        current_graph = g0
        current_score = base_score
        current_parts = base_parts
        residual_remaining = set(residual_pairs)
        cc_applied = False
        ra_applied = False
        cc_candidate_score = None
        ra_candidate_score = None
        n_cc_projected = 0
        n_cc_kept = 0
        n_ra_projected = 0
        n_ra_kept = 0

        base_coverage = self._covered_manipulative(g0)
        if base_meta['lp_failed']:
            trace.append(self._skip_stage('repair', 'lp_failed', len(residual_remaining)))
        elif not base_coverage:
            # Preserve the outer loop's fallback when LP has no target coverage.
            trace.append(self._skip_stage('repair', 'lp_has_no_manipulative_ancestor_coverage',
                                          len(residual_remaining)))
        elif not residual_remaining:
            trace.append(self._skip_stage('repair', 'empty_residual_set', len(residual_remaining)))
        else:
            stage = self._run_stage(
                stage='cc',
                engine=self.cc_engine,
                data=data,
                variable_names=variable_names,
                domain_knowledge=domain_knowledge,
                base_graph=current_graph,
                base_score=current_score,
                residual_pairs=residual_remaining,
                backbone_pairs=backbone_pairs,
                candidate_min_gain=self.min_gain_cc,
                candidate_max_edits=self.max_cc_edits,
                candidate_priority_key=priority_key,
            )
            trace.append(stage['trace'])
            n_cc_projected = int(stage['projected_edits'])
            n_cc_kept = int(stage['kept_edits'])
            cc_candidate_score = stage['candidate_score']
            if stage['accepted']:
                current_graph = stage['graph']
                current_score = stage['candidate_score']
                current_parts = stage['score_parts']
                residual_remaining -= set(stage['kept_pairs'])
                cc_applied = True

            if self.use_ra and residual_remaining:
                stage = self._run_stage(
                    stage='ra',
                    engine=self.ra_engine,
                    data=data,
                    variable_names=variable_names,
                    domain_knowledge=domain_knowledge,
                    base_graph=current_graph,
                    base_score=current_score,
                    residual_pairs=residual_remaining,
                    backbone_pairs=backbone_pairs,
                    candidate_min_gain=self.min_gain_ra,
                    candidate_max_edits=self.max_ra_edits,
                    candidate_priority_key=priority_key,
                )
                trace.append(stage['trace'])
                n_ra_projected = int(stage['projected_edits'])
                n_ra_kept = int(stage['kept_edits'])
                ra_candidate_score = stage['candidate_score']
                if stage['accepted']:
                    current_graph = stage['graph']
                    current_score = stage['candidate_score']
                    current_parts = stage['score_parts']
                    residual_remaining -= set(stage['kept_pairs'])
                    ra_applied = True
            elif self.use_ra:
                trace.append(self._skip_stage('ra', 'empty_residual_set_after_cc', 0))

        metadata = {
            'controlled_variant': self.variant,
            **base_meta,
            'proposal_mode': 'full_graph_diff_projected_to_lp_residual',
            'lp_confidence': c0,
            'cc_applied': cc_applied,
            'ra_applied': ra_applied,
            'base_score': base_score,
            'base_score_parts': base_parts,
            'cc_candidate_score': cc_candidate_score,
            'ra_candidate_score': ra_candidate_score,
            'accepted_score': current_score,
            'accepted_score_parts': current_parts,
            'residual_size_initial': len(residual_pairs),
            'residual_size_final': len(residual_remaining),
            'n_cc_projected_edits': n_cc_projected,
            'n_cc_kept_edits': n_cc_kept,
            'n_ra_projected_edits': n_ra_projected,
            'n_ra_kept_edits': n_ra_kept,
            'max_cc_edits': self.max_cc_edits,
            'max_ra_edits': self.max_ra_edits,
            'min_gain_cc': self.min_gain_cc,
            'min_gain_ra': self.min_gain_ra,
            'frozen_backbone_edges': [
                self._named_pair(variable_names, i, j) for i, j in sorted(backbone_pairs)
            ],
            'controlled_trace': trace,
        }
        return self._annotate_graph(current_graph, metadata), c0

    def _run_stage(
        self,
        stage: str,
        engine: CausalDiscoveryEngine,
        data: np.ndarray,
        variable_names: Sequence[str],
        domain_knowledge: Optional[str],
        base_graph: CausalGraph,
        base_score: float,
        residual_pairs: Set[Pair],
        backbone_pairs: Set[Pair],
        candidate_min_gain: float,
        candidate_max_edits: Optional[int],
        candidate_priority_key,
    ) -> Dict[str, object]:
        raw_graph, raw_conf, error = self._safe(engine, data, list(variable_names), domain_knowledge)
        if raw_graph is None:
            return {
                'accepted': False,
                'graph': base_graph,
                'candidate_score': None,
                'score_parts': None,
                'projected_edits': 0,
                'kept_edits': 0,
                'kept_pairs': [],
                'trace': {
                    'stage': stage,
                    'accepted': False,
                    'reason': 'challenger_error',
                    'confidence': 0.0,
                    'score': None,
                    'score_parts': None,
                    'projected_edits': 0,
                    'kept_edits': 0,
                    'residual_before': len(residual_pairs),
                    'residual_after': len(residual_pairs),
                    'error': error,
                },
            }

        projected = self._project_candidate(
            base_graph,
            raw_graph,
            residual_pairs,
            backbone_pairs,
            variable_names,
            raw_conf,
            max_edits=candidate_max_edits,
            priority_key=candidate_priority_key,
        )
        projected_edits = int(projected['projected_count'])
        kept_edits = int(projected['kept_count'])
        if kept_edits == 0:
            return {
                'accepted': False,
                'graph': base_graph,
                'candidate_score': None,
                'score_parts': None,
                'projected_edits': projected_edits,
                'kept_edits': 0,
                'kept_pairs': [],
                'trace': {
                    'stage': stage,
                    'accepted': False,
                    'reason': 'no_projectable_safe_edits',
                    'confidence': raw_conf,
                    'score': None,
                    'score_parts': None,
                    'projected_edits': projected_edits,
                    'kept_edits': 0,
                    'changed_pairs': [],
                    'residual_before': len(residual_pairs),
                    'residual_after': len(residual_pairs),
                    'error': None,
                },
            }

        candidate = projected['graph']
        candidate_score, score_parts = _score_graph(
            candidate, data,
            holdout_ratio=self.holdout_ratio,
            lambda_sp=self.lambda_sp,
        )
        safety_ok, safety_reason = self._passes_cbo_safety(base_graph, candidate)
        score_ok = candidate_score < (base_score - candidate_min_gain)
        accepted = bool(score_ok and safety_ok)
        reason = 'accepted'
        if not score_ok:
            reason = 'score_gate_failed'
        elif not safety_ok:
            reason = safety_reason

        return {
            'accepted': accepted,
            'graph': candidate if accepted else base_graph,
            'candidate_score': candidate_score,
            'score_parts': score_parts,
            'projected_edits': projected_edits,
            'kept_edits': kept_edits,
            'kept_pairs': projected['kept_pairs'] if accepted else [],
            'trace': {
                'stage': stage,
                'accepted': accepted,
                'reason': reason,
                'confidence': raw_conf,
                'score': candidate_score,
                'score_parts': score_parts,
                'score_gain': base_score - candidate_score,
                'min_gain': candidate_min_gain,
                'projected_edits': projected_edits,
                'kept_edits': kept_edits,
                'changed_pairs': projected['changed_named_pairs'],
                'residual_before': len(residual_pairs),
                'residual_after': len(residual_pairs) - kept_edits if accepted else len(residual_pairs),
                'error': None,
            },
        }

    def _skip_stage(self, stage: str, reason: str, residual_size: int) -> Dict[str, object]:
        return {
            'stage': stage,
            'accepted': False,
            'reason': reason,
            'confidence': 0.0,
            'score': None,
            'score_parts': None,
            'projected_edits': 0,
            'kept_edits': 0,
            'residual_before': residual_size,
            'residual_after': residual_size,
            'error': None,
        }

    def _covered_manipulative(self, graph: CausalGraph) -> Set[str]:
        try:
            ancestors = graph.get_ancestors(self.target_variable)
        except Exception:
            return set()
        return set(self.manipulative_variables) & ancestors

    def _exploration_sets(self, graph: CausalGraph) -> Set[Tuple[str, ...]]:
        try:
            return set(graph.compute_exploration_sets(
                self.target_variable, self.manipulative_variables))
        except Exception:
            return set()

    def _passes_cbo_safety(self, base_graph: CausalGraph,
                           candidate_graph: CausalGraph) -> Tuple[bool, str]:
        if self.preserve_coverage:
            base_covered = self._covered_manipulative(base_graph)
            candidate_covered = self._covered_manipulative(candidate_graph)
            if not candidate_covered.issuperset(base_covered):
                return False, 'coverage_guard_failed'
        if self.preserve_exploration_sets:
            base_es = self._exploration_sets(base_graph)
            candidate_es = self._exploration_sets(candidate_graph)
            if len(candidate_es) < len(base_es):
                return False, 'exploration_set_guard_failed'
        return True, 'ok'

    def _safe(self, engine, data, variable_names, domain_knowledge):
        try:
            graph, confidence = engine.discover(data, variable_names, domain_knowledge)
            return graph, confidence, None
        except Exception as exc:
            return None, 0.0, f'{type(exc).__name__}: {exc}'

    def _named_pair(self, variable_names: Sequence[str], i: int, j: int) -> NamedPair:
        return tuple(sorted((variable_names[i], variable_names[j])))

    def _to_index_pairs(self, pairs: Iterable[Sequence[str]], variable_names: Sequence[str]) -> Set[Pair]:
        index = {name: i for i, name in enumerate(variable_names)}
        out = set()
        for pair in pairs:
            if len(pair) != 2:
                continue
            a, b = pair
            if a not in index or b not in index:
                continue
            i, j = sorted((index[a], index[b]))
            out.add((i, j))
        return out

    def _target_relevant_pairs(self, graph: CausalGraph, variable_names: Sequence[str]) -> Set[Pair]:
        if self.target_variable not in variable_names:
            return set()
        target_relevant = set(self.manipulative_variables)
        try:
            target_relevant.update(graph.get_ancestors(self.target_variable))
        except Exception:
            pass
        target_relevant.add(self.target_variable)
        pairs = set()
        for i, j in itertools.combinations(range(len(variable_names)), 2):
            ni, nj = variable_names[i], variable_names[j]
            if ni in target_relevant or nj in target_relevant:
                pairs.add((i, j))
        return pairs

    def _backbone_pairs(self, lp_graph: CausalGraph, variable_names: Sequence[str]) -> Set[Pair]:
        prior = lp_graph.metadata.get('lp_prior_matrix')
        pc_adj = lp_graph.metadata.get('lp_pc_adj')
        if prior is None or pc_adj is None:
            return set()
        prior = np.asarray(prior, dtype=float)
        pc_adj = np.asarray(pc_adj, dtype=int)
        out = set()
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                p_ij = float(prior[i, j])
                p_ji = float(prior[j, i])
                if p_ij >= 0.8 and p_ji <= 0.2 and pc_adj[i, j] == 1 and pc_adj[j, i] == 0:
                    out.add((i, j))
                elif p_ji >= 0.8 and p_ij <= 0.2 and pc_adj[j, i] == 1 and pc_adj[i, j] == 0:
                    out.add((i, j))
        return out

    def _build_residual_pairs(self, lp_graph: CausalGraph, variable_names: Sequence[str]) -> Tuple[Set[Pair], Set[Pair]]:
        low = self._to_index_pairs(lp_graph.metadata.get('lp_low_confidence_pairs', []), variable_names)
        conflicts = self._to_index_pairs(lp_graph.metadata.get('lp_conflict_pairs', []), variable_names)
        relevant = self._target_relevant_pairs(lp_graph, variable_names)
        backbone = self._backbone_pairs(lp_graph, variable_names)
        residual = (low | conflicts | relevant) - backbone
        return residual, backbone

    def _priority_sort_key(self, pair: Pair, variable_names: Sequence[str],
                           conflict_pairs: Set[Pair], low_pairs: Set[Pair]) -> Tuple[int, int, int, int, int, int]:
        i, j = pair
        ni, nj = variable_names[i], variable_names[j]
        target_flag = int(ni == self.target_variable or nj == self.target_variable)
        manip_count = int(ni in self.manipulative_variables) + int(nj in self.manipulative_variables)
        conflict_flag = int(pair in conflict_pairs)
        low_flag = int(pair in low_pairs)
        # Higher priority first, then deterministic fallback on indices.
        return (-conflict_flag, -target_flag, -manip_count, -low_flag, i, j)

    def _merge_confidence_scores(self, base_graph: CausalGraph, candidate_graph: CausalGraph,
                                 new_adj: np.ndarray, kept_pairs: Sequence[Pair],
                                 candidate_conf: float) -> Dict[Tuple[int, int], float]:
        conf = dict(base_graph.confidence_scores)
        cand_conf = getattr(candidate_graph, 'confidence_scores', {}) or {}
        for i, j in kept_pairs:
            conf.pop((i, j), None)
            conf.pop((j, i), None)
            if new_adj[i, j] == 1:
                conf[(i, j)] = float(cand_conf.get((i, j), candidate_conf))
            elif new_adj[j, i] == 1:
                conf[(j, i)] = float(cand_conf.get((j, i), candidate_conf))
        return conf

    def _project_candidate(self, base_graph: CausalGraph, candidate_graph: CausalGraph,
                           allowed_pairs: Set[Pair], backbone_pairs: Set[Pair],
                           variable_names: Sequence[str], candidate_conf: float,
                           max_edits: Optional[int] = None,
                           priority_key=None) -> Dict[str, object]:
        base_adj = _align_adj(base_graph, variable_names)
        cand_adj = _align_adj(candidate_graph, variable_names)
        new_adj = base_adj.copy()
        changed_pairs = []
        allowed = [p for p in allowed_pairs if p not in backbone_pairs]
        if priority_key is None:
            allowed = sorted(allowed)
        else:
            allowed = sorted(allowed, key=priority_key)
        projected_count = 0
        for i, j in allowed:
            base_state = (int(base_adj[i, j]), int(base_adj[j, i]))
            cand_state = (int(cand_adj[i, j]), int(cand_adj[j, i]))
            if base_state == cand_state:
                continue
            projected_count += 1
            if max_edits is not None and len(changed_pairs) >= max_edits:
                continue
            trial = new_adj.copy()
            trial[i, j] = 0
            trial[j, i] = 0
            if cand_state[0] == 1:
                trial[i, j] = 1
            elif cand_state[1] == 1:
                trial[j, i] = 1
            if PCAlgorithm._has_cycle(trial, trial.shape[0]):
                continue
            new_adj = trial
            changed_pairs.append((i, j))
        new_graph = CausalGraph(
            DAG.from_adjacency_matrix(new_adj),
            list(variable_names),
            self._merge_confidence_scores(base_graph, candidate_graph, new_adj, changed_pairs, candidate_conf),
            dict(base_graph.metadata),
        )
        return {
            'graph': new_graph,
            'projected_count': projected_count,
            'kept_count': len(changed_pairs),
            'kept_pairs': changed_pairs,
            'changed_named_pairs': [self._named_pair(variable_names, i, j) for i, j in changed_pairs],
        }

    def _lp_base_stage(self, data: np.ndarray, variable_names: Sequence[str], domain_knowledge: Optional[str]):
        g0, c0, lp_error = self._safe(self.lp_engine, data, list(variable_names), domain_knowledge)
        if g0 is not None:
            return g0, c0, {
                'lp_failed': False,
                'lp_error': None,
                'base_source': 'lp',
            }
        target = self.target_variable if self.target_variable in variable_names else variable_names[-1]
        fallback = CausalGraph.fully_connected(list(variable_names), target)
        return fallback, 0.0, {
            'lp_failed': True,
            'lp_error': lp_error,
            'base_source': 'lp_failed_fully_connected',
        }

    def _annotate_graph(self, graph: CausalGraph, metadata: Dict[str, object]) -> CausalGraph:
        graph.metadata.update(metadata)
        return graph
