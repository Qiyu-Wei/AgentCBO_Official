from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..causal.dag import CausalGraph
from ..causal.graph_metrics import CausalGraphMetrics
from ..engines.base import CausalDiscoveryEngine
from .base_cbo import BaseCBO, OptResult


class AgentCBO(BaseCBO):
    """Engine-agnostic CBO: accepts any CausalDiscoveryEngine, updates graph dynamically."""

    def __init__(self, benchmark, engine: CausalDiscoveryEngine,
                 update_freq: int = 5, confidence_threshold: float = 0.5,
                 debug: bool = False, **kwargs):
        super().__init__(benchmark, **kwargs)
        self.engine = engine
        self.update_freq = update_freq
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        self.discovered_graphs: List[CausalGraph] = []
        self.gate_decisions: List[Dict] = []
        self._archived_data: Dict[Tuple, Tuple] = {}

    def _get_causal_graph(self) -> CausalGraph:
        data = self.observational_samples.values
        names = list(self.observational_samples.columns)
        raw_graph, confidence = self.engine.discover(
            data, names, self.benchmark.domain_knowledge)
        target = self.benchmark.target_variable
        manip = set(self.benchmark.manipulative_variables)
        ancestors = raw_graph.get_ancestors(target)
        covered = manip & ancestors
        used_fallback = False
        if not covered:
            graph = CausalGraph.fully_connected(names, target)
            used_fallback = True
        else:
            graph = raw_graph
        # Preserve discovery metadata through fallback
        graph.metadata['raw_n_edges'] = raw_graph.dag.n_edges()
        graph.metadata['raw_ancestors_of_target'] = sorted(raw_graph.get_ancestors(target))
        graph.metadata['covered_manip'] = sorted(covered)
        graph.metadata['coverage_ratio'] = len(covered) / max(len(manip), 1)
        graph.metadata['used_fallback'] = used_fallback
        graph.metadata['coverage_fallback_mode'] = (
            'fully_connected' if used_fallback else 'none')
        graph.metadata['confidence'] = confidence
        # Keep discovery details when coverage fallback replaces the graph.
        graph.metadata['raw_engine_meta'] = dict(raw_graph.metadata)
        if used_fallback:
            graph.metadata['raw_engine_meta_overridden_by'] = 'fully_connected_fallback'
        for key in (
            'lp_stats', 'llm_prior_stats', 'ra_source', 'ra_reasoning', 'ra_variable_order',
            'ra_finalize_error', 'controlled_variant', 'controlled_trace',
            'proposal_mode', 'lp_failed', 'base_source', 'lp_confidence',
            'cc_applied', 'ra_applied', 'base_score', 'cc_candidate_score',
            'ra_candidate_score', 'accepted_score', 'residual_size_initial',
            'residual_size_final', 'n_cc_projected_edits',
            'n_cc_kept_edits', 'n_ra_projected_edits', 'n_ra_kept_edits',
        ):
            if key in raw_graph.metadata:
                graph.metadata[key] = raw_graph.metadata[key]
                if used_fallback and key in {'ra_source', 'controlled_variant'}:
                    graph.metadata[f'{key}_overridden_by'] = 'fully_connected_fallback'
        self.discovered_graphs.append(graph)
        return graph

    def _optimization_loop(self, graph: CausalGraph) -> OptResult:
        current_graph = graph
        current_best_y: Dict[Tuple, List[float]] = {es: [] for es in self.exploration_sets}
        current_best_x: Dict[Tuple, List] = {es: [] for es in self.exploration_sets}
        # Preserve the best intervention when its exploration set is archived.
        global_best_y: float = float('inf') if self.task == 'min' else float('-inf')
        global_best_x: Optional[np.ndarray] = None
        global_best_es: Tuple = ()
        global_opt: List[float] = []
        cost_trace: List[float] = []
        y_history: List[float] = []
        cumulative_cost = 0.0
        type_trial: List[str] = []
        last_intervened_es: Optional[Tuple] = None
        graph_metrics_trace: List[Dict] = []
        debug_trace: List[Dict] = []
        rng = np.random.RandomState(self.seed)

        for step in range(self.num_trials):
            # Periodic graph update
            if step > 0 and step % self.update_freq == 0:
                names = list(self.observational_samples.columns)
                all_data = self.observational_samples.values
                import time as _time
                _t_cd = _time.time()
                try:
                    new_graph, confidence = self.engine.discover(
                        all_data, names, self.benchmark.domain_knowledge)
                except Exception:
                    new_graph, confidence = current_graph, 0.0
                self._time_causal_discovery += _time.time() - _t_cd

                if self._should_accept_graph(current_graph, new_graph, confidence):
                    current_graph = new_graph
                    self.discovered_graphs.append(new_graph)
                    old_es = set(self.exploration_sets)
                    self.exploration_sets = self._compute_exploration_sets(current_graph)
                    new_es = set(self.exploration_sets)
                    # Archive disappeared ES data
                    for es in old_es - new_es:
                        # Retain the best value before removing a set.
                        if current_best_y.get(es):
                            for x_arr, y_val in zip(current_best_x[es], current_best_y[es]):
                                if (self.task == 'min' and y_val < global_best_y) or \
                                   (self.task == 'max' and y_val > global_best_y):
                                    global_best_y = y_val
                                    global_best_x = x_arr
                                    global_best_es = es
                        if self.interventional_data_x.get(es) is not None:
                            self._archived_data[es] = (
                                self.interventional_data_x[es], self.interventional_data_y[es])
                        for d in (self.interventional_data_x, self.interventional_data_y,
                                  self.bo_models, self.mean_functions, self.variance_functions):
                            d.pop(es, None)
                        current_best_y.pop(es, None)
                        current_best_x.pop(es, None)
                    # Restore archived data for new ES
                    for es in new_es - old_es:
                        if es in self._archived_data:
                            ax, ay = self._archived_data.pop(es)
                            self.interventional_data_x[es] = ax
                            self.interventional_data_y[es] = ay
                            current_best_y[es] = ay.flatten().tolist()
                            current_best_x[es] = [ax[k:k+1] for k in range(len(ax))]
                        else:
                            self.interventional_data_x[es] = None
                            self.interventional_data_y[es] = None
                            current_best_y[es] = []
                            current_best_x[es] = []
                        self.bo_models[es] = None
                    # Refit everything
                    self._fit_conditional_gps(current_graph)
                    self._update_do_functions(current_graph)
                    for es in self.exploration_sets:
                        self._update_bo_model(es)
                    last_intervened_es = None

                    # Record graph metrics
                    try:
                        gt = self.benchmark.get_ground_truth_dag()
                        common_vars = sorted(set(gt.variable_names) & set(new_graph.variable_names))
                        gt_sub = gt.subgraph(common_vars)
                        ng_sub = new_graph.subgraph(common_vars)
                        metrics = {
                            'trial': step,
                            'shd': CausalGraphMetrics.shd(ng_sub, gt_sub),
                            'nshd': CausalGraphMetrics.nshd(ng_sub, gt_sub),
                            'skeleton_f1': CausalGraphMetrics.skeleton_f1(ng_sub, gt_sub)['f1'],
                            'arrow_f1': CausalGraphMetrics.arrow_f1(ng_sub, gt_sub)['f1'],
                            'direction_accuracy': CausalGraphMetrics.direction_accuracy(ng_sub, gt_sub),
                        }
                        graph_metrics_trace.append(metrics)
                        if self.debug:
                            lp_stats = new_graph.metadata.get('lp_stats', {})
                            debug_trace.append({
                                'event': 'graph_update', 'step': step,
                                'shd': metrics['shd'], 'n_es': len(self.exploration_sets),
                                'confidence': confidence,
                                **{f'lp_{k}': v for k, v in lp_stats.items()},
                            })
                    except Exception:
                        pass

            # Observe-intervene tradeoff
            if step == 0:
                do_observe = True
            elif step == 1:
                do_observe = False
            else:
                coverage = self._compute_coverage()
                rescale = len(self.observational_samples) / self.max_N
                epsilon = min(coverage / rescale, self.epsilon_cap) if rescale > 0 else 0.0
                do_observe = rng.uniform() < epsilon

            if do_observe:
                type_trial.append('o')
                new_obs = self._collect_observations()
                self.observational_samples = pd.concat(
                    [self.observational_samples, new_obs], ignore_index=True)
                self._refit_conditional_gps()
                self._update_do_functions(current_graph)
                prior_opt = self._compute_do_prior_optimum()
                global_opt.append(global_opt[-1] if global_opt else prior_opt)
                cost_trace.append(cumulative_cost)
                y_history.append(np.nan)
            else:
                type_trial.append('i')
                current_global = self._find_current_global(current_best_y)
                # Include the best value from archived exploration sets.
                if (self.task == 'min' and global_best_y < current_global) or \
                   (self.task == 'max' and global_best_y > current_global):
                    current_global = global_best_y

                prev_was_obs = len(type_trial) >= 2 and type_trial[-2] == 'o'
                if step > 0 and prev_was_obs:
                    for es in self.exploration_sets:
                        self._update_bo_model(es)
                elif last_intervened_es is not None:
                    self._update_bo_model(last_intervened_es)

                y_acq_list, x_new_list = {}, {}
                for es in self.exploration_sets:
                    y_acq, x_new = self._evaluate_acquisition(es, current_global)
                    y_acq_list[es] = y_acq
                    x_new_list[es] = x_new

                best_es = max(y_acq_list, key=y_acq_list.get)
                x_new = x_new_list[best_es]
                if self.debug:
                    debug_trace.append({
                        'event': 'intervene', 'step': step,
                        'chosen_es': str(best_es), 'n_es': len(self.exploration_sets),
                        'ei_values': {str(k): float(v) for k, v in y_acq_list.items()},
                    })
                intervention = {var: float(x_new[0, j]) for j, var in enumerate(best_es)}
                eval_seed = self.seed + 10000 + step if self.seed is not None else None
                y_new = self.benchmark.evaluate_intervention(intervention, seed=eval_seed)

                self._append_intervention_data(best_es, x_new, y_new)
                last_intervened_es = best_es
                cumulative_cost += self._compute_intervention_cost(best_es, intervention)
                cost_trace.append(cumulative_cost)
                current_best_x[best_es].append(x_new)
                current_best_y[best_es].append(y_new)
                current_global = self._find_current_global(current_best_y)
                # Include the best value from archived exploration sets.
                if (self.task == 'min' and global_best_y < current_global) or \
                   (self.task == 'max' and global_best_y > current_global):
                    current_global = global_best_y
                global_opt.append(current_global)
                y_history.append(y_new)

        # Build result
        all_interventions = []
        for es in self.exploration_sets:
            if current_best_y.get(es):
                for x, y in zip(current_best_x[es], current_best_y[es]):
                    all_interventions.append((es, x, y))
        # An archived set may still hold the best intervention.
        if global_best_x is not None:
            all_interventions.append((global_best_es, global_best_x, global_best_y))
        if all_interventions:
            if self.task == 'min':
                best_idx = min(range(len(all_interventions)), key=lambda k: all_interventions[k][2])
            else:
                best_idx = max(range(len(all_interventions)), key=lambda k: all_interventions[k][2])
            best_es, best_x_arr, best_y_val = all_interventions[best_idx]
            best_x_dict = {var: float(best_x_arr[0, j]) for j, var in enumerate(best_es)}
        else:
            best_x_dict, best_y_val, best_es = {}, float('inf'), ()

        # Collect initial graph metadata for analysis
        init_graph_meta = self.discovered_graphs[0].metadata if self.discovered_graphs else {}

        return OptResult(
            best_x=best_x_dict, best_y=best_y_val, best_es=best_es,
            y_history=y_history, best_y_trace=global_opt, cost_trace=cost_trace,
            discovered_graphs=self.discovered_graphs, graph_metrics_trace=graph_metrics_trace,
            total_time=0.0,
            metadata={'type_trial': type_trial, 'n_interventions': type_trial.count('i'),
                      'n_trials': self.num_trials, 'n_obs_init': self.n_obs_init,
                      'n_exploration_sets': len(self.exploration_sets),
                      'exploration_sets': [str(es) for es in self.exploration_sets],
                      'engine': self.engine.name,
                      'gate_decisions': list(self.gate_decisions),
                      'init_graph_meta': init_graph_meta,
                      'debug_trace': debug_trace,
                      **self._get_llm_metadata()})

    def _should_accept_graph(self, old_graph, new_graph, confidence) -> bool:
        accepted, reason, nshd = self._commit_decision(old_graph, new_graph, confidence)
        self.gate_decisions.append({'accepted': accepted, 'reason': reason,
                                    'confidence': confidence, 'ancestor_nshd': nshd})
        return accepted

    def _commit_decision(self, old_graph, new_graph, confidence):
        """Return acceptance, its reason, and ancestor-subgraph nSHD."""
        changed = self._graph_changed(old_graph, new_graph)
        if confidence < self.confidence_threshold:
            return False, 'confidence_below_threshold', None
        if not changed:
            return False, 'unchanged', None
        target = self.benchmark.target_variable
        new_ancestors = new_graph.get_ancestors(target)
        if not (set(self.benchmark.manipulative_variables) & new_ancestors):
            return False, 'no_manipulative_ancestor', None
        union_ancestors = (old_graph.get_ancestors(target)
                           | new_graph.get_ancestors(target) | {target})
        # Ensure both subgraphs have same node set
        common = sorted(union_ancestors & set(old_graph.variable_names) & set(new_graph.variable_names))
        if len(common) < 2:
            return True, 'ancestor_subgraph_too_small', None
        ancestor_shd = CausalGraphMetrics.shd(
            old_graph.subgraph(common), new_graph.subgraph(common))
        n = len(common)
        nshd = ancestor_shd / max(n * (n - 1) // 2, 1)
        required = min(self.confidence_threshold + 0.5 * nshd, 0.9)
        ok = confidence >= required
        return ok, 'nshd_threshold_passed' if ok else 'nshd_threshold_failed', nshd

    def _get_llm_metadata(self) -> Dict:
        if hasattr(self.engine, 'llm') and self.engine.llm is not None:
            return {'llm_calls': self.engine.llm.total_calls,
                    'llm_total_tokens': self.engine.llm.total_tokens}
        return {}

    def _graph_changed(self, old_graph, new_graph) -> bool:
        target = self.benchmark.target_variable
        union_ancestors = (old_graph.get_ancestors(target)
                           | new_graph.get_ancestors(target) | {target})
        common = sorted(union_ancestors & set(old_graph.variable_names) & set(new_graph.variable_names))
        if len(common) < 2:
            return False
        return CausalGraphMetrics.shd(
            old_graph.subgraph(common), new_graph.subgraph(common)) > 0
