from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.spatial import ConvexHull, QhullError

from ..acquisition.causal_acqf import CausalExpectedImprovement, ManualCausalEI
from ..causal.dag import CausalGraph
from ..models.causal_gp import CausalGP
from ..models.conditional_gps import ConditionalGPNetwork
from ..models.do_calculus import DoCalculusComputer


@dataclass
class OptResult:
    best_x: Dict[str, float]
    best_y: float
    best_es: Tuple[str, ...]
    y_history: List[float]
    best_y_trace: List[float]
    cost_trace: List[float]
    discovered_graphs: List[CausalGraph]
    graph_metrics_trace: List[Dict]
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_intervention_grid(limits: List[Tuple[float, float]], n_per_dim: int,
                              max_total: int = 50000) -> np.ndarray:
    d = len(limits)
    if d == 1:
        return np.linspace(limits[0][0], limits[0][1], n_per_dim).reshape(-1, 1)
    # Reduce n_per_dim to keep total points under max_total
    effective_n = max(2, min(n_per_dim, int(max_total ** (1.0 / d))))
    grids = [np.linspace(lo, hi, effective_n) for lo, hi in limits]
    mesh = np.meshgrid(*grids, indexing='ij')
    return np.column_stack([m.ravel() for m in mesh])


class BaseCBO(ABC):
    """Per-ES GP + do-calculus prior + coverage-based epsilon."""

    def __init__(self, benchmark, task: str = 'min',
                 n_obs_init: int = 100, num_trials: int = 50,
                 num_anchor_points: int = 100, max_N: int = 200,
                 max_grid_total: int = 50000,
                 acqf_restarts: int = 5, acqf_raw_samples: int = None,
                 n_obs_per_step: int = 20, seed: int = None,
                 epsilon_cap: float = 0.3):
        self.benchmark = benchmark
        self.task = task
        self.n_obs_init = n_obs_init
        self.num_trials = num_trials
        self.num_anchor_points = num_anchor_points
        # Record grid and acquisition settings with each result.
        self.max_grid_total = int(max_grid_total)
        self.acqf_restarts = int(acqf_restarts)
        self.acqf_raw_samples = int(acqf_raw_samples if acqf_raw_samples is not None
                                    else num_anchor_points)
        self.max_N = max_N
        self.n_obs_per_step = n_obs_per_step
        self.seed = seed
        self.epsilon_cap = float(np.clip(epsilon_cap, 0.0, 1.0))
        self.observational_samples: Optional[pd.DataFrame] = None
        self.conditional_gps: Optional[ConditionalGPNetwork] = None
        self.exploration_sets: List[Tuple[str, ...]] = []
        self.interventional_data_x: Dict[Tuple, Optional[np.ndarray]] = {}
        self.interventional_data_y: Dict[Tuple, Optional[np.ndarray]] = {}
        self.bo_models: Dict[Tuple, Optional[Any]] = {}
        self.mean_functions: Dict[Tuple, Callable] = {}
        self.variance_functions: Dict[Tuple, Callable] = {}
        self._obs_pool_offset: int = 0
        self._time_causal_discovery: float = 0.0
        self._time_gp_fitting: float = 0.0

    @abstractmethod
    def _get_causal_graph(self) -> CausalGraph:
        ...

    def _compute_exploration_sets(self, graph: CausalGraph) -> List[Tuple[str, ...]]:
        return graph.compute_exploration_sets(
            self.benchmark.target_variable, self.benchmark.manipulative_variables)

    def optimize(self) -> OptResult:
        import time
        t0 = time.time()
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.observational_samples = self.benchmark.sample_observational(
            self.n_obs_init, self.seed)
        t_cd = time.time()
        graph = self._get_causal_graph()
        self._time_causal_discovery += time.time() - t_cd
        # Filter graph to observable variables (handles hidden vars like CompleteGraph U1/U2)
        obs_vars = set(self.observational_samples.columns)
        if not set(graph.variable_names) <= obs_vars:
            common = [v for v in graph.variable_names if v in obs_vars]
            graph = graph.subgraph(common)
        self.exploration_sets = self._compute_exploration_sets(graph)
        t_gp = time.time()
        self._fit_conditional_gps(graph)
        self._time_gp_fitting += time.time() - t_gp
        self._update_do_functions(graph)
        self._init_intervention_data()
        result = self._optimization_loop(graph)
        result.total_time = time.time() - t0
        result.metadata['time_causal_discovery'] = self._time_causal_discovery
        result.metadata['time_gp_fitting'] = self._time_gp_fitting
        # Record the exploration family alongside the evaluation budget.
        n_es = len(self.exploration_sets)
        result.metadata['n_exploration_sets'] = n_es
        result.metadata['fidelity'] = {'num_anchor_points': self.num_anchor_points,
                                       'max_grid_total': self.max_grid_total,
                                       'acqf_restarts': self.acqf_restarts,
                                       'acqf_raw_samples': self.acqf_raw_samples}
        result.metadata['trials_per_exploration_set'] = self.num_trials / n_es if n_es else None
        return result

    def _optimization_loop(self, graph: CausalGraph) -> OptResult:
        current_best_y: Dict[Tuple, List[float]] = {es: [] for es in self.exploration_sets}
        current_best_x: Dict[Tuple, List] = {es: [] for es in self.exploration_sets}
        global_opt: List[float] = []
        cost_trace: List[float] = []
        y_history: List[float] = []
        cumulative_cost = 0.0
        type_trial: List[str] = []
        last_intervened_es: Optional[Tuple] = None
        rng = np.random.RandomState(self.seed)

        for step in range(self.num_trials):
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
                self._update_do_functions(graph)
                prior_opt = self._compute_do_prior_optimum()
                global_opt.append(global_opt[-1] if global_opt else prior_opt)
                cost_trace.append(cumulative_cost)
                y_history.append(np.nan)
            else:
                type_trial.append('i')
                current_global = self._find_current_global(current_best_y)

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
                global_opt.append(current_global)
                y_history.append(y_new)

        # Build result
        all_interventions = []
        for es in self.exploration_sets:
            if current_best_y[es]:
                for x, y in zip(current_best_x[es], current_best_y[es]):
                    all_interventions.append((es, x, y))

        if all_interventions:
            if self.task == 'min':
                best_idx = min(range(len(all_interventions)), key=lambda k: all_interventions[k][2])
            else:
                best_idx = max(range(len(all_interventions)), key=lambda k: all_interventions[k][2])
            best_es, best_x_arr, best_y_val = all_interventions[best_idx]
            best_x_dict = {var: float(best_x_arr[0, j]) for j, var in enumerate(best_es)}
        else:
            best_x_dict, best_y_val, best_es = {}, float('inf'), ()

        return OptResult(
            best_x=best_x_dict, best_y=best_y_val, best_es=best_es,
            y_history=y_history, best_y_trace=global_opt, cost_trace=cost_trace,
            discovered_graphs=[], graph_metrics_trace=[], total_time=0.0,
            metadata={'type_trial': type_trial, 'n_interventions': type_trial.count('i'),
                      'llm_calls': 0, 'llm_total_tokens': 0})

    def _init_intervention_data(self):
        for es in self.exploration_sets:
            self.interventional_data_x[es] = None
            self.interventional_data_y[es] = None
            self.bo_models[es] = None

    def _compute_coverage(self) -> float:
        manip_vars = self.benchmark.manipulative_variables
        obs_manip = self.observational_samples[manip_vars].values
        ranges = self.benchmark.get_intervention_ranges()
        if len(manip_vars) == 1:
            lo, hi = ranges[manip_vars[0]]
            total = hi - lo
            raw = (obs_manip.max() - obs_manip.min()) / total if total > 0 else 0.0
            return min(raw, 1.0)
        if len(obs_manip) < len(manip_vars) + 1:
            return 0.0
        try:
            hull = ConvexHull(obs_manip)
            hull_vol = hull.volume
        except (QhullError, ValueError):
            return 0.0
        total_vol = np.prod([ranges[v][1] - ranges[v][0] for v in manip_vars])
        raw = hull_vol / total_vol if total_vol > 0 else 0.0
        return min(raw, 1.0)

    def _collect_observations(self) -> pd.DataFrame:
        s = self.seed + self._obs_pool_offset if self.seed is not None else None
        self._obs_pool_offset += 1
        return self.benchmark.sample_observational(self.n_obs_per_step, seed=s)

    def _find_current_global(self, current_best_y: Dict) -> float:
        all_vals = [v for vals in current_best_y.values() for v in vals]
        if not all_vals:
            return self._compute_do_prior_optimum()
        return min(all_vals) if self.task == 'min' else max(all_vals)

    def _compute_do_prior_optimum(self) -> float:
        best = float('inf') if self.task == 'min' else float('-inf')
        ranges = self.benchmark.get_intervention_ranges()
        for es in self.exploration_sets:
            if es not in self.mean_functions:
                continue
            limits = [ranges[v] for v in es]
            grid = create_intervention_grid(limits, self.num_anchor_points, self.max_grid_total)
            means = self.mean_functions[es](grid)
            val = float(np.min(means)) if self.task == 'min' else float(np.max(means))
            best = min(best, val) if self.task == 'min' else max(best, val)
        return best

    def _append_intervention_data(self, es, x_new, y_new):
        y_arr = np.array([[y_new]])
        if self.interventional_data_x[es] is None:
            self.interventional_data_x[es] = x_new
            self.interventional_data_y[es] = y_arr
        else:
            self.interventional_data_x[es] = np.vstack([self.interventional_data_x[es], x_new])
            self.interventional_data_y[es] = np.vstack([self.interventional_data_y[es], y_arr])

    def _compute_intervention_cost(self, es, intervention) -> float:
        cost_fns = self.benchmark.get_cost_functions()
        return sum(cost_fns.get(v, lambda x: 1.0)(intervention[v]) for v in es)

    def _fit_conditional_gps(self, graph):
        self.conditional_gps = ConditionalGPNetwork(graph)
        self.conditional_gps.fit(self.observational_samples)

    def _refit_conditional_gps(self):
        import time
        t = time.time()
        self.conditional_gps.refit(self.observational_samples)
        self._time_gp_fitting += time.time() - t

    def _update_do_functions(self, graph):
        computer = DoCalculusComputer(
            graph, self.conditional_gps, self.observational_samples,
            self.benchmark.target_variable)
        for es in self.exploration_sets:
            self.mean_functions[es] = computer.get_mean_function(es)
            self.variance_functions[es] = computer.get_variance_function(es)

    def _update_bo_model(self, es: Tuple[str, ...]):
        data_x = self.interventional_data_x[es]
        data_y = self.interventional_data_y[es]
        if data_x is not None and len(data_x) > 0:
            ranges = self.benchmark.get_intervention_ranges()
            limits = [ranges[v] for v in es]
            bounds = torch.tensor(limits, dtype=torch.float64).T
            train_X = torch.tensor(data_x, dtype=torch.float64)
            train_Y = torch.tensor(data_y, dtype=torch.float64)

            model = CausalGP(
                train_X=train_X, train_Y=train_Y,
                variance_adjustment=self.variance_functions.get(es),
                mean_function=self.mean_functions.get(es),
                bounds=bounds)
            try:
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
            except Exception:
                pass
            self.bo_models[es] = model
        else:
            self.bo_models[es] = None

    def _evaluate_acquisition(self, es, current_best):
        ranges = self.benchmark.get_intervention_ranges()
        limits = [ranges[v] for v in es]
        cost_fns = self.benchmark.get_cost_functions()

        if self.bo_models[es] is not None:
            model = self.bo_models[es]
            model.eval()
            acq = CausalExpectedImprovement(
                model=model, best_f=current_best, task=self.task,
                cost_functions=cost_fns, exploration_set=es)
            bounds = torch.tensor(limits, dtype=torch.float64).T
            try:
                from botorch.optim import optimize_acqf
                x_new, y_acq = optimize_acqf(
                    acq, bounds=bounds, q=1,
                    num_restarts=self.acqf_restarts, raw_samples=self.acqf_raw_samples)
            except Exception:
                grid = create_intervention_grid(limits, self.num_anchor_points, self.max_grid_total)
                grid_t = torch.tensor(grid, dtype=torch.float64).unsqueeze(1)
                with torch.no_grad():
                    values = acq(grid_t)
                idx = values.argmax()
                x_new = grid_t[idx]
                y_acq = values[idx]
            return float(y_acq), x_new.detach().numpy().reshape(1, -1)
        else:
            acq = ManualCausalEI(
                current_best, self.task,
                self.mean_functions[es], self.variance_functions[es],
                cost_functions=cost_fns, exploration_set=es)
            grid = create_intervention_grid(limits, self.num_anchor_points, self.max_grid_total)
            values = acq.evaluate(grid).ravel()
            idx = np.argmax(values)
            return float(values[idx]), grid[idx:idx + 1]
