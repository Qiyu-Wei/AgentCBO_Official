from __future__ import annotations
import json
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..causal.dag import DAG, CausalGraph
from ..causal.pc_algorithm import PCAlgorithm
from ..llm.client import LLMClient
from .base import CausalDiscoveryEngine

_SYSTEM_PROMPT = """You are an expert causal discovery agent.

## Methodology
1. EXPLORE: describe_variables + compute_pairwise_correlations
2. REFINE the PC-initialized graph: test_conditional_independence to verify edges, then use add_edge/remove_edge only when the evidence justifies a change
3. ORIENT: check_collider for v-structures, check_confounder for common causes, orient_edge based on evidence
3.5 EVALUATE: compute_bic to check graph fit
4. VERIFY: check_acyclicity, compare with domain knowledge
5. FINALIZE: call finalize_graph with JSON adjacency matrix

## Causal Principles
- Correlation != Causation. Only orient edges with statistical evidence.
- Collider (X->Z<-Y): conditioning on Z opens X-Y path
- Confounder (X<-Z->Y): conditioning on Z blocks spurious X-Y correlation
- Prefer sparser graphs when evidence is ambiguous

{pc_initialization}

The PC algorithm has already run. Focus on refining: orient edges, check colliders, verify with CI tests.

Variables: {variable_names} ({n_samples} samples)
Domain: {domain_knowledge}

Respond with JSON only:
{{"thought": "...", "action": "tool_name", "action_input": {{...}}}}

If a tool takes no arguments, use:
{{"thought": "...", "action": "tool_name", "action_input": {{}}}}

To finish, return:
{{"thought": "...", "action": "finalize_graph", "action_input": {{"adjacency_matrix": [[...]], "variable_order": [...], "confidence": 0.0-1.0, "reasoning": "..."}}}}

{few_shot}
"""

_FEW_SHOT = """## Worked Example
For graphs with 10 variables or fewer, a compact refinement trace can look like this:
1. {"thought": "Inspect the PC-initialized graph before changing anything.", "action": "get_current_graph", "action_input": {}}
2. {"thought": "Rain and Sprinkler should stay disconnected if they are marginally independent, so test that directly.", "action": "test_conditional_independence", "action_input": {"X": "Rain", "Y": "Sprinkler", "cond_set": []}}
3. {"thought": "If the current orientations remain plausible, compute BIC to verify that the refined graph still fits the data well.", "action": "compute_bic", "action_input": {}}
4. {"thought": "The PC-initialized graph is acyclic, statistically consistent, and the fit is acceptable, so finalize it.", "action": "finalize_graph", "action_input": {"adjacency_matrix": [[0, 0, 1], [0, 0, 1], [0, 0, 0]], "variable_order": ["Rain", "Sprinkler", "WetGrass"], "confidence": 0.74, "reasoning": "PC already supplied the two-parent structure. The marginal CI check supports leaving Rain and Sprinkler disconnected, and the BIC check does not indicate a worse fit."}}
"""


class ReActStep(BaseModel):
    thought: str = ""
    action: str
    action_input: Dict[str, Any] = Field(default_factory=dict)


class RAEngine(CausalDiscoveryEngine):
    """ReAct agent for causal discovery with tool-assisted graph refinement."""

    def __init__(self, llm_client: LLMClient, max_iterations: int = 10):
        self.llm = llm_client
        self.max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "RA"

    def discover(self, data: np.ndarray, variable_names: List[str],
                 domain_knowledge: str = None) -> Tuple[CausalGraph, float]:
        self._data = data
        self._variable_names = variable_names
        self._pairwise_correlations = self._compute_pairwise_correlations(data, variable_names)
        self._top_correlations = self._rank_top_correlations(self._pairwise_correlations)
        self._current_adj = np.zeros((len(variable_names), len(variable_names)), dtype=int)
        self._pc_graph = None
        self._pc_initialized = False
        self._pc_confidence = 0.2

        try:
            pc = PCAlgorithm()
            self._pc_graph, _, _ = pc.run(data, variable_names)
            self._current_adj = self._pc_graph.dag.get_adjacency_matrix().copy()
            self._pc_initialized = True
            self._pc_confidence = 0.5
            pc_initialization = self._build_pc_initialization(self._pc_graph, self._pc_confidence)
        except Exception as e:
            pc_initialization = "\n".join([
                "## PC Initialization",
                f"Initial PC run failed: {e}",
                "Current graph starts empty. Use the tools to recover cautiously.",
            ])

        tools = self._create_tools(data, variable_names)
        system_prompt = _SYSTEM_PROMPT.format(
            variable_names=variable_names,
            n_samples=len(data),
            domain_knowledge=domain_knowledge or 'None',
            pc_initialization=pc_initialization,
            few_shot=_FEW_SHOT if len(variable_names) <= 10 else '',
        )

        history = []
        for _ in range(self.max_iterations):
            prompt = self._format_prompt(system_prompt, history)

            try:
                react_step = self.llm.query_structured(
                    prompt,
                    ReActStep,
                    system_prompt=system_prompt,
                )
                thought = react_step.thought.strip()
                action = react_step.action.strip()
                action_input = react_step.action_input
            except Exception:
                try:
                    response = self.llm.query(prompt, system_prompt=system_prompt)
                except Exception:
                    break
                thought, action, action_input = self._parse_react(response)

            if action == 'finalize_graph':
                return self._parse_final(action_input, variable_names)
            if action not in tools:
                history.append(('system', f'Unknown tool: {action}. Available: {list(tools.keys())}'))
                continue
            if self._is_duplicate(action, action_input, history):
                history.append(('system', 'Already performed. Try different approach.'))
                continue

            action_input_str = self._stringify_action_input(action_input)
            try:
                observation = tools[action](action_input)
            except Exception as e:
                observation = f'Error: {e}'
            history.append((thought, action, action_input_str, str(observation)))

        if self._pc_graph is not None:
            n = len(self._variable_names)
            if PCAlgorithm._has_cycle(self._current_adj, n):
                graph, pc_conf = self._fallback_pc(data, variable_names)
                graph.metadata['ra_source'] = 'pc_fallback_due_cycle'
            else:
                from agentcbo.causal.dag import DAG, CausalGraph
                graph = CausalGraph(
                    DAG.from_adjacency_matrix(self._current_adj),
                    self._variable_names)
                graph.metadata['ra_source'] = 'pc_with_edits'
                pc_conf = self._pc_confidence
        else:
            graph, pc_conf = self._fallback_pc(data, variable_names)
        history_conf = self._estimate_confidence(history)
        return graph, min(pc_conf, history_conf)

    def _create_tools(self, data, names) -> Dict[str, Callable]:
        n = len(names)

        def describe_variables(inp):
            stats = {}
            for i, v in enumerate(names):
                col = data[:, i]
                stats[v] = {
                    'mean': float(np.mean(col)),
                    'std': float(np.std(col)),
                    'min': float(np.min(col)),
                    'max': float(np.max(col)),
                }
            return json.dumps(stats, indent=2)

        def compute_pairwise_correlations(inp):
            return json.dumps(self._pairwise_correlations)

        def test_ci(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            x_idx = names.index(inp['X'])
            y_idx = names.index(inp['Y'])
            cond = [names.index(c) for c in inp.get('cond_set', [])]
            from causallearn.utils.cit import CIT
            ci_test = CIT(data, 'fisherz')
            p = ci_test(x_idx, y_idx, cond)
            corr = np.corrcoef(data[:, x_idx], data[:, y_idx])[0, 1]
            corr = float(np.nan_to_num(corr, nan=0.0))
            stat = abs(corr) * np.sqrt(max(len(data) - len(cond) - 3, 1))
            return json.dumps({
                'stat': round(float(stat), 4),
                'p_value': round(float(p), 4),
                'independent': p > 0.05,
            })

        def run_pc(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            alpha = inp.get('alpha', 0.05) if isinstance(inp, dict) else 0.05
            pc = PCAlgorithm(alpha=alpha)
            graph, _, _ = pc.run(data, names)
            adj = graph.dag.get_adjacency_matrix()
            self._current_adj = adj.copy()
            return json.dumps({'adjacency_matrix': adj.tolist(), 'n_edges': int(adj.sum())})

        def add_edge(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            i, j = names.index(inp['from']), names.index(inp['to'])
            self._current_adj[i, j] = 1
            return f'Added {inp["from"]} -> {inp["to"]}'

        def remove_edge(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            i, j = names.index(inp['from']), names.index(inp['to'])
            self._current_adj[i, j] = 0
            return f'Removed {inp["from"]} -> {inp["to"]}'

        def orient_edge(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            i, j = names.index(inp['from']), names.index(inp['to'])
            self._current_adj[i, j] = 1
            self._current_adj[j, i] = 0
            return f'Oriented {inp["from"]} -> {inp["to"]}'

        def get_current(inp):
            return json.dumps({'adjacency_matrix': self._current_adj.tolist()})

        def check_collider(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            a = names.index(inp['var1'])
            mid = names.index(inp['var_mid'])
            b = names.index(inp['var2'])
            is_collider = bool(
                self._current_adj[a, mid] and self._current_adj[b, mid]
                and not self._current_adj[mid, a] and not self._current_adj[mid, b]
            )
            return json.dumps({'is_collider': is_collider})

        def check_confounder(inp):
            if isinstance(inp, str):
                inp = json.loads(inp)
            a_idx = names.index(inp['var1'])
            b_idx = names.index(inp['var2'])
            candidates = []
            for k, v in enumerate(names):
                if k in (a_idx, b_idx):
                    continue
                if self._current_adj[k, a_idx] and self._current_adj[k, b_idx]:
                    candidates.append(v)
            return json.dumps({'confounders': candidates})

        def check_acyclicity(inp):
            return json.dumps({'acyclic': not PCAlgorithm._has_cycle(self._current_adj, n)})

        def compute_bic(inp):
            adj = self._current_adj
            if isinstance(inp, str):
                inp = json.loads(inp)
            if isinstance(inp, dict) and 'adjacency_matrix' in inp:
                adj = np.array(inp['adjacency_matrix'], dtype=int)
            n_samples = len(data)
            log_n = np.log(max(n_samples, 2))
            total_bic = 0.0
            per_variable_bic = {}
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
                bic = n_samples * np.log(max(rss / max(n_samples, 1), 1e-10)) + k * log_n
                total_bic += float(bic)
                per_variable_bic[names[j]] = round(float(bic), 4)
            return json.dumps({
                'total_bic': round(float(total_bic), 4),
                'per_variable_bic': per_variable_bic,
                'n_edges': int(adj.sum()),
                'lower_is_better': True,
            })

        return {
            'describe_variables': describe_variables,
            'compute_pairwise_correlations': compute_pairwise_correlations,
            'test_conditional_independence': test_ci,
            'run_pc_algorithm': run_pc,
            'add_edge': add_edge,
            'remove_edge': remove_edge,
            'orient_edge': orient_edge,
            'get_current_graph': get_current,
            'check_collider': check_collider,
            'check_confounder': check_confounder,
            'check_acyclicity': check_acyclicity,
            'compute_bic': compute_bic,
        }

    def _build_state_summary(self, history):
        steps = [h for h in history if len(h) >= 4]
        tools_used = []
        seen = set()
        for _, action, _, _ in steps:
            if action not in seen:
                seen.add(action)
                tools_used.append(action)

        ci_count = sum(1 for _, action, _, _ in steps if action == 'test_conditional_independence')
        bic_values = []
        for _, action, _, observation in steps:
            if action != 'compute_bic':
                continue
            parsed = self._safe_json_loads(observation)
            if isinstance(parsed, dict) and 'total_bic' in parsed:
                bic_values.append(round(float(parsed['total_bic']), 4))

        top_corr = ', '.join(self._top_correlations) if self._top_correlations else 'none'
        bic_trend = ' -> '.join(str(v) for v in bic_values[-5:]) if bic_values else 'not computed'
        edge_count = int(self._current_adj.sum())
        current_edges = self._format_edges(self._current_adj, self._variable_names, limit=20)

        return "\n".join([
            "## State Summary",
            f"- Step count: {len(steps)}/{self.max_iterations}",
            f"- Tools used: {', '.join(tools_used) if tools_used else 'none'}",
            f"- Top correlations: {top_corr}",
            f"- CI test count: {ci_count}",
            f"- BIC trend (lower is better): {bic_trend}",
            f"- Current edges ({edge_count}): {current_edges}",
        ])

    def _format_prompt(self, system_prompt, history):
        recent_items = []
        for h in history[-5:]:
            if h[0] == 'system':
                recent_items.append(f'System Observation: {h[1]}')
            else:
                recent_items.append(
                    f'Thought: {h[0]}\nAction: {h[1]}\nAction Input: {h[2]}\nObservation: {h[3]}'
                )
        recent_history = '\n\n'.join(recent_items) if recent_items else 'No prior actions yet.'
        return (
            f'{self._build_state_summary(history)}\n\n'
            f'## Recent History\n{recent_history}\n\n'
            'Choose exactly one next action and respond with one JSON object.'
        )

    def _parse_react(self, response):
        payload = self._safe_json_loads(response)
        if isinstance(payload, dict) and 'action' in payload:
            return (
                str(payload.get('thought', '')).strip(),
                str(payload.get('action', '_unparseable')).strip(),
                payload.get('action_input', {}),
            )

        thought = ''
        action = None
        action_input = '{}'
        m_thought = re.search(r'Thought:\s*(.+?)(?=\nAction:|\Z)', response, re.S)
        if m_thought:
            thought = m_thought.group(1).strip()
        m_action = re.search(r'Action:\s*(\S+)', response)
        if m_action:
            action = m_action.group(1).strip()
        m_input = re.search(r'Action Input:\s*(.+?)(?=\nThought:|\nObservation:|\Z)', response, re.S)
        if m_input:
            action_input = m_input.group(1).strip()
        if action is None:
            action = '_unparseable'
        return thought, action, action_input

    def _parse_final(self, action_input, variable_names):
        try:
            if isinstance(action_input, str):
                data = json.loads(action_input)
            else:
                data = action_input
            adj = np.array(data['adjacency_matrix'], dtype=int)
            order = data.get('variable_order')
            if order is not None:
                order = list(order)
                if sorted(order) != sorted(variable_names):
                    raise ValueError('finalize_graph variable_order does not match variables')
                if order != list(variable_names):
                    idx = [order.index(v) for v in variable_names]
                    adj = adj[np.ix_(idx, idx)]
            if adj.shape != (len(variable_names), len(variable_names)):
                raise ValueError('finalize_graph adjacency_matrix has wrong shape')
            if PCAlgorithm._has_cycle(adj, len(variable_names)):
                raise ValueError("finalize_graph returned cyclic graph")
            conf = float(data.get('confidence', 0.5))
            graph = CausalGraph(DAG.from_adjacency_matrix(adj), variable_names)
            graph.metadata['ra_source'] = 'react_finalized'
            if order is not None:
                graph.metadata['ra_variable_order'] = order
            if 'reasoning' in data:
                graph.metadata['ra_reasoning'] = data['reasoning']
        except Exception as exc:
            graph, pc_conf = self._fallback_pc(self._data, variable_names)
            graph.metadata.setdefault('ra_source', 'pc_fallback_after_finalize_error')
            graph.metadata['ra_finalize_error'] = f'{type(exc).__name__}: {exc}'
            return graph, pc_conf
        return graph, min(conf, 0.9)

    def _is_duplicate(self, action, action_input, history):
        action_input_str = self._stringify_action_input(action_input)
        for h in history:
            if len(h) >= 3 and h[1] == action and h[2] == action_input_str:
                return True
        return False

    def _fallback_pc(self, data, variable_names):
        try:
            pc = PCAlgorithm()
            graph, _, _ = pc.run(data, variable_names)
            graph.metadata['ra_source'] = 'pc_fallback'
            return graph, 0.4
        except Exception:
            graph = CausalGraph.fully_connected(variable_names, variable_names[-1])
            graph.metadata['ra_source'] = 'fully_connected_fallback'
            return graph, 0.2

    def _estimate_confidence(self, history):
        steps = [h for h in history if len(h) >= 4]
        total = len(steps)
        if total == 0:
            return 0.3
        success = sum(1 for h in steps if 'error' not in h[3].lower())
        ci_count = sum(1 for h in steps if h[1] == 'test_conditional_independence')
        n_pairs = len(self._variable_names) * (len(self._variable_names) - 1) / 2
        coverage = min(1.0, ci_count / max(n_pairs, 1))
        used_pc = self._pc_initialized or any(h[1] == 'run_pc_algorithm' for h in steps)
        score = 0.3 * (success / max(total, 1)) + 0.3 * coverage + 0.2 * float(used_pc) + 0.1
        return min(0.5, score)

    def _compute_pairwise_correlations(self, data, variable_names):
        result = {}
        if len(variable_names) < 2:
            return result
        corr = np.atleast_2d(np.corrcoef(data.T))
        corr = np.nan_to_num(corr, nan=0.0)
        for i in range(len(variable_names)):
            for j in range(i + 1, len(variable_names)):
                result[f'{variable_names[i]}-{variable_names[j]}'] = round(float(corr[i, j]), 4)
        return result

    def _rank_top_correlations(self, pairwise_correlations, top_k: int = 5):
        ranked = sorted(
            pairwise_correlations.items(),
            key=lambda kv: (abs(kv[1]), kv[0]),
            reverse=True,
        )
        return [f'{pair}: {value}' for pair, value in ranked[:top_k]]

    def _build_pc_initialization(self, graph, confidence):
        adj = graph.dag.get_adjacency_matrix()
        lines = [
            "## PC Initialization",
            f"Initial PC confidence baseline: {confidence:.2f}",
            f"Initial PC edges ({int(adj.sum())}): {self._format_edges(adj, self._variable_names, limit=20)}",
        ]
        if len(self._variable_names) <= 10:
            lines.append(f"Initial PC adjacency matrix: {adj.tolist()}")
        return "\n".join(lines)

    def _format_edges(self, adj, variable_names, limit=None):
        edges = []
        for i, src in enumerate(variable_names):
            for j, dst in enumerate(variable_names):
                if adj[i, j]:
                    edges.append(f'{src}->{dst}')
        if not edges:
            return 'none'
        if limit is not None and len(edges) > limit:
            shown = ', '.join(edges[:limit])
            return f'{shown}, ... (+{len(edges) - limit} more)'
        return ', '.join(edges)

    def _safe_json_loads(self, value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        fenced = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, re.S)
        if fenced:
            text = fenced.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _stringify_action_input(self, action_input):
        if isinstance(action_input, str):
            return action_input
        try:
            return json.dumps(action_input, sort_keys=True)
        except Exception:
            return str(action_input)
