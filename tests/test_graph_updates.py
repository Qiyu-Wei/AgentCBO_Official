from __future__ import annotations
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from agentcbo.causal.dag import DAG, CausalGraph
from agentcbo.causal.pc_algorithm import PCAlgorithm
from agentcbo.engines.ra_engine import RAEngine


def test_cycle_detection():
    cyclic = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # A->B->C->A
    acyclic = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])  # A->B, A->C, B->C
    assert PCAlgorithm._has_cycle(cyclic, 3)
    assert not PCAlgorithm._has_cycle(acyclic, 3)


def test_parse_final_rejects_cycle():
    """_parse_final should fall back when LLM returns cyclic adjacency."""
    engine = RAEngine(llm_client=MagicMock(), max_iterations=1)
    engine._data = np.random.RandomState(0).randn(50, 3)
    engine._variable_names = ['A', 'B', 'C']
    cyclic_adj = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    graph, conf = engine._parse_final(
        {'adjacency_matrix': cyclic_adj,
         'variable_order': ['A', 'B', 'C'],
         'confidence': 0.9},
        ['A', 'B', 'C'])
    assert graph.metadata.get('ra_source', '').startswith('pc_fallback')


def test_parse_final_accepts_dag():
    """_parse_final should keep valid DAG as react_finalized."""
    engine = RAEngine(llm_client=MagicMock(), max_iterations=1)
    engine._data = np.random.RandomState(0).randn(50, 3)
    engine._variable_names = ['A', 'B', 'C']
    dag_adj = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    graph, conf = engine._parse_final(
        {'adjacency_matrix': dag_adj,
         'variable_order': ['A', 'B', 'C'],
         'confidence': 0.7},
        ['A', 'B', 'C'])
    assert graph.metadata.get('ra_source') == 'react_finalized'


def test_graph_update_preserves_archived_best_and_budget(monkeypatch):
    from agentcbo.optimizers.agent_cbo import AgentCBO

    names = ['A', 'B', 'Y']
    initial = CausalGraph(DAG([[], [], [0, 1]], 3), names)
    updated = CausalGraph(DAG([[], [], [1]], 3), names)
    engine = MagicMock()
    engine.name = 'test'
    engine.llm = None
    engine.discover.side_effect = [(initial, 0.9), (updated, 0.9)]
    benchmark = MagicMock()
    benchmark.target_variable = 'Y'
    benchmark.manipulative_variables = ['A', 'B']
    benchmark.domain_knowledge = None
    benchmark.sample_observational.return_value = pd.DataFrame(np.zeros((10, 3)), columns=names)
    benchmark.get_intervention_ranges.return_value = {'A': (-1, 1), 'B': (-1, 1)}
    benchmark.get_cost_functions.return_value = {}
    benchmark.get_ground_truth_dag.return_value = initial
    benchmark.evaluate_intervention.side_effect = (
        lambda intervention, seed=None: -10.0 if len(intervention) == 2 else -1.0)
    optimizer = AgentCBO(benchmark, engine=engine, num_trials=4, update_freq=2, seed=0)
    monkeypatch.setattr(optimizer, '_fit_conditional_gps', lambda graph: None)
    monkeypatch.setattr(optimizer, '_refit_conditional_gps', lambda: None)
    monkeypatch.setattr(optimizer, '_update_do_functions', lambda graph: None)
    monkeypatch.setattr(optimizer, '_update_bo_model', lambda es: None)
    monkeypatch.setattr(optimizer, '_compute_coverage', lambda: 0.0)
    monkeypatch.setattr(optimizer, '_compute_do_prior_optimum', lambda: 0.0)
    monkeypatch.setattr(optimizer, '_evaluate_acquisition',
                        lambda es, best: (float(len(es)), np.zeros((1, len(es)))))

    result = optimizer.optimize()
    assert result.best_y == -10.0
    assert result.best_es == ('A', 'B')
    assert result.best_y_trace == [0.0, -10.0, -10.0, -10.0]
    assert result.metadata['type_trial'] == ['o', 'i', 'i', 'i']
    assert result.metadata['n_interventions'] == 3
    assert ('A', 'B') in optimizer._archived_data
    assert ('A', 'B') not in optimizer.exploration_sets
    assert optimizer.gate_decisions[0]['accepted']
