from __future__ import annotations

import numpy as np

from agentcbo.causal.dag import CausalGraph, DAG
from agentcbo.engines.lp_controlled_repair_engine import LPControlledRepairEngine
from agentcbo.engines.lp_engine import LPEngine
from agentcbo.engines.ra_engine import RAEngine
from experiments.run_comparison import _create_engine, create_benchmark


def _make_graph(variable_names, edges, metadata=None):
    index = {name: i for i, name in enumerate(variable_names)}
    adj = np.zeros((len(variable_names), len(variable_names)), dtype=int)
    for src, dst in edges:
        adj[index[src], index[dst]] = 1
    return CausalGraph(
        DAG.from_adjacency_matrix(adj),
        list(variable_names),
        metadata=metadata or {},
    )


def _lp_metadata(names, low_pairs):
    n = len(names)
    return {
        'lp_low_confidence_pairs': low_pairs,
        'lp_conflict_pairs': [],
        'lp_prior_matrix': np.full((n, n), 0.4).tolist(),
        'lp_pc_adj': np.zeros((n, n), dtype=int).tolist(),
    }


class _StubEngine:
    def __init__(self, name, graph=None, confidence=0.6, error=None):
        self._name = name
        self._graph = graph
        self._confidence = confidence
        self._error = error
        self.llm = None

    @property
    def name(self):
        return self._name

    def discover(self, data, variable_names, domain_knowledge=None):
        if self._error is not None:
            raise self._error
        return self._graph, self._confidence


def test_lp_cc_abstains_when_lp_has_no_target_coverage():
    names = ['X', 'Y']
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = 2.0 * x + 0.1 * rng.randn(100)
    data = np.column_stack([x, y])
    lp_graph = _make_graph(names, [], metadata=_lp_metadata(names, [('X', 'Y')]))
    cc_graph = _make_graph(names, [('X', 'Y')])

    engine = LPControlledRepairEngine(
        lp_engine=_StubEngine('LP', lp_graph),
        cc_engine=_StubEngine('CC', cc_graph),
        target_variable='Y',
        manipulative_variables=['X'],
        variant='LP-CC',
    )

    graph, conf = engine.discover(data, names, None)
    assert conf == 0.6
    assert graph.metadata['controlled_variant'] == 'LP-CC'
    assert graph.metadata['cc_applied'] is False
    assert graph.metadata['controlled_trace'][-1]['reason'] == 'lp_has_no_manipulative_ancestor_coverage'
    assert np.array_equal(graph.dag.get_adjacency_matrix(), lp_graph.dag.get_adjacency_matrix())


def test_lp_cc_coverage_guard_rejects_deleting_lp_ancestor():
    names = ['X', 'Y']
    rng = np.random.RandomState(1)
    data = rng.randn(200, 2)
    lp_graph = _make_graph(names, [('X', 'Y')], metadata=_lp_metadata(names, [('X', 'Y')]))
    cc_graph = _make_graph(names, [])

    engine = LPControlledRepairEngine(
        lp_engine=_StubEngine('LP', lp_graph),
        cc_engine=_StubEngine('CC', cc_graph),
        target_variable='Y',
        manipulative_variables=['X'],
        variant='LP-CC',
        min_gain_cc=-1e6,
    )

    graph, _ = engine.discover(data, names, None)
    assert graph.metadata['cc_applied'] is False
    assert graph.metadata['controlled_trace'][1]['reason'] == 'coverage_guard_failed'
    assert np.array_equal(graph.dag.get_adjacency_matrix(), lp_graph.dag.get_adjacency_matrix())


def test_full_caps_repair_edits():
    names = ['A', 'B', 'Y']
    rng = np.random.RandomState(2)
    a = rng.randn(200)
    b = rng.randn(200)
    y = 2.0 * a + 0.1 * rng.randn(200)
    data = np.column_stack([a, b, y])
    lp_graph = _make_graph(
        names,
        [('A', 'Y')],
        metadata=_lp_metadata(names, [('A', 'Y'), ('B', 'Y')]),
    )
    ra_graph = _make_graph(names, [('A', 'Y'), ('B', 'Y'), ('A', 'B')])

    engine = LPControlledRepairEngine(
        lp_engine=_StubEngine('LP', lp_graph),
        cc_engine=_StubEngine('CC', lp_graph),
        ra_engine=_StubEngine('RA', ra_graph),
        target_variable='Y',
        manipulative_variables=['A', 'B'],
        variant='LP-CC-RA',
        min_gain_ra=-1e6,
        max_ra_edits=1,
    )

    graph, _ = engine.discover(data, names, None)
    assert graph.metadata['controlled_variant'] == 'LP-CC-RA'
    assert graph.metadata['cc_applied'] is False
    assert graph.metadata['n_ra_kept_edits'] <= 1


def test_create_engine_registers_controlled_variants():
    benchmark = create_benchmark('ToyGraph')
    assert _create_engine('LP-CC', benchmark=benchmark, temperature=0.0).name == 'LP-CC'
    assert _create_engine('LP', benchmark=benchmark, temperature=0.0).name == 'LP'
    assert _create_engine('LP-CC-RA', benchmark=benchmark, temperature=0.0).name == 'LP-CC-RA'


def test_full_accepts_supported_calibration_and_residual_repair():
    names = ['A', 'B', 'Y']
    rng = np.random.RandomState(7)
    a = rng.randn(200)
    b = 2.0 * a + 0.3 * rng.randn(200)
    y = 2.0 * b + 0.1 * rng.randn(200)
    lp = _make_graph(names, [('A', 'Y')],
                     _lp_metadata(names, [('A', 'B'), ('A', 'Y'), ('B', 'Y')]))
    cc = _make_graph(names, [('A', 'Y'), ('A', 'B')])
    ra = _make_graph(names, [('A', 'Y'), ('A', 'B'), ('B', 'Y')])
    engine = LPControlledRepairEngine(
        lp_engine=_StubEngine('LP', lp), cc_engine=_StubEngine('CC', cc),
        ra_engine=_StubEngine('RA', ra), target_variable='Y',
        manipulative_variables=['A'], variant='LP-CC-RA')

    graph, _ = engine.discover(np.column_stack([a, b, y]), names)
    assert graph.metadata['cc_applied']
    assert graph.metadata['ra_applied']
    assert graph.metadata['n_ra_kept_edits'] == 1
    assert np.array_equal(graph.dag.get_adjacency_matrix(), ra.dag.get_adjacency_matrix())


def test_lp_supplement_records_actual_directed_edge():
    names = ['X', 'Y']
    graph = _make_graph(names, [])
    llm_prior = np.array([[0.0, 0.1], [0.9, 0.0]])

    class _FakePCGraph:
        sepset = [[None, None], [None, None]]

    engine = LPEngine(llm_client=object())
    out, llm_added = engine._supplement_llm_edges(
        graph,
        np.zeros((2, 2), dtype=int),
        llm_prior,
        _FakePCGraph(),
        np.random.RandomState(3).randn(50, 2),
        names,
    )
    adj = out.dag.get_adjacency_matrix()
    assert adj[1, 0] == 1
    assert (1, 0) in llm_added
    assert (0, 1) not in llm_added


def test_ra_parse_final_respects_variable_order():
    engine = RAEngine(llm_client=object())
    payload = {
        'adjacency_matrix': [[0, 0], [1, 0]],
        'variable_order': ['Y', 'X'],
        'confidence': 0.8,
    }
    graph, conf = engine._parse_final(payload, ['X', 'Y'])
    adj = graph.dag.get_adjacency_matrix()
    assert adj[0, 1] == 1
    assert conf == 0.8
    assert graph.metadata['ra_variable_order'] == ['Y', 'X']
