import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments import run_comparison


def test_runner_saves_successful_seeds_and_reports_failure(tmp_path, monkeypatch):
    def create_optimizer(*args, seed, **kwargs):
        if seed == 2:
            raise RuntimeError('test failure')
        result = SimpleNamespace(best_y=-1.0, total_time=0.0, metadata={})
        return SimpleNamespace(optimize=lambda: result)

    monkeypatch.setattr(run_comparison, 'create_optimizer', create_optimizer)
    with pytest.raises(RuntimeError, match='1 runs failed'):
        run_comparison.run_comparison(
            'ToyGraph', ['UnknownDAG-CBO'], n_seeds=2, start_seed=1,
            output_dir=str(tmp_path))

    paths = list(tmp_path.glob('ToyGraph_UnknownDAG-CBO_*.pkl'))
    assert len(paths) == 1
    with paths[0].open('rb') as f:
        saved = pickle.load(f)
    assert saved.metadata['seed'] == 1
    assert saved.metadata['llm_model'] is None
    assert saved.best_y == -1.0
    summary = pd.read_csv(tmp_path / 'ToyGraph_summary.csv')
    assert summary['seed'].tolist() == [1]
    assert not (tmp_path / 'ToyGraph_UnknownDAG-CBO_2.pkl').exists()
