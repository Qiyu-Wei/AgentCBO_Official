"""Run AgentCBO and its matched PC-only and known-graph controls."""
from __future__ import annotations

import argparse
import fcntl
import os
import pickle
import re
import time
from typing import Dict, List, Optional

import pandas as pd

ALGORITHMS = ['AgentCBO-LP', 'AgentCBO-LP-CC', 'AgentCBO-LP-CC-RA',
              'UnknownDAG-CBO', 'OracleCBO']


def create_benchmark(name: str):
    from agentcbo.benchmarks.toy_graph import ToyGraph
    from agentcbo.benchmarks.psa_graph import PSAGraph
    from agentcbo.benchmarks.complete_graph import CompleteGraph
    from agentcbo.benchmarks.acbo_synthetic import ACBOSynthetic
    from agentcbo.benchmarks.synthetic import SyntheticBenchmark
    from agentcbo.benchmarks.er_graph import ERGraphBenchmark
    from agentcbo.benchmarks.ccbo_health import CCBOHealth
    from agentcbo.benchmarks.hiv_epidemiology import HIVEpidemiology
    from agentcbo.benchmarks.predator_prey import PredatorPrey
    from agentcbo.benchmarks.pishcat_graph import PISHCATGraph
    from agentcbo.benchmarks.alarm_graph import AlarmGraph

    registry = {
        'ToyGraph': ToyGraph,
        'PSAGraph': PSAGraph,
        'CompleteGraph': CompleteGraph,
        'CCBOHealth': CCBOHealth,
        'HIVEpidemiology': HIVEpidemiology,
        'PredatorPrey': PredatorPrey,
        'PISHCATGraph': PISHCATGraph,
        'ALARM': AlarmGraph,
        'Rosenbrock': ACBOSynthetic.rosenbrock,
        'Ackley': ACBOSynthetic.ackley,
        'Chain-5': lambda: SyntheticBenchmark.chain(5),
        'ER-10': lambda: ERGraphBenchmark(n_nodes=10, seed=0),
        'ER-20': lambda: ERGraphBenchmark(n_nodes=20, seed=0),
        'ER-50': lambda: ERGraphBenchmark(n_nodes=50, seed=0),
    }
    # Graph-instance seeds are separate from optimizer seeds.
    er_instance = re.match(r'^ER-(\d+)-g(\d+)$', name)
    if er_instance:
        return ERGraphBenchmark(n_nodes=int(er_instance.group(1)),
                                seed=int(er_instance.group(2)))
    factory = registry[name]
    return factory() if callable(factory) else factory


def create_optimizer(algo_name: str, benchmark, seed: int = None,
                     num_trials: int = 50, n_obs_init: int = 100,
                     debug: bool = False, temperature: float = 0.5,
                     update_freq: int = 5, pc_alpha: float = 0.05,
                     fidelity: Optional[Dict] = None):
    from agentcbo.optimizers.agent_cbo import AgentCBO
    from agentcbo.optimizers.oracle_cbo import OracleCBO

    common = dict(task='min', num_trials=num_trials, seed=seed,
                  n_obs_init=n_obs_init)
    if fidelity:
        common.update({k: v for k, v in fidelity.items() if v is not None})
    if algo_name == 'OracleCBO':
        return OracleCBO(benchmark, **common)
    if algo_name == 'UnknownDAG-CBO':
        from agentcbo.engines.pc_only_engine import PCOnlyEngine
        engine = PCOnlyEngine(alpha=pc_alpha)
    elif algo_name in ALGORITHMS:
        engine = _create_engine(algo_name[len('AgentCBO-'):], benchmark,
                                temperature=temperature, pc_alpha=pc_alpha)
    else:
        raise ValueError(f'Unknown algorithm: {algo_name}')
    return AgentCBO(benchmark, engine=engine, update_freq=update_freq,
                    debug=debug, **common)


def _create_engine(name: str, benchmark, temperature: float = 0.5,
                   pc_alpha: float = 0.05):
    from agentcbo.engines.cc_engine import CCEngine
    from agentcbo.engines.lp_controlled_repair_engine import LPControlledRepairEngine
    from agentcbo.engines.lp_engine import LPEngine
    from agentcbo.engines.ra_engine import RAEngine
    from agentcbo.llm.client import LLMClient

    if name not in {'LP', 'LP-CC', 'LP-CC-RA'}:
        raise ValueError(f'Unknown engine: {name}')
    client = LLMClient(temperature=temperature)
    lp = LPEngine(client, pc_alpha=pc_alpha)
    if name == 'LP':
        return lp
    return LPControlledRepairEngine(
        lp_engine=lp, cc_engine=CCEngine(client),
        ra_engine=RAEngine(client) if name == 'LP-CC-RA' else None,
        target_variable=benchmark.target_variable,
        manipulative_variables=benchmark.manipulative_variables,
        variant=name,
    )


def run_comparison(benchmark_name: str, algorithms: List[str],
                   n_trials: int = 50, n_seeds: int = 20,
                   output_dir: str = 'results', n_obs_init: int = 100,
                   debug: bool = False, temperature: float = 0.5,
                   update_freq: int = 5, pc_alpha: float = 0.05,
                   start_seed: int = 0, fidelity: Optional[Dict] = None):
    os.makedirs(output_dir, exist_ok=True)
    benchmark = create_benchmark(benchmark_name)
    results = {name: [] for name in algorithms}
    summary_rows = []
    failures = []

    for algo_name in algorithms:
        for seed in range(start_seed, start_seed + n_seeds):
            fpath = os.path.join(output_dir, f'{benchmark_name}_{algo_name}_{seed}.pkl')
            if os.path.exists(fpath):
                print(f'[{benchmark_name}] {algo_name} seed={seed}: existing result', flush=True)
                continue
            print(f'[{benchmark_name}] {algo_name} seed={seed}', flush=True)
            try:
                optimizer = create_optimizer(
                    algo_name, benchmark, seed=seed, num_trials=n_trials,
                    n_obs_init=n_obs_init, debug=debug, temperature=temperature,
                    update_freq=update_freq, pc_alpha=pc_alpha, fidelity=fidelity)
                result = optimizer.optimize()
                engine = getattr(optimizer, 'engine', None)
                client = getattr(engine, 'llm', None)
                result.metadata.update({
                    'benchmark': benchmark_name, 'algorithm': algo_name, 'seed': seed,
                    'n_trials': n_trials, 'n_obs_init': n_obs_init,
                    'update_freq': update_freq, 'pc_alpha': pc_alpha,
                    'temperature': temperature if client is not None else None,
                    'llm_model': getattr(client, 'model', None),
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                })
                results[algo_name].append(result)
                # Atomic replacement keeps readers from seeing partial results.
                tmp_out = f'{fpath}.tmp{os.getpid()}'
                with open(tmp_out, 'wb') as f:
                    pickle.dump(result, f)
                os.replace(tmp_out, fpath)
                summary_rows.append({
                    'benchmark': benchmark_name, 'algorithm': algo_name, 'seed': seed,
                    'best_y': result.best_y, 'time': result.total_time,
                    'n_interventions': result.metadata.get('n_interventions', 0),
                    'n_trials': n_trials,
                    'n_es': result.metadata.get('n_exploration_sets', 0),
                })
                print(f'  best_y={result.best_y:.6g}, time={result.total_time:.1f}s', flush=True)
            except Exception as exc:
                failures.append((algo_name, seed, str(exc)))
                print(f'  ERROR: {exc}', flush=True)

    if summary_rows:
        df_new = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, f'{benchmark_name}_summary.csv')
        # Different seeds may finish concurrently on separate workers.
        with open(csv_path + '.lock', 'w') as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                if os.path.exists(csv_path):
                    df_old = pd.read_csv(csv_path)
                    keys = ['benchmark', 'algorithm', 'seed']
                    df_old = df_old[~df_old.set_index(keys).index.isin(df_new.set_index(keys).index)]
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                df_new.to_csv(csv_path, index=False)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        print(df_new.groupby('algorithm')['best_y'].agg(['mean', 'std']).to_string())
    if failures:
        raise RuntimeError(f'{len(failures)} runs failed: {failures}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default='ToyGraph')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=ALGORITHMS[:3])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--n_obs_init', type=int, default=100)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--debug', action='store_true', help='record graph and acquisition traces')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--update_freq', type=int, default=5)
    parser.add_argument('--pc_alpha', type=float, default=0.05)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--num_anchor_points', type=int, default=None)
    parser.add_argument('--max_grid_total', type=int, default=None)
    parser.add_argument('--acqf_restarts', type=int, default=None)
    parser.add_argument('--acqf_raw_samples', type=int, default=None)
    args = parser.parse_args()
    run_comparison(
        args.benchmark, args.algorithms, args.n_trials, args.n_seeds,
        args.output_dir, args.n_obs_init, debug=args.debug,
        temperature=args.temperature, update_freq=args.update_freq,
        pc_alpha=args.pc_alpha, start_seed=args.start_seed,
        fidelity={'num_anchor_points': args.num_anchor_points,
                  'max_grid_total': args.max_grid_total,
                  'acqf_restarts': args.acqf_restarts,
                  'acqf_raw_samples': args.acqf_raw_samples})
