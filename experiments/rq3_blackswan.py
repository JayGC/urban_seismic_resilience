"""
RQ3 — Black swan events:
Run black-swan scenarios; measure re-planning latency and recovery.
Compare hierarchical vs heuristic-only response.
"""

import os
import sys
import yaml
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.runner import ExperimentRunner
from controller import SimulationController


def measure_replan_latency(trajectory: list, black_swan_steps: list) -> dict:
    """
    Measure how quickly the system recovers after a black swan event.
    Recovery = return to upward rescue trend or new rescues within N steps.
    """
    latencies = []

    for bs_step in black_swan_steps:
        # Find rescue rate before event
        pre_rescued = 0
        post_first_rescue_step = None

        for entry in trajectory:
            step = entry.get('step', 0)
            rescued = entry.get('metrics', {}).get('rescued', 0)

            if step == bs_step:
                pre_rescued = rescued

            if step > bs_step and rescued > pre_rescued:
                post_first_rescue_step = step
                break

        if post_first_rescue_step is not None:
            latency = post_first_rescue_step - bs_step
        else:
            latency = len(trajectory) - bs_step  # Never recovered

        latencies.append({
            'black_swan_step': bs_step,
            'replan_latency': latency,
            'recovered': post_first_rescue_step is not None,
        })

    return {
        'latencies': latencies,
        'mean_latency': sum(l['replan_latency'] for l in latencies) / max(1, len(latencies)),
        'recovery_rate': sum(1 for l in latencies if l['recovered']) / max(1, len(latencies)),
    }


def run_rq3(results_dir: str = 'results', num_seeds: int = 5, verbose: bool = True):
    """Run RQ3 experiment."""
    print("\n" + "="*60)
    print("RQ3: Black Swan Events — Re-planning Latency")
    print("="*60)

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

    with open(os.path.join(config_dir, 'black_swan.yaml')) as f:
        bs_config = yaml.safe_load(f)

    black_swan_steps = [bs['step'] for bs in bs_config['seismic'].get('black_swans', [])]

    # Hierarchical
    hier_config = copy.deepcopy(bs_config)
    hier_config['controller_mode'] = 'hierarchical'
    hier_config['commander_type'] = 'heuristic'

    # Decentralized
    decen_config = copy.deepcopy(bs_config)
    decen_config['controller_mode'] = 'decentralized'

    seeds = list(range(42, 42 + num_seeds))
    all_results = {'hierarchical': [], 'decentralized': []}

    for label, config in [('hierarchical', hier_config), ('decentralized', decen_config)]:
        if verbose:
            print(f"\n--- {label.upper()} ---")

        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg['seed'] = seed

            ctrl = SimulationController(cfg)
            ctrl.setup()
            trajectory = ctrl.run(verbose=False)

            replan = measure_replan_latency(trajectory, black_swan_steps)
            final_metrics = ctrl.get_final_metrics()

            all_results[label].append({
                'seed': seed,
                'final_metrics': final_metrics,
                'replan_metrics': replan,
                'condition': label,
            })

        if verbose:
            latencies = [r['replan_metrics']['mean_latency'] for r in all_results[label]]
            recovery = [r['replan_metrics']['recovery_rate'] for r in all_results[label]]
            survival = [r['final_metrics']['survival_rate'] for r in all_results[label]]
            print(f"  Mean replan latency: {sum(latencies)/len(latencies):.1f} steps")
            print(f"  Mean recovery rate: {sum(recovery)/len(recovery):.1%}")
            print(f"  Mean survival rate: {sum(survival)/len(survival):.2%}")

    # Save
    runner = ExperimentRunner(results_dir=results_dir)
    runner.save_results(all_results, 'rq3_results.json')

    return all_results


if __name__ == '__main__':
    run_rq3()
