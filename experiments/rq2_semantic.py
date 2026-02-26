"""
RQ2 — Semantic resilience:
Compare semantic vs raw messaging at 0%, 20%, 50% dropout.
Metrics: swarm coherence (conflicting assignments), map fidelity, survival rate.
"""

import os
import sys
import yaml
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.runner import ExperimentRunner


def compute_coherence(results: list) -> float:
    """
    Estimate swarm coherence from agent stats.
    Higher coherence = less idle time, more coordinated actions.
    """
    idle_rates = []
    for r in results:
        for aid, stats in r.get('agent_stats', {}).items():
            if aid != 'commander' and isinstance(stats, dict):
                idle_rates.append(stats.get('idle_rate', 0))
    if idle_rates:
        return 1.0 - (sum(idle_rates) / len(idle_rates))
    return 0.0


def run_rq2(results_dir: str = 'results', num_seeds: int = 5, verbose: bool = True):
    """Run RQ2 experiment."""
    print("\n" + "="*60)
    print("RQ2: Semantic Resilience — Message Mode × Dropout Rate")
    print("="*60)

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    with open(os.path.join(config_dir, 'medium_hazard.yaml')) as f:
        base_config = yaml.safe_load(f)

    dropout_rates = [0.0, 0.2, 0.5]
    message_modes = ['semantic', 'raw']

    configs = {}
    for mode in message_modes:
        for dropout in dropout_rates:
            cfg = copy.deepcopy(base_config)
            cfg['message_mode'] = mode
            cfg['dropout_rate'] = dropout
            cfg['controller_mode'] = 'hierarchical'
            cfg['commander_type'] = 'heuristic'
            name = f'{mode}_dropout{int(dropout*100)}'
            configs[name] = cfg

    runner = ExperimentRunner(results_dir=results_dir)
    all_results = runner.run_comparison(configs, num_seeds=num_seeds, verbose=verbose)

    # Save
    runner.save_results(all_results, 'rq2_results.json')

    # Summary
    if verbose:
        print("\n--- RQ2 Summary ---")
        for condition, results in all_results.items():
            agg = ExperimentRunner.aggregate_results(results)
            coherence = compute_coherence(results)
            print(f"\n{condition}:")
            print(f"  Survival rate: {agg['survival_rate']['mean']:.2%} ± {agg['survival_rate']['std']:.2%}")
            print(f"  Rescued: {agg['rescued']['mean']:.1f}")
            print(f"  Swarm coherence: {coherence:.2%}")
            if 'agent_idle_rate' in agg:
                print(f"  Agent idle rate: {agg['agent_idle_rate']['mean']:.2%}")

    return all_results


if __name__ == '__main__':
    run_rq2()
