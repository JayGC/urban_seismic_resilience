"""
RQ1 — Resilience under load:
Compare hierarchical (commander) vs decentralized across low/medium/high task density.
Metrics: survival rate, rescued per step, mean agent idle time.
"""

import os
import sys
import yaml
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.runner import ExperimentRunner


def run_rq1(results_dir: str = 'results', num_seeds: int = 5, verbose: bool = True):
    """Run RQ1 experiment."""
    print("\n" + "="*60)
    print("RQ1: Resilience Under Load — Hierarchical vs Decentralized")
    print("="*60)

    # Load base configs
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

    density_levels = {
        'low': os.path.join(config_dir, 'low_hazard.yaml'),
        'medium': os.path.join(config_dir, 'medium_hazard.yaml'),
        'high': os.path.join(config_dir, 'high_hazard.yaml'),
    }

    all_results = {}

    for density_name, config_path in density_levels.items():
        with open(config_path) as f:
            base_config = yaml.safe_load(f)

        # Hierarchical condition
        hier_config = copy.deepcopy(base_config)
        hier_config['controller_mode'] = 'hierarchical'
        hier_config['commander_type'] = 'heuristic'

        # Decentralized condition
        decen_config = copy.deepcopy(base_config)
        decen_config['controller_mode'] = 'decentralized'
        decen_config['commander_type'] = 'none'

        configs = {
            f'{density_name}_hierarchical': hier_config,
            f'{density_name}_decentralized': decen_config,
        }

        runner = ExperimentRunner(results_dir=results_dir)
        results = runner.run_comparison(configs, num_seeds=num_seeds, verbose=verbose)
        all_results.update(results)

    # Save
    runner = ExperimentRunner(results_dir=results_dir)
    runner.save_results(all_results, 'rq1_results.json')

    # Print summary
    if verbose:
        print("\n--- RQ1 Summary ---")
        for condition, results in all_results.items():
            agg = ExperimentRunner.aggregate_results(results)
            print(f"\n{condition}:")
            print(f"  Survival rate: {agg['survival_rate']['mean']:.2%} ± {agg['survival_rate']['std']:.2%}")
            print(f"  Rescued: {agg['rescued']['mean']:.1f} ± {agg['rescued']['std']:.1f}")
            print(f"  Dead: {agg['dead']['mean']:.1f}")
            if 'agent_idle_rate' in agg:
                print(f"  Agent idle rate: {agg['agent_idle_rate']['mean']:.2%}")

    return all_results


if __name__ == '__main__':
    run_rq1()
