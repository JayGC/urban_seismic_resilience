"""
RQ4 — Strategic horizon:
Run trap scenarios; compare global vs local efficiency.
Track sacrificial moves (locally worse, globally better).
Compare LLM commander (simulated) vs greedy decentralized.
"""

import os
import sys
import yaml
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.runner import ExperimentRunner
from controller import SimulationController


def compute_efficiency_metrics(trajectory: list) -> dict:
    """
    Compute local vs global efficiency.
    Local efficiency: rescues per step (immediate gains)
    Global efficiency: total survival rate weighted by time-to-rescue
    """
    if not trajectory:
        return {'local_eff': 0, 'global_eff': 0, 'rescue_rate_curve': []}

    rescue_curve = []
    prev_rescued = 0

    for entry in trajectory:
        rescued = entry.get('metrics', {}).get('rescued', 0)
        rescue_curve.append(rescued - prev_rescued)
        prev_rescued = rescued

    # Local efficiency: average rescues per step
    local_eff = sum(rescue_curve) / max(1, len(rescue_curve))

    # Global efficiency: weighted by time discount (earlier rescues worth more)
    total_victims = trajectory[-1].get('metrics', {}).get('total_victims', 1)
    weighted_rescues = 0
    for i, r in enumerate(rescue_curve):
        time_weight = 1.0 - (i / len(rescue_curve)) * 0.5  # Discount later rescues
        weighted_rescues += r * time_weight

    global_eff = weighted_rescues / max(1, total_victims)

    # Detect "sacrificial" moves: steps where rescue rate dips but recovers higher
    sacrificial_steps = []
    window = 3
    for i in range(window, len(rescue_curve) - window):
        pre_avg = sum(rescue_curve[i-window:i]) / window
        post_avg = sum(rescue_curve[i+1:i+1+window]) / window
        if rescue_curve[i] < pre_avg * 0.5 and post_avg > pre_avg:
            sacrificial_steps.append(i)

    return {
        'local_efficiency': local_eff,
        'global_efficiency': global_eff,
        'rescue_rate_curve': rescue_curve,
        'sacrificial_steps': sacrificial_steps,
        'total_rescued': sum(rescue_curve),
    }


def run_rq4(results_dir: str = 'results', num_seeds: int = 5, verbose: bool = True):
    """Run RQ4 experiment."""
    print("\n" + "="*60)
    print("RQ4: Strategic Horizon — Global vs Local Efficiency")
    print("="*60)

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

    with open(os.path.join(config_dir, 'trap_scenario.yaml')) as f:
        trap_config = yaml.safe_load(f)

    # Hierarchical (LLM-simulated) commander
    hier_config = copy.deepcopy(trap_config)
    hier_config['controller_mode'] = 'hierarchical'
    hier_config['commander_type'] = 'llm'  # Uses simulated LLM

    # Decentralized (greedy)
    greedy_config = copy.deepcopy(trap_config)
    greedy_config['controller_mode'] = 'decentralized'

    seeds = list(range(42, 42 + num_seeds))
    all_results = {'hierarchical_llm': [], 'greedy_decentralized': []}

    for label, config in [('hierarchical_llm', hier_config),
                          ('greedy_decentralized', greedy_config)]:
        if verbose:
            print(f"\n--- {label.upper()} ---")

        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg['seed'] = seed

            ctrl = SimulationController(cfg)
            ctrl.setup()
            trajectory = ctrl.run(verbose=False)

            eff_metrics = compute_efficiency_metrics(trajectory)
            final_metrics = ctrl.get_final_metrics()

            all_results[label].append({
                'seed': seed,
                'final_metrics': final_metrics,
                'efficiency_metrics': {k: v for k, v in eff_metrics.items()
                                       if k != 'rescue_rate_curve'},
                'rescue_curve': eff_metrics['rescue_rate_curve'],
                'condition': label,
            })

        if verbose:
            local_effs = [r['efficiency_metrics']['local_efficiency'] for r in all_results[label]]
            global_effs = [r['efficiency_metrics']['global_efficiency'] for r in all_results[label]]
            survival = [r['final_metrics']['survival_rate'] for r in all_results[label]]
            print(f"  Mean local efficiency: {sum(local_effs)/len(local_effs):.3f}")
            print(f"  Mean global efficiency: {sum(global_effs)/len(global_effs):.3f}")
            print(f"  Mean survival rate: {sum(survival)/len(survival):.2%}")

    # Save
    runner = ExperimentRunner(results_dir=results_dir)
    runner.save_results(all_results, 'rq4_results.json')

    return all_results


if __name__ == '__main__':
    run_rq4()
