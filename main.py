#!/usr/bin/env python3
"""
Urban Seismic Resilience Framework — Main Runner
Runs all experiments (RQ1-RQ4), generates plots, and saves results.

Usage:
    python main.py                    # Run all experiments
    python main.py --rq 1             # Run only RQ1
    python main.py --demo             # Quick demo run
    python main.py --visualize        # Only generate viz from existing results
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import SimulationController
from experiments.rq1_resilience import run_rq1
from experiments.rq2_semantic import run_rq2
from experiments.rq3_blackswan import run_rq3
from experiments.rq4_strategic import run_rq4
from analysis.plots import plot_rq1, plot_rq2, plot_rq3, plot_rq4


def run_demo():
    """Quick demo: run a single simulation and render the result."""
    print("\n" + "="*60)
    print("DEMO: Single Simulation Run")
    print("="*60)

    config_path = os.path.join('configs', 'medium_hazard.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ctrl = SimulationController(config)
    ctrl.setup()

    # Render initial state
    ctrl.env.render(show=False, save_path='results/demo_initial.png')

    # Run simulation
    trajectory = ctrl.run(verbose=True)

    # Render final state
    ctrl.env.render(show=False, save_path='results/demo_final.png')

    # Save trajectory
    ctrl.save_trajectory('results/demo_trajectory.json')

    # Print agent stats
    print("\n--- Agent Statistics ---")
    for aid, stats in ctrl.get_agent_stats().items():
        if isinstance(stats, dict) and 'idle_rate' in stats:
            print(f"  {aid}: idle_rate={stats['idle_rate']:.2%}, "
                  f"steps={stats['total_steps']}")

    print(f"\nDemo outputs saved to results/")


def run_all_experiments(num_seeds: int = 5):
    """Run all four research questions."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    print("\n" + "#"*60)
    print("# Urban Seismic Resilience Framework — Full Experiment Suite")
    print("#"*60)

    # RQ1
    run_rq1(results_dir=results_dir, num_seeds=num_seeds)

    # RQ2
    run_rq2(results_dir=results_dir, num_seeds=num_seeds)

    # RQ3
    run_rq3(results_dir=results_dir, num_seeds=num_seeds)

    # RQ4
    run_rq4(results_dir=results_dir, num_seeds=num_seeds)

    # Generate plots
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    generate_plots()

    print("\n" + "#"*60)
    print("# ALL EXPERIMENTS COMPLETE")
    print(f"# Results saved to {results_dir}/")
    print(f"# Plots saved to {results_dir}/plots/")
    print("#"*60)


def generate_plots():
    """Generate all plots from existing results."""
    results_dir = 'results'
    plots_dir = os.path.join(results_dir, 'plots')

    if os.path.exists(os.path.join(results_dir, 'rq1_results.json')):
        plot_rq1(os.path.join(results_dir, 'rq1_results.json'), plots_dir)

    if os.path.exists(os.path.join(results_dir, 'rq2_results.json')):
        plot_rq2(os.path.join(results_dir, 'rq2_results.json'), plots_dir)

    if os.path.exists(os.path.join(results_dir, 'rq3_results.json')):
        plot_rq3(os.path.join(results_dir, 'rq3_results.json'), plots_dir)

    if os.path.exists(os.path.join(results_dir, 'rq4_results.json')):
        plot_rq4(os.path.join(results_dir, 'rq4_results.json'), plots_dir)


def main():
    parser = argparse.ArgumentParser(description='Urban Seismic Resilience Framework')
    parser.add_argument('--rq', type=int, choices=[1, 2, 3, 4],
                       help='Run a specific research question')
    parser.add_argument('--demo', action='store_true',
                       help='Run a quick demo simulation')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate plots from existing results')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of seeds per condition (default: 5)')

    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    if args.demo:
        run_demo()
    elif args.visualize:
        generate_plots()
    elif args.rq:
        rq_funcs = {1: run_rq1, 2: run_rq2, 3: run_rq3, 4: run_rq4}
        rq_funcs[args.rq](num_seeds=args.seeds)
        generate_plots()
    else:
        run_all_experiments(num_seeds=args.seeds)


if __name__ == '__main__':
    main()
