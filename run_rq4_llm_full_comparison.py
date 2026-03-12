#!/usr/bin/env python3
"""
RQ4 LLM Full Comparison: Run BOTH trap scenarios for final report.

Scenario A (Greedy-Favoured): trap_scenario_llm.yaml
  - Decentralized typically outperforms hierarchical
  - Shows Greedy Steps can dominate

Scenario B (Hierarchical-Favoured): trap_scenario_llm_hierarchical_favoured.yaml
  - Designed so hierarchical outperforms decentralized
  - Shows Second-Order Thinking when scenario favours coordination

Usage:
  python run_rq4_llm_full_comparison.py              # Run both (3 seeds each)
  python run_rq4_llm_full_comparison.py --seeds 2   # Fewer seeds (faster)
  python run_rq4_llm_full_comparison.py --scenario greedy    # Only scenario A
  python run_rq4_llm_full_comparison.py --scenario hierarchical  # Only scenario B
  python run_rq4_llm_full_comparison.py --visualize # Plots from existing results
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rq4_llm.experiment import run_rq4_llm_experiment
from rq4_llm.experiment_hierarchical_favoured import run_rq4_llm_hierarchical_favoured_experiment
from rq4_llm.analysis_full_comparison import run_full_comparison_analysis


def main():
    parser = argparse.ArgumentParser(
        description='RQ4 LLM: Run both trap scenarios for full comparison'
    )
    parser.add_argument('--seeds', type=int, default=3, help='Seeds per scenario')
    parser.add_argument('--scenario', '--s', type=str, default='both',
                        choices=['both', 'greedy', 'hierarchical'],
                        help='Which scenario(s) to run')
    parser.add_argument('--visualize', action='store_true',
                        help='Only generate plots from existing results')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Save LLM logs for first hierarchical seed (greedy scenario)')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    results_dir = args.results_dir
    plots_dir = os.path.join(results_dir, 'plots')

    if args.visualize:
        run_full_comparison_analysis(results_dir, plots_dir)
        return

    if args.scenario in ('both', 'greedy'):
        print("\n" + "=" * 60)
        print("SCENARIO A: Greedy-Favoured (trap_scenario_llm)")
        print("=" * 60)
        run_rq4_llm_experiment(
            num_seeds=args.seeds,
            results_dir=results_dir,
            save_frames=False,
            log_file=args.log_file,
        )
        try:
            from rq4_llm.generate_findings import main as gen_findings
            gen_findings()
        except Exception as e:
            print(f"Note: Could not update FINDINGS.md: {e}")

    if args.scenario in ('both', 'hierarchical'):
        print("\n" + "=" * 60)
        print("SCENARIO B: Hierarchical-Favoured (trap_scenario_llm_hierarchical_favoured)")
        print("=" * 60)
        run_rq4_llm_hierarchical_favoured_experiment(
            num_seeds=args.seeds,
            results_dir=results_dir,
            save_frames=False,
            log_file=None,
        )
        try:
            from rq4_llm.generate_findings_hierarchical_favoured import main as gen_findings
            gen_findings()
        except Exception as e:
            print(f"Note: Could not update FINDINGS_HIERARCHICAL_FAVOURED.md: {e}")

    # Generate combined analysis and report
    run_full_comparison_analysis(results_dir, plots_dir)
    print("\nFull comparison complete. See results/rq4_llm_full_comparison_report.json")
    print("Plots: results/plots/rq4_llm_full_comparison.png")


if __name__ == '__main__':
    main()
