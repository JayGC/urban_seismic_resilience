#!/usr/bin/env python3
"""
RQ4 Strategic Horizon Analysis with LLM Commander.

Compares hierarchical (LLM Commander = Second-Order Thinking) vs decentralized (Greedy Steps)
in trap scenario. Answers: Do agents exhibit Second-Order Thinking vs Greedy Steps?

Usage:
  python run_rq4_llm.py              # Run experiment (3 seeds)
  python run_rq4_llm.py --seeds 5     # Run with 5 seeds
  python run_rq4_llm.py --log-file out_rq4_llm.log  # Save LLM logs
  python run_rq4_llm.py --visualize   # Generate plots from existing results
  python run_rq4_llm.py --save-frames # Save trap scenario frames
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rq4_llm.experiment import run_rq4_llm_experiment
from rq4_llm.analysis import run_rq4_llm_analysis


def main():
    parser = argparse.ArgumentParser(
        description='RQ4 Strategic Horizon: LLM Commander (Second-Order Thinking) vs Greedy Steps'
    )
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds (default: 3)')
    parser.add_argument('--visualize', action='store_true', help='Generate plots from existing results')
    parser.add_argument('--save-frames', action='store_true', help='Save trap scenario frames')
    parser.add_argument('--log-file', type=str, default=None, metavar='PATH',
                        help='Save LLM step-by-step logs for first hierarchical seed')
    parser.add_argument('--config', type=str, default=None, metavar='PATH',
                        help='Path to scenario YAML (default: configs/trap_scenario_llm.yaml)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory for results (default: results)')
    args = parser.parse_args()

    results_path = os.path.join(args.results_dir, 'rq4_llm_results.json')
    plots_dir = os.path.join(args.results_dir, 'plots')

    if args.visualize:
        run_rq4_llm_analysis(results_path, plots_dir)
    else:
        run_rq4_llm_experiment(
            num_seeds=args.seeds,
            results_dir=args.results_dir,
            save_frames=args.save_frames,
            log_file=args.log_file,
            config_path=args.config,
        )
        run_rq4_llm_analysis(results_path, plots_dir)
        # Update FINDINGS.md with results
        try:
            from rq4_llm.generate_findings import main as gen_findings
            gen_findings()
        except Exception as e:
            print(f"Note: Could not update FINDINGS.md: {e}")


if __name__ == '__main__':
    main()
