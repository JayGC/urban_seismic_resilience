"""
RQ4 LLM Full Comparison: Analysis and report for both trap scenarios.

Combines:
- Scenario A (Greedy-Favoured): trap_scenario_llm
- Scenario B (Hierarchical-Favoured): trap_scenario_llm_hierarchical_favoured

Produces side-by-side bar chart and combined JSON report.
"""

import json
import os
import sys
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _load_results(results_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_full_comparison_analysis(results_dir: str = 'results',
                                 plots_dir: Optional[str] = None) -> Optional[str]:
    """
    Generate combined analysis and plot for both scenarios.
    Returns path to generated plot, or None if results missing.
    """
    if plots_dir is None:
        plots_dir = os.path.join(results_dir, 'plots')

    greedy_results = _load_results(results_dir, 'rq4_llm_results.json')
    hierarchical_results = _load_results(results_dir, 'rq4_llm_hierarchical_favoured_results.json')

    if not greedy_results:
        print("rq4_llm_results.json not found. Run full comparison first.")
    if not hierarchical_results:
        print("rq4_llm_hierarchical_favoured_results.json not found. Run full comparison first.")

    if not greedy_results and not hierarchical_results:
        return None

    # Build combined report
    report = {
        'experiment': 'RQ4_LLM_Full_Comparison',
        'scenarios': {},
        'summary': {},
    }

    if greedy_results:
        h = greedy_results['conditions'].get('hierarchical', {})
        d = greedy_results['conditions'].get('decentralized', {})
        report['scenarios']['greedy_favoured'] = {
            'name': 'Greedy-Favoured (trap_scenario_llm)',
            'hierarchical': {
                'survivor_rate_mean': h.get('survivor_rate_mean', 0),
                'survivor_rate_std': h.get('survivor_rate_std', 0),
                'rescued_mean': h.get('rescued_mean', 0),
            },
            'decentralized': {
                'survivor_rate_mean': d.get('survivor_rate_mean', 0),
                'survivor_rate_std': d.get('survivor_rate_std', 0),
                'rescued_mean': d.get('rescued_mean', 0),
            },
            'winner': 'decentralized' if d.get('survivor_rate_mean', 0) > h.get('survivor_rate_mean', 0) else 'hierarchical',
        }

    if hierarchical_results:
        h = hierarchical_results['conditions'].get('hierarchical', {})
        d = hierarchical_results['conditions'].get('decentralized', {})
        report['scenarios']['hierarchical_favoured'] = {
            'name': 'Hierarchical-Favoured (trap_scenario_llm_hierarchical_favoured)',
            'hierarchical': {
                'survivor_rate_mean': h.get('survivor_rate_mean', 0),
                'survivor_rate_std': h.get('survivor_rate_std', 0),
                'rescued_mean': h.get('rescued_mean', 0),
            },
            'decentralized': {
                'survivor_rate_mean': d.get('survivor_rate_mean', 0),
                'survivor_rate_std': d.get('survivor_rate_std', 0),
                'rescued_mean': d.get('rescued_mean', 0),
            },
            'winner': 'hierarchical' if h.get('survivor_rate_mean', 0) > d.get('survivor_rate_mean', 0) else 'decentralized',
        }

    report['summary'] = {
        'greedy_favoured_winner': report['scenarios'].get('greedy_favoured', {}).get('winner'),
        'hierarchical_favoured_winner': report['scenarios'].get('hierarchical_favoured', {}).get('winner'),
    }

    # Save report
    report_path = os.path.join(results_dir, 'rq4_llm_full_comparison_report.json')
    os.makedirs(results_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full comparison report saved to {report_path}")

    # Create combined plot
    if greedy_results or hierarchical_results:
        n_plots = 2 if (greedy_results and hierarchical_results) else 1
        fig, axes_arr = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        axes = [axes_arr] if n_plots == 1 else list(axes_arr)

        scenarios_to_plot = []
        if greedy_results:
            scenarios_to_plot.append(('greedy_favoured', 'Greedy-Favoured\nScenario', greedy_results))
        if hierarchical_results:
            scenarios_to_plot.append(('hierarchical_favoured', 'Hierarchical-Favoured\nScenario', hierarchical_results))

        for idx, (scenario_key, scenario_name, results) in enumerate(scenarios_to_plot):
            if not results:
                continue

            ax = axes[idx]
            conditions = results.get('conditions', {})
            h = conditions.get('hierarchical', {})
            d = conditions.get('decentralized', {})

            labels = ['Hierarchical', 'Decentralized']
            means = [
                h.get('survivor_rate_mean', 0),
                d.get('survivor_rate_mean', 0),
            ]
            stds = [
                h.get('survivor_rate_std', 0),
                d.get('survivor_rate_std', 0),
            ]

            x = np.arange(len(labels))
            width = 0.5
            bars = ax.bar(x, means, width, yerr=stds, capsize=6,
                          color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)

            ax.set_ylabel('Survivor Rate', fontsize=11)
            ax.set_xlabel('Condition', fontsize=11)
            ax.set_title(scenario_name, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.grid(axis='y', alpha=0.3)

            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(f'{mean:.2%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        if len(axes) == 2:
            fig.suptitle('RQ4: Full Comparison — Second-Order Thinking vs Greedy Steps',
                         fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'rq4_llm_full_comparison.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Full comparison plot saved to {out_path}")
        return out_path

    return None


if __name__ == '__main__':
    main_results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    run_full_comparison_analysis(main_results_dir)
