"""
RQ4 LLM Analysis: Global vs Local Efficiency, Second-Order Thinking vs Greedy Steps.
"""

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _plot_rescue_progress_over_time(results: dict, plots_dir: str, max_steps: int = 50) -> Optional[str]:
    """
    Generate rescue progress over time plot (mean ± std across seeds).
    Requires rescued_over_time in per_seed data.
    """
    conditions = results.get('conditions', {})
    if not conditions:
        return None

    series = {}
    for cond_name, cond_data in conditions.items():
        per_seed = cond_data.get('per_seed', [])
        arrays = []
        for s in per_seed:
            rot = s.get('rescued_over_time')
            if rot is None:
                continue
            arr = list(rot)[: max_steps + 1]
            if len(arr) < max_steps + 1:
                last = arr[-1] if arr else 0
                arr.extend([last] * (max_steps + 1 - len(arr)))
            arrays.append(arr)
        if not arrays:
            continue
        series[cond_name] = np.array(arrays)

    if not series:
        print("RQ4 LLM rescue progress: No rescued_over_time data. Re-run experiment to generate.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'hierarchical': '#3498db', 'decentralized': '#e74c3c'}
    labels = {'hierarchical': 'Hierarchical (LLM)\nSecond-Order Thinking',
              'decentralized': 'Decentralized\nGreedy Steps'}

    steps = np.arange(max_steps + 1)
    for cond_name, data in series.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) if data.shape[0] > 1 else np.zeros_like(mean)
        color = colors.get(cond_name, '#333333')
        label = labels.get(cond_name, cond_name)
        ax.plot(steps, mean, color=color, linewidth=2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.3)

    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Cumulative Rescued', fontsize=12)
    ax.set_title('RQ4 LLM: Rescue Progress Over Time (Trap Scenario)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'rq4_llm_rescue_progress_over_time.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RQ4 LLM rescue progress plot saved to {out_path}")
    return out_path


def run_rq4_llm_analysis(results_path: str, plots_dir: str) -> Optional[str]:
    """
    Generate RQ4 LLM bar chart and rescue progress plot from rq4_llm_results.json.
    Returns path to bar chart, or None if results file not found.
    """
    if not os.path.exists(results_path):
        print(f"RQ4 LLM analysis: {results_path} not found. Run experiment first.")
        return None

    with open(results_path) as f:
        results = json.load(f)

    conditions = results.get('conditions', {})
    if not conditions:
        print("RQ4 LLM analysis: No conditions in results.")
        return None

    labels = []
    means = []
    stds = []

    for cond_name, cond_data in conditions.items():
        if cond_name == 'hierarchical':
            labels.append('Hierarchical (LLM)\nSecond-Order Thinking')
        elif cond_name == 'decentralized':
            labels.append('Decentralized\nGreedy Steps')
        else:
            labels.append(cond_name)

        means.append(cond_data.get('survivor_rate_mean', 0))
        stds.append(cond_data.get('survivor_rate_std', 0))

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(x, means, width, yerr=stds, capsize=6,
                  color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Survivor Rate', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('RQ4: Strategic Horizon — Second-Order Thinking (LLM) vs Greedy Steps (Trap Scenario)',
                 fontsize=13, fontweight='bold')
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
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'rq4_llm_second_order_vs_greedy.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"RQ4 LLM plot saved to {out_path}")

    # Rescue progress over time (if rescued_over_time available)
    _plot_rescue_progress_over_time(results, plots_dir)

    return out_path
