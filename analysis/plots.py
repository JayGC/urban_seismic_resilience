"""
Plot generators for RQ1–RQ4.
Each function loads a results JSON and produces publication-quality figures.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.metrics import build_timeseries_envelope


# ── Shared style ─────────────────────────────────────────────────────────────

MODE_COLORS = {
    'hierarchical_llm': '#7c3aed',       # purple
    'hierarchical_heuristic': '#2563eb',  # blue
    'hierarchical': '#2563eb',            # blue (legacy)
    'decentralized': '#dc2626',           # red
}
MODE_LABELS = {
    'hierarchical_llm': 'Hierarchical (LLM)',
    'hierarchical_heuristic': 'Hierarchical (Heuristic)',
    'hierarchical': 'Hierarchical (Commander)',
    'decentralized': 'Decentralized (No Commander)',
}
DENSITY_ORDER = ['low', 'medium', 'high']


def _style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })


def _get_modes(summary: dict) -> list:
    """Detect which modes are present in the results (handles 2 or 3 modes)."""
    sample_density = next(iter(summary))
    return list(summary[sample_density].keys())


# ── RQ1: Resilience Under Load ──────────────────────────────────────────────

def plot_rq1(results_path: str, plots_dir: str = 'results/plots'):
    """Generate RQ1 figures from rq1_results.json.

    Dynamically handles 2 modes (old) or 3 modes (new: LLM + heuristic + decentr.)
    """
    _style()
    os.makedirs(plots_dir, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    summary = data['summary']
    raw = data['raw_results']
    modes = _get_modes(summary)
    n_modes = len(modes)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 1: Survival Rate bar chart
    # ────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(DENSITY_ORDER))
    width = 0.75 / n_modes

    for i, mode in enumerate(modes):
        color = MODE_COLORS.get(mode, '#888888')
        label = MODE_LABELS.get(mode, mode)
        means = [summary[d][mode]['survival_rate_mean'] * 100 for d in DENSITY_ORDER]
        stds = [summary[d][mode]['survival_rate_std'] * 100 for d in DENSITY_ORDER]
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=label, color=color,
                      capsize=4, edgecolor='black', linewidth=0.5)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Task Density')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('RQ1: Survival Rate vs Task Density')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DENSITY_ORDER])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'rq1_survival_rate.png'))
    plt.close(fig)
    print(f"  [RQ1] Saved rq1_survival_rate.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 2: Civilians Saved per Step
    # ────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, mode in enumerate(modes):
        color = MODE_COLORS.get(mode, '#888888')
        label = MODE_LABELS.get(mode, mode)
        means = [summary[d][mode]['saved_per_step_mean'] for d in DENSITY_ORDER]
        stds = [summary[d][mode]['saved_per_step_std'] for d in DENSITY_ORDER]
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=label, color=color,
                      capsize=4, edgecolor='black', linewidth=0.5)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Task Density')
    ax.set_ylabel('Civilians Saved / Step')
    ax.set_title('RQ1: Rescue Efficiency vs Task Density')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DENSITY_ORDER])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'rq1_saved_per_step.png'))
    plt.close(fig)
    print(f"  [RQ1] Saved rq1_saved_per_step.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 3: Rescue time-series (one subplot per density)
    # ────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for idx, density in enumerate(DENSITY_ORDER):
        ax = axes[idx]
        for mode in modes:
            color = MODE_COLORS.get(mode, '#888888')
            label = MODE_LABELS.get(mode, mode)
            runs = raw[density][mode]
            max_steps = max(r.get('total_steps', 50) for r in runs)
            ts = build_timeseries_envelope(runs, 'rescued', max_steps=max_steps)
            steps = ts['steps']
            mean = np.array(ts['mean'])
            std = np.array(ts['std'])

            ax.plot(steps, mean, color=color, label=label, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.12)

        ax.set_title(f'{density.capitalize()} Density')
        ax.set_xlabel('Simulation Step')
        if idx == 0:
            ax.set_ylabel('Cumulative Rescued')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle('RQ1: Rescue Progress Over Time', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'rq1_rescue_timeseries.png'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  [RQ1] Saved rq1_rescue_timeseries.png")


# ── Stubs for RQ2–RQ4 ───────────────────────────────────────────────────────

def plot_rq2(results_path: str, plots_dir: str = 'results/plots'):
    print("  [RQ2] Plot not yet implemented.")


def plot_rq3(results_path: str, plots_dir: str = 'results/plots'):
    print("  [RQ3] Plot not yet implemented.")


def plot_rq4(results_path: str, plots_dir: str = 'results/plots'):
    print("  [RQ4] Plot not yet implemented.")
