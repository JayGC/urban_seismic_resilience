"""
Visualization plots for RQ1–RQ4 results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional


COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0', '#00BCD4']


def plot_rq1(results_path: str = 'results/rq1_results.json',
             save_dir: str = 'results/plots'):
    """RQ1: Survival rate and rescued counts vs task density."""
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path) as f:
        all_results = json.load(f)

    densities = ['low', 'medium', 'high']
    modes = ['hierarchical', 'decentralized']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Survival rate
    ax = axes[0]
    x = np.arange(len(densities))
    width = 0.35
    for i, mode in enumerate(modes):
        means, stds = [], []
        for d in densities:
            key = f'{d}_{mode}'
            if key in all_results:
                vals = [r['final_metrics']['survival_rate'] for r in all_results[key]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=mode.capitalize(),
               color=COLORS[i], alpha=0.8, capsize=4)
    ax.set_xlabel('Task Density')
    ax.set_ylabel('Survival Rate')
    ax.set_title('RQ1: Survival Rate vs Task Density')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([d.capitalize() for d in densities])
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 2: Rescued count
    ax = axes[1]
    for i, mode in enumerate(modes):
        means, stds = [], []
        for d in densities:
            key = f'{d}_{mode}'
            if key in all_results:
                vals = [r['final_metrics']['rescued'] for r in all_results[key]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=mode.capitalize(),
               color=COLORS[i], alpha=0.8, capsize=4)
    ax.set_xlabel('Task Density')
    ax.set_ylabel('Civilians Rescued')
    ax.set_title('RQ1: Rescued vs Task Density')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([d.capitalize() for d in densities])
    ax.legend()

    # Plot 3: Agent idle rate
    ax = axes[2]
    for i, mode in enumerate(modes):
        means = []
        for d in densities:
            key = f'{d}_{mode}'
            if key in all_results:
                idle_rates = []
                for r in all_results[key]:
                    for aid, stats in r.get('agent_stats', {}).items():
                        if aid != 'commander' and isinstance(stats, dict):
                            idle_rates.append(stats.get('idle_rate', 0))
                means.append(np.mean(idle_rates) if idle_rates else 0)
            else:
                means.append(0)
        ax.bar(x + i * width, means, width, label=mode.capitalize(),
               color=COLORS[i], alpha=0.8)
    ax.set_xlabel('Task Density')
    ax.set_ylabel('Agent Idle Rate')
    ax.set_title('RQ1: Agent Idle Rate')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([d.capitalize() for d in densities])
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'rq1_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RQ1 plot saved to {path}")


def plot_rq2(results_path: str = 'results/rq2_results.json',
             save_dir: str = 'results/plots'):
    """RQ2: Survival rate and coherence vs dropout rate for semantic/raw."""
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path) as f:
        all_results = json.load(f)

    dropouts = [0, 20, 50]
    modes = ['semantic', 'raw']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Survival rate vs dropout
    ax = axes[0]
    for i, mode in enumerate(modes):
        means, stds = [], []
        for d in dropouts:
            key = f'{mode}_dropout{d}'
            if key in all_results:
                vals = [r['final_metrics']['survival_rate'] for r in all_results[key]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
        ax.errorbar(dropouts, means, yerr=stds, marker='o', label=mode.capitalize(),
                    color=COLORS[i], linewidth=2, capsize=5)
    ax.set_xlabel('Dropout Rate (%)')
    ax.set_ylabel('Survival Rate')
    ax.set_title('RQ2: Survival Rate vs Dropout')
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 2: Coherence (1 - idle rate) vs dropout
    ax = axes[1]
    for i, mode in enumerate(modes):
        coherence_vals = []
        for d in dropouts:
            key = f'{mode}_dropout{d}'
            if key in all_results:
                idle_rates = []
                for r in all_results[key]:
                    for aid, stats in r.get('agent_stats', {}).items():
                        if aid != 'commander' and isinstance(stats, dict):
                            idle_rates.append(stats.get('idle_rate', 0))
                coherence_vals.append(1.0 - np.mean(idle_rates) if idle_rates else 0)
            else:
                coherence_vals.append(0)
        ax.plot(dropouts, coherence_vals, marker='s', label=mode.capitalize(),
                color=COLORS[i], linewidth=2)
    ax.set_xlabel('Dropout Rate (%)')
    ax.set_ylabel('Swarm Coherence')
    ax.set_title('RQ2: Coherence vs Dropout')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'rq2_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RQ2 plot saved to {path}")


def plot_rq3(results_path: str = 'results/rq3_results.json',
             save_dir: str = 'results/plots'):
    """RQ3: Re-planning latency comparison."""
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path) as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Mean replan latency
    ax = axes[0]
    conditions = ['hierarchical', 'decentralized']
    for i, cond in enumerate(conditions):
        if cond in all_results:
            latencies = [r['replan_metrics']['mean_latency'] for r in all_results[cond]]
            ax.bar(i, np.mean(latencies), yerr=np.std(latencies),
                   color=COLORS[i], alpha=0.8, capsize=5, label=cond.capitalize())
    ax.set_ylabel('Re-planning Latency (steps)')
    ax.set_title('RQ3: Re-planning Latency')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.capitalize() for c in conditions])
    ax.legend()

    # Plot 2: Survival rate under black swans
    ax = axes[1]
    for i, cond in enumerate(conditions):
        if cond in all_results:
            survival = [r['final_metrics']['survival_rate'] for r in all_results[cond]]
            ax.bar(i, np.mean(survival), yerr=np.std(survival),
                   color=COLORS[i], alpha=0.8, capsize=5, label=cond.capitalize())
    ax.set_ylabel('Survival Rate')
    ax.set_title('RQ3: Survival Under Black Swans')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.capitalize() for c in conditions])
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(save_dir, 'rq3_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RQ3 plot saved to {path}")


def plot_rq4(results_path: str = 'results/rq4_results.json',
             save_dir: str = 'results/plots'):
    """RQ4: Global vs local efficiency in trap scenarios."""
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path) as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conditions = ['hierarchical_llm', 'greedy_decentralized']
    labels = ['Hierarchical (LLM)', 'Greedy Decentralized']

    # Plot 1: Local vs Global efficiency
    ax = axes[0]
    for i, cond in enumerate(conditions):
        if cond in all_results:
            local_effs = [r['efficiency_metrics']['local_efficiency'] for r in all_results[cond]]
            global_effs = [r['efficiency_metrics']['global_efficiency'] for r in all_results[cond]]
            ax.scatter(local_effs, global_effs, color=COLORS[i], label=labels[i],
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Local Efficiency')
    ax.set_ylabel('Global Efficiency')
    ax.set_title('RQ4: Local vs Global Efficiency')
    ax.legend()

    # Plot 2: Average rescue curves
    ax = axes[1]
    for i, cond in enumerate(conditions):
        if cond in all_results:
            curves = [r.get('rescue_curve', []) for r in all_results[cond]]
            max_len = max(len(c) for c in curves) if curves else 0
            padded = [c + [0] * (max_len - len(c)) for c in curves]
            if padded:
                mean_curve = np.mean(padded, axis=0)
                cumulative = np.cumsum(mean_curve)
                ax.plot(cumulative, label=labels[i], color=COLORS[i], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Rescues')
    ax.set_title('RQ4: Rescue Progress (Trap Scenario)')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'rq4_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RQ4 plot saved to {path}")


def plot_episode_replay(trajectory: list, env, save_dir: str = 'results/plots',
                        key_frames: Optional[List[int]] = None):
    """Generate key-frame views of an episode."""
    os.makedirs(save_dir, exist_ok=True)
    if key_frames is None:
        total = len(trajectory)
        key_frames = [0, total // 4, total // 2, 3 * total // 4, total - 1]

    for frame_idx in key_frames:
        if 0 <= frame_idx < len(trajectory):
            path = os.path.join(save_dir, f'frame_{frame_idx:03d}.png')
            env.render(show=False, save_path=path)
            print(f"Saved frame {frame_idx} to {path}")
