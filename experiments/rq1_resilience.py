"""
RQ1: Resilience Under Load (Stress Test)

Compare three coordination strategies across low / medium / high task densities:
  1. Hierarchical + LLM Commander (Triton API)
  2. Hierarchical + Heuristic Commander (rule-based)
  3. Decentralized (no commander, local-only agents)

Metric: Survival Rate and Civilians Saved per Step vs Task Density.
Expectation: Hierarchical outperforms decentralized; LLM adds strategic value.
"""

import copy
import json
import os
import numpy as np
from typing import Dict, List

from experiments.runner import run_single

# ── Density presets ──────────────────────────────────────────────────────────

DENSITY_LEVELS = {
    'low': {
        'num_victims': 15,
        'num_fires': 5,
        'building_density': 0.20,
        'victim_decay_rate': 1.0,
        'fire_spread_prob': 0.02,
    },
    'medium': {
        'num_victims': 30,
        'num_fires': 10,
        'building_density': 0.25,
        'victim_decay_rate': 1.5,
        'fire_spread_prob': 0.05,
    },
    'high': {
        'num_victims': 50,
        'num_fires': 20,
        'building_density': 0.30,
        'victim_decay_rate': 2.0,
        'fire_spread_prob': 0.08,
    },
}

# Three coordination modes to compare
MODES = ['hierarchical_llm', 'hierarchical_heuristic', 'decentralized']

MODE_LABELS = {
    'hierarchical_llm': 'Hierarchical (LLM)',
    'hierarchical_heuristic': 'Hierarchical (Heuristic)',
    'decentralized': 'Decentralized',
}

# Shared base config
BASE_CONFIG = {
    'grid_width': 50,
    'grid_height': 50,
    'max_steps': 50,
    'dropout_rate': 0.0,
    'message_mode': 'semantic',
    'seismic': {
        'epicenter': [25, 25],
        'magnitude': 6.5,
        'decay_k': 0.05,
        'intensity_scale': 1.0,
        'aftershocks': [
            {'step': 8, 'magnitude': 5.0},
            {'step': 15, 'magnitude': 4.5},
        ],
        'black_swans': [],
    },
    'agents': {
        'num_scouts': 3,
        'num_firefighters': 3,
        'num_medics': 4,
    },
}


def _build_config(density: str, mode: str, seed: int) -> dict:
    """Build a full config for one experimental condition."""
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(DENSITY_LEVELS[density])
    cfg['seed'] = seed

    if mode == 'hierarchical_llm':
        cfg['controller_mode'] = 'hierarchical'
        cfg['commander_type'] = 'llm'
        cfg['llm_model'] = 'api-gpt-oss-120b'
        cfg['llm_provider'] = 'openai'
    elif mode == 'hierarchical_heuristic':
        cfg['controller_mode'] = 'hierarchical'
        cfg['commander_type'] = 'heuristic'
    else:  # decentralized
        cfg['controller_mode'] = 'decentralized'
        cfg['commander_type'] = 'none'

    return cfg


def run_rq1(results_dir: str = 'results', num_seeds: int = 5) -> dict:
    """Run the full RQ1 experiment grid and save results JSON.

    Conditions: 3 densities x 3 modes x num_seeds seeds = 45 runs (default).
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results: Dict[str, Dict[str, List[dict]]] = {}

    total_runs = len(DENSITY_LEVELS) * len(MODES) * num_seeds
    run_idx = 0

    for density in DENSITY_LEVELS:
        all_results[density] = {}
        for mode in MODES:
            all_results[density][mode] = []
            for seed_i in range(num_seeds):
                run_idx += 1
                seed = 42 + seed_i
                print(f"\n{'='*60}")
                print(f"[RQ1] Run {run_idx}/{total_runs}: "
                      f"density={density}, mode={mode}, seed={seed}")
                print(f"{'='*60}")

                cfg = _build_config(density, mode, seed)
                result = run_single(cfg, verbose=False)

                fm = result['final_metrics']
                all_results[density][mode].append({
                    'seed': seed,
                    'survival_rate': fm.get('survival_rate', 0.0),
                    'rescued': fm.get('rescued', 0),
                    'dead': fm.get('dead', 0),
                    'total_victims': fm.get('total_victims', 0),
                    'total_steps': result['total_steps'],
                    'wall_time_s': result['wall_time_s'],
                    'trajectory': result['trajectory'],
                })

    # ── Aggregate means/stds ──
    summary = {}
    for density in DENSITY_LEVELS:
        summary[density] = {}
        for mode in MODES:
            runs = all_results[density][mode]
            rates = [r['survival_rate'] for r in runs]
            rescued = [r['rescued'] for r in runs]
            dead = [r['dead'] for r in runs]
            steps = [r['total_steps'] for r in runs]
            saved_per_step = [r['rescued'] / max(1, r['total_steps']) for r in runs]

            summary[density][mode] = {
                'survival_rate_mean': float(np.mean(rates)),
                'survival_rate_std': float(np.std(rates)),
                'rescued_mean': float(np.mean(rescued)),
                'rescued_std': float(np.std(rescued)),
                'dead_mean': float(np.mean(dead)),
                'dead_std': float(np.std(dead)),
                'saved_per_step_mean': float(np.mean(saved_per_step)),
                'saved_per_step_std': float(np.std(saved_per_step)),
                'total_steps_mean': float(np.mean(steps)),
                'n_seeds': len(runs),
            }

    output = {
        'experiment': 'RQ1_resilience_under_load',
        'conditions': {
            'densities': list(DENSITY_LEVELS.keys()),
            'modes': MODES,
            'num_seeds': num_seeds,
        },
        'summary': summary,
        'raw_results': all_results,
    }

    out_path = os.path.join(results_dir, 'rq1_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[RQ1] Results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'RQ1 SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Density':<10} {'Mode':<26} {'Survival%':>10} {'Rescued':>10} "
          f"{'Dead':>8} {'Saved/Step':>11}")
    print(f"{'-'*80}")
    for density in DENSITY_LEVELS:
        for mode in MODES:
            s = summary[density][mode]
            print(f"{density:<10} {MODE_LABELS[mode]:<26} "
                  f"{s['survival_rate_mean']:>9.1%} "
                  f"{s['rescued_mean']:>9.1f} "
                  f"{s['dead_mean']:>7.1f} "
                  f"{s['saved_per_step_mean']:>10.3f}")
    print(f"{'='*80}")

    return output
