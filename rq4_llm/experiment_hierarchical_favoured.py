"""
RQ4 LLM: Hierarchical-Favoured Trap Scenario.

Same structure as experiment.py but uses trap_scenario_llm_hierarchical_favoured.yaml.
Designed so hierarchical (LLM Commander) outperforms decentralized (greedy).
"""

import copy
import io
import json
import os
import sys
from contextlib import redirect_stdout
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

from controller import SimulationController


def run_rq4_llm_hierarchical_favoured_experiment(
    num_seeds: int = 3,
    results_dir: str = 'results',
    save_frames: bool = False,
    log_file: Optional[str] = None,
) -> str:
    """
    Run RQ4 with LLM Commander in the hierarchical-favoured trap scenario.
    Returns path to rq4_llm_hierarchical_favoured_results.json.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(
        project_root, 'configs', 'trap_scenario_llm_hierarchical_favoured.yaml'
    )
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    base_config = copy.deepcopy(base_config)
    base_seed = base_config.get('seed', 42)
    max_steps = base_config.get('max_steps', 50)

    # Read trap collapse step from config
    trap_step = 10
    for bs in base_config.get('seismic', {}).get('black_swans', []):
        if bs.get('type') == 'collapse':
            trap_step = bs.get('step', 10)
            break

    results: Dict[str, Any] = {
        'experiment': 'RQ4_Strategic_Horizon_Hierarchical_Favoured',
        'metric': 'Global vs Local Efficiency',
        'commander': 'LLM',
        'scenario': 'trap_scenario_hierarchical_favoured',
        'conditions': {},
        'metadata': {
            'num_seeds': num_seeds,
            'trap_collapse_step': trap_step,
            'config_path': config_path,
            'description': (
                'Hierarchical (LLM Commander) vs Decentralized (greedy). '
                'RQ4: Second-Order Thinking vs Greedy Steps. '
                'Scenario designed to favour hierarchical coordination.'
            ),
        },
    }

    conditions = [
        ('hierarchical', 'llm'),
        ('decentralized', 'greedy'),
    ]
    frame_steps = [trap_step - 1, trap_step, trap_step + 1, 20]

    for condition_name, label in conditions:
        config = copy.deepcopy(base_config)
        config['controller_mode'] = condition_name
        if condition_name == 'decentralized':
            config['commander_type'] = 'none'

        per_seed: List[Dict[str, Any]] = []

        for seed_idx in range(num_seeds):
            seed = base_seed + seed_idx
            config['seed'] = seed

            ctrl = SimulationController(config)
            ctrl.setup()

            assert ctrl.mode == condition_name
            if condition_name == 'decentralized':
                assert ctrl.commander is None
            else:
                assert ctrl.commander is not None

            use_log = log_file and condition_name == 'hierarchical' and seed_idx == 0
            if use_log:
                f = open(log_file, 'w')
                f.write(
                    f"{'='*60}\n"
                    f"RQ4 LLM (Hierarchical-Favoured): {condition_name} (seed {seed})\n"
                    f"{'='*60}\n\n"
                )
                f.flush()
                ctx = redirect_stdout(f)
            else:
                ctx = redirect_stdout(io.StringIO())

            try:
                with ctx:
                    ctrl.run(verbose=False)
            finally:
                if use_log:
                    f.close()

            final = ctrl.final_metrics
            sr = final['survivor_rate']

            rescued_at_collapse = 0
            if len(ctrl.trajectory) >= trap_step:
                rescued_at_collapse = ctrl.trajectory[trap_step - 1]['metrics'].get('rescued', 0)
            rescued_post_collapse = sr['rescued'] - rescued_at_collapse

            per_seed.append({
                'seed': seed,
                'survivor_rate': sr['survivor_rate'],
                'rescued': sr['rescued'],
                'rescued_at_collapse': rescued_at_collapse,
                'rescued_post_collapse': rescued_post_collapse,
                'total_victims': sr['total_victims'],
                'dead': sr['dead'],
                'alive_unrescued': sr['alive_unrescued'],
                'simulation_steps': ctrl.step_count,
            })

        survivor_rates = [s['survivor_rate'] for s in per_seed]
        rescued_counts = [s['rescued'] for s in per_seed]
        post_collapse = [s['rescued_post_collapse'] for s in per_seed]

        results['conditions'][condition_name] = {
            'survivor_rate_mean': float(np.mean(survivor_rates)),
            'survivor_rate_std': float(np.std(survivor_rates)) if len(survivor_rates) > 1 else 0.0,
            'rescued_mean': float(np.mean(rescued_counts)),
            'rescued_post_collapse_mean': float(np.mean(post_collapse)),
            'rescued_post_collapse_std': float(np.std(post_collapse)) if len(post_collapse) > 1 else 0.0,
            'per_seed': per_seed,
        }

    if save_frames:
        frames_dir = os.path.join(results_dir, 'plots', 'rq4_llm_hierarchical_favoured_frames')
        os.makedirs(frames_dir, exist_ok=True)
        for condition_name, _ in conditions:
            config = copy.deepcopy(base_config)
            config['controller_mode'] = condition_name
            if condition_name == 'decentralized':
                config['commander_type'] = 'none'
            config['seed'] = base_seed

            ctrl = SimulationController(config)
            ctrl.setup()

            for _ in range(max_steps):
                with redirect_stdout(io.StringIO()):
                    step_data = ctrl.run_step()
                step_num = step_data['step']
                if step_num in frame_steps or step_data['done']:
                    save_path = os.path.join(frames_dir, f'{condition_name}_step{step_num:02d}.png')
                    ctrl.env.render(show=False, save_path=save_path)
                    if condition_name == 'hierarchical' and ctrl.commander and ctrl.commander.mental_map:
                        mental_path = os.path.join(frames_dir, f'{condition_name}_mental_step{step_num:02d}.png')
                        ctrl.env.render_mental_map(ctrl.commander.mental_map, show=False, save_path=mental_path)
                if step_data['done']:
                    break

    out_path = os.path.join(results_dir, 'rq4_llm_hierarchical_favoured_results.json')
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"RQ4 LLM (hierarchical-favoured) experiment complete. Results saved to {out_path}")
    return out_path
