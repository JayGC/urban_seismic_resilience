"""
RQ4 Strategic Horizon Analysis with LLM Commander.

Compares hierarchical (LLM Commander = global/strategic) vs decentralized (greedy)
in trap scenario. Answers RQ4: Do agents exhibit Second-Order Thinking
(ignoring nearby small reward for distant large reward) vs Greedy Steps?
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


def run_rq4_llm_experiment(
    num_seeds: int = 3,
    results_dir: str = 'results',
    save_frames: bool = False,
    log_file: Optional[str] = None,
    config_path: Optional[str] = None,
) -> str:
    """
    Run RQ4 with LLM Commander: hierarchical (LLM) vs decentralized (greedy).
    Returns path to rq4_llm_results.json.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is None:
        config_path = os.path.join(project_root, 'configs', 'trap_scenario_llm.yaml')
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    base_config = copy.deepcopy(base_config)
    base_seed = base_config.get('seed', 123)
    max_steps = base_config.get('max_steps', 50)

    results: Dict[str, Any] = {
        'experiment': 'RQ4_Strategic_Horizon',
        'metric': 'Global vs Local Efficiency',
        'commander': 'LLM',
        'scenario': 'trap_scenario',
        'conditions': {},
        'metadata': {
            'num_seeds': num_seeds,
            'trap_collapse_step': 10,
            'config_path': config_path,
            'description': (
                'Hierarchical (LLM Commander) vs Decentralized (greedy). '
                'RQ4: Second-Order Thinking vs Greedy Steps in trap scenario.'
            ),
        },
    }

    conditions = [
        ('hierarchical', 'llm'),   # LLM Commander = global/strategic
        ('decentralized', 'greedy'),  # No commander = local/greedy
    ]
    frame_steps = [9, 10, 11, 20]

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

            # Log first seed of hierarchical only (LLM output is verbose)
            use_log = log_file and condition_name == 'hierarchical' and seed_idx == 0
            if use_log:
                f = open(log_file, 'w')
                f.write(
                    f"{'='*60}\n"
                    f"RQ4 LLM: {condition_name} (seed {seed}) - Trap Scenario\n"
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

            trap_step = 10
            rescued_at_collapse = 0
            if len(ctrl.trajectory) >= trap_step:
                rescued_at_collapse = ctrl.trajectory[trap_step - 1]['metrics'].get('rescued', 0)
            rescued_post_collapse = sr['rescued'] - rescued_at_collapse

            # Build rescued_over_time: cumulative rescued at each step 0..max_steps
            # Index 0 = before any action (always 0)
            # Index s = cumulative rescued after step s
            rescued_over_time = [0] * (max_steps + 1)
            for t in ctrl.trajectory:
                s = t['step']
                if s <= max_steps:
                    rescued_over_time[s] = t['metrics'].get('rescued', 0)
            # Fill remaining steps after early termination with final count
            final_rescued = sr['rescued']
            last_step = len(ctrl.trajectory)
            for i in range(last_step + 1, max_steps + 1):
                rescued_over_time[i] = final_rescued

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
                'rescued_over_time': rescued_over_time,
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
        frames_dir = os.path.join(results_dir, 'plots', 'rq4_llm_trap_frames')
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

    out_path = os.path.join(results_dir, 'rq4_llm_results.json')
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"RQ4 LLM experiment complete. Results saved to {out_path}")
    return out_path
