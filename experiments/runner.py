"""
ExperimentRunner: reusable harness that runs a SimulationController for
a given config dict, collects per-step trajectory data, and returns a
standardised result dictionary.
"""

import copy
import time
from typing import Dict, List, Optional

from controller import SimulationController


def run_single(config: dict, verbose: bool = False) -> dict:
    """Run one simulation and return a result dict.

    Returns:
        {
            'config': <config used>,
            'trajectory': [per-step metrics dicts],
            'final_metrics': {rescued, dead, survival_rate, ...},
            'wall_time_s': float,
        }
    """
    cfg = copy.deepcopy(config)
    ctrl = SimulationController(cfg)
    ctrl.setup()

    t0 = time.time()
    trajectory = ctrl.run(verbose=verbose)
    wall_time = time.time() - t0

    final = ctrl.get_final_metrics()

    # Build per-step rescue time-series
    rescue_ts = []
    for step_data in trajectory:
        m = step_data.get('metrics', {})
        rescue_ts.append({
            'step': step_data['step'],
            'rescued': m.get('rescued', 0),
            'dead': m.get('dead', 0),
            'alive_unrescued': m.get('alive_unrescued', 0),
            'active_fires': m.get('active_fires', 0),
            'survival_rate': m.get('survival_rate', 0.0),
        })

    return {
        'config': cfg,
        'trajectory': rescue_ts,
        'final_metrics': final,
        'wall_time_s': round(wall_time, 2),
        'total_steps': ctrl.step_count,
    }
