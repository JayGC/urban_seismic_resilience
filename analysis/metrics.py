"""
Metrics aggregation for experiment results.
"""

import json
import numpy as np
from typing import Dict, List, Any


def load_results(path: str) -> dict:
    """Load results JSON file."""
    with open(path) as f:
        return json.load(f)


def aggregate_experiment_results(results: Dict[str, List[dict]]) -> Dict[str, dict]:
    """
    Aggregate results across seeds for each condition.
    Returns: {condition: {metric: {mean, std, min, max}}}
    """
    aggregated = {}

    for condition, runs in results.items():
        metrics_keys = ['survival_rate', 'rescued', 'dead', 'active_fires',
                       'blocked_roads', 'collapsed_buildings']
        agg = {}
        for key in metrics_keys:
            values = [r.get('final_metrics', {}).get(key, 0) for r in runs]
            if values:
                agg[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values,
                }

        # Idle rates
        idle_rates = []
        for r in runs:
            for aid, stats in r.get('agent_stats', {}).items():
                if aid != 'commander' and isinstance(stats, dict):
                    idle_rates.append(stats.get('idle_rate', 0))
        if idle_rates:
            agg['agent_idle_rate'] = {
                'mean': float(np.mean(idle_rates)),
                'std': float(np.std(idle_rates)),
            }

        aggregated[condition] = agg

    return aggregated


def compute_map_fidelity(commander_mental_map: dict, ground_truth: dict) -> float:
    """
    Compare commander's mental map with ground truth.
    Returns fidelity score 0-1.
    """
    if not commander_mental_map or not ground_truth:
        return 0.0

    matches = 0
    total = 0

    for agent_id, info in ground_truth.items():
        total += 1
        if agent_id in commander_mental_map:
            cmd_info = commander_mental_map[agent_id]
            # Check if reported position matches
            if cmd_info.get('metadata', {}).get('position') == info.get('position'):
                matches += 1

    return matches / max(1, total)
