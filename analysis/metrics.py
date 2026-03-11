"""Aggregation utilities for experiment results."""

import numpy as np
from typing import Dict, List


def aggregate_seeds(runs: List[dict], key: str) -> dict:
    """Compute mean and std of a metric across seed runs."""
    values = [r[key] for r in runs if key in r]
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'n': 0}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'n': len(values),
    }


def build_timeseries_envelope(runs: List[dict], metric: str = 'rescued',
                               max_steps: int = 50) -> dict:
    """Build mean/std time-series across seeds for a given metric.

    Each run must have a 'trajectory' list of dicts with 'step' and `metric` keys.

    Returns:
        {'steps': [0..max_steps], 'mean': [...], 'std': [...]}
    """
    # Pad shorter trajectories with their last value
    matrix = []
    for run in runs:
        traj = run.get('trajectory', [])
        values = [t.get(metric, 0) for t in traj]
        if len(values) < max_steps:
            last = values[-1] if values else 0
            values.extend([last] * (max_steps - len(values)))
        matrix.append(values[:max_steps])

    arr = np.array(matrix, dtype=float)
    return {
        'steps': list(range(1, max_steps + 1)),
        'mean': arr.mean(axis=0).tolist(),
        'std': arr.std(axis=0).tolist(),
    }
