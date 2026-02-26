"""
Experiment runner: config-driven harness for running experiments across conditions and seeds.
"""

import os
import json
import yaml
import copy
import time
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import SimulationController


class ExperimentRunner:
    """
    Runs experiments across multiple configurations and seeds.
    Saves results to results/ directory.
    """

    def __init__(self, base_config_path: str = None, base_config: dict = None,
                 results_dir: str = 'results'):
        if base_config_path:
            with open(base_config_path) as f:
                self.base_config = yaml.safe_load(f)
        elif base_config:
            self.base_config = base_config
        else:
            self.base_config = {}
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run_single(self, config: dict, verbose: bool = False) -> dict:
        """Run a single simulation with given config. Returns summary."""
        ctrl = SimulationController(config)
        ctrl.setup()

        start_time = time.time()
        trajectory = ctrl.run(verbose=verbose)
        elapsed = time.time() - start_time

        final_metrics = ctrl.get_final_metrics()
        agent_stats = ctrl.get_agent_stats()
        msg_stats = ctrl.message_bus.get_stats()

        return {
            'config': {k: v for k, v in config.items() if k != 'api_key'},
            'final_metrics': final_metrics,
            'agent_stats': agent_stats,
            'message_stats': msg_stats,
            'elapsed_seconds': elapsed,
            'num_steps': len(trajectory),
            'trajectory': trajectory,
        }

    def run_sweep(self, param_name: str, param_values: list,
                  seeds: List[int] = None, num_seeds: int = 5,
                  verbose: bool = False) -> List[dict]:
        """
        Sweep over a parameter with multiple seeds.

        Args:
            param_name: Config key to vary (supports nested keys with '.')
            param_values: List of values to try
            seeds: Explicit seed list, or auto-generate
            num_seeds: Number of seeds if auto-generating

        Returns:
            List of result dicts
        """
        if seeds is None:
            seeds = list(range(42, 42 + num_seeds))

        all_results = []
        total_runs = len(param_values) * len(seeds)
        run_idx = 0

        for val in param_values:
            for seed in seeds:
                run_idx += 1
                config = copy.deepcopy(self.base_config)
                config['seed'] = seed
                self._set_nested(config, param_name, val)

                if verbose:
                    print(f"  Run {run_idx}/{total_runs}: {param_name}={val}, seed={seed}")

                result = self.run_single(config, verbose=False)
                result['param_name'] = param_name
                result['param_value'] = val
                result['seed'] = seed
                # Don't store full trajectory in sweep results to save memory
                result.pop('trajectory', None)
                all_results.append(result)

        return all_results

    def run_comparison(self, configs: Dict[str, dict],
                       seeds: List[int] = None, num_seeds: int = 5,
                       verbose: bool = True) -> Dict[str, List[dict]]:
        """
        Compare multiple named configurations.

        Args:
            configs: Dict of name -> config
            seeds: Seed list

        Returns:
            Dict of name -> list of result dicts (one per seed)
        """
        if seeds is None:
            seeds = list(range(42, 42 + num_seeds))

        all_results = {}
        for name, config in configs.items():
            if verbose:
                print(f"\n=== Running condition: {name} ===")
            results = []
            for seed in seeds:
                cfg = copy.deepcopy(config)
                cfg['seed'] = seed
                result = self.run_single(cfg, verbose=False)
                result['condition'] = name
                result['seed'] = seed
                result.pop('trajectory', None)
                results.append(result)
            all_results[name] = results

            if verbose:
                avg_survival = sum(r['final_metrics']['survival_rate'] for r in results) / len(results)
                avg_rescued = sum(r['final_metrics']['rescued'] for r in results) / len(results)
                print(f"  Avg survival rate: {avg_survival:.2%}, Avg rescued: {avg_rescued:.1f}")

        return all_results

    def save_results(self, results: Any, filename: str):
        """Save results to JSON file."""
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {path}")

    @staticmethod
    def _set_nested(d: dict, key: str, value):
        """Set a value in a nested dict using dot notation."""
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @staticmethod
    def aggregate_results(results: List[dict]) -> dict:
        """Compute mean/std of metrics across runs."""
        import numpy as np
        metrics_keys = ['survival_rate', 'rescued', 'dead', 'active_fires',
                       'blocked_roads', 'collapsed_buildings']
        agg = {}
        for key in metrics_keys:
            values = [r['final_metrics'].get(key, 0) for r in results]
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        # Agent idle rates
        idle_rates = []
        for r in results:
            for aid, stats in r.get('agent_stats', {}).items():
                if aid != 'commander' and isinstance(stats, dict):
                    idle_rates.append(stats.get('idle_rate', 0))
        if idle_rates:
            agg['agent_idle_rate'] = {
                'mean': float(np.mean(idle_rates)),
                'std': float(np.std(idle_rates)),
            }

        return agg
