# Urban Seismic Resilience Framework

**Team #6** · Satyam Srivastava, Jay Chaudhary, Seemandhar Jain, Adyasha Patra, Abhay Jain, Keshav Gupta

A hierarchical multi-agent coordination framework for urban disaster response after seismic events. Features a grid-based simulation environment, hierarchical agent control (Commander + Field agents), configurable communication protocols, and experiments addressing four research questions.

---

## Quick Start

### Prerequisites

```bash
python 3.10+
pip install numpy matplotlib pyyaml networkx
```

### Run a Demo

```bash
python main.py --demo
```

This runs a single simulation with the medium-hazard scenario, prints step-by-step metrics, and saves initial/final grid renders to `results/`.

### Run All Experiments (RQ1–RQ4)

```bash
python main.py
```

Or run a specific research question:

```bash
python main.py --rq 1    # Resilience under load
python main.py --rq 2    # Semantic resilience
python main.py --rq 3    # Black swan events
python main.py --rq 4    # Strategic horizon
```

### Generate Plots Only (from existing results)

```bash
python main.py --visualize
```

### Adjust Number of Seeds

```bash
python main.py --seeds 10    # 10 seeds per condition (default: 5)
```

---

## Project Structure

```
project_root/
├── main.py                 # Main entry point (run experiments, demo, plots)
├── controller.py           # Simulation controller (orchestrates env + agents)
├── plan.md                 # Technical plan document
├── README.md               # This file
├── configs/                # Scenario YAML configs
│   ├── low_hazard.yaml
│   ├── medium_hazard.yaml
│   ├── high_hazard.yaml
│   ├── black_swan.yaml
│   ├── trap_scenario.yaml
│   └── decentralized.yaml
├── env/                    # Simulation environment
│   ├── grid.py             # Cell, Building, Grid data structures + A*
│   ├── seismic.py          # Seismic model (decay, aftershocks, black swans)
│   └── environment.py      # UrbanDisasterEnv (Gymnasium-style API)
├── agents/                 # Agent implementations
│   ├── messages.py         # Message protocol + MessageBus with dropout
│   ├── field_agents.py     # Scout, Firefighter, Medic (heuristic policies)
│   └── commander.py        # HeuristicCommander + LLMCommander
├── experiments/            # RQ experiment runners
│   ├── runner.py           # ExperimentRunner harness
│   ├── rq1_resilience.py   # RQ1: Hierarchical vs decentralized at varying load
│   ├── rq2_semantic.py     # RQ2: Semantic vs raw × dropout rates
│   ├── rq3_blackswan.py    # RQ3: Black swan re-planning latency
│   └── rq4_strategic.py    # RQ4: Global vs local efficiency (trap scenarios)
├── analysis/               # Metrics and visualization
│   ├── metrics.py          # Aggregation utilities
│   └── plots.py            # Plot generators for RQ1–RQ4
└── results/                # Output directory (auto-created)
    ├── *.json              # Experiment results
    └── plots/              # Generated figures
```

---

## Environment API

The environment follows a Gymnasium-style interface:

```python
from env import UrbanDisasterEnv

env = UrbanDisasterEnv(config_path='configs/medium_hazard.yaml')
obs = env.reset()

# Place agents
env.place_agents([
    {'id': 'medic_0', 'type': 'medic', 'position': 'random'},
    {'id': 'firefighter_0', 'type': 'firefighter', 'position': 'random'},
])

# Step loop
for step in range(50):
    actions = {
        'medic_0': {'type': 'rescue'},
        'firefighter_0': {'type': 'extinguish'},
    }
    obs, rewards, done, info = env.step(actions)
    if done:
        break

print(env.get_metrics())
env.render(save_path='output.png')
```

### Actions

| Action | Description | Agents |
|--------|-------------|--------|
| `move` | Move dx,dy on road grid | All |
| `scan` | Explore 3×3 area around agent | Scout |
| `rescue` | Rescue victims in adjacent cells | Medic |
| `extinguish` | Reduce fire intensity nearby | Firefighter |
| `treat` | Stabilize victim health | Medic |
| `noop` | Do nothing | All |

---

## Research Questions

### RQ1: Resilience Under Load
How does hierarchical coordination compare to decentralized agents across low/medium/high task densities?

### RQ2: Semantic Resilience
How does natural language (semantic) vs state-code (raw) messaging perform under 0%/20%/50% message dropout?

### RQ3: Black Swan Events
How quickly do hierarchical vs decentralized systems re-plan after unexpected events (building collapses, sudden fires)?

### RQ4: Strategic Horizon
Can a commander make globally optimal decisions (sacrificing local efficiency for better overall outcomes) in trap scenarios?

---

## LLM Commander

The `LLMCommander` supports OpenAI and Anthropic APIs. Set via config:

```yaml
commander_type: llm
llm_provider: openai    # or 'anthropic'
llm_model: gpt-4o-mini
api_key: sk-...          # or set OPENAI_API_KEY env var
```

Without an API key, the LLM commander uses a simulated response that mirrors heuristic logic (for testing).

---

## Configuration

All scenarios are driven by YAML configs in `configs/`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `grid_width/height` | Grid dimensions | 50 |
| `building_density` | Fraction of area as buildings | 0.25 |
| `num_victims` | Total victims placed | 30 |
| `num_fires` | Initial fire count | 10 |
| `victim_decay_rate` | Health loss per step | 1.5 |
| `seismic.magnitude` | Earthquake magnitude | 6.5 |
| `dropout_rate` | Message drop probability | 0.0 |
| `message_mode` | `semantic` or `raw` | semantic |
| `controller_mode` | `hierarchical` or `decentralized` | hierarchical |

---

## Scope Notes

**Included:** OSM-style grid, seismic decay + aftershocks, integrity/collapse + spillover, two-level hierarchy (Commander + field), semantic/raw messaging + dropout, heuristic field agents, LLM commander, A* pathfinding, RQ1–RQ4 experiments.

**Deferred:** Zone Coordinators (Alpha/Beta/Gamma layer), LLM-driven field agents, large-scale ablation sweeps, multiple real cities via OSM API.
