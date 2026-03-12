# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent coordination framework for urban disaster response after seismic events. Python 3.10+, no build system — just `pip install -r requirements.txt` (numpy, matplotlib, pyyaml, networkx).

## Common Commands

```bash
python main.py --demo              # Single simulation with medium_hazard config
python main.py                     # Run all experiments (RQ1-RQ4)
python main.py --rq 1              # Run specific research question (1-4)
python main.py --seeds 10          # Set seeds per condition (default: 5)
python main.py --visualize         # Generate plots from existing results

python app.py                      # Flask web UI at http://127.0.0.1:5002

python -m pytest test_mental_map.py test_message_bus.py  # Run tests
```

## Architecture

**Simulation loop** (orchestrated by `SimulationController` in `controller.py`):
1. Commander observes global state + receives field reports via MessageBus
2. Commander issues task assignments back through MessageBus
3. Field agents observe local area + receive commands → produce actions
4. Actions fed to `UrbanDisasterEnv.step()` → environment advances

**Two operating modes** (`controller_mode` in YAML configs):
- `hierarchical`: Commander + field agents communicate via MessageBus
- `decentralized`: Field agents act independently (no commander)

**Key modules:**

- `env/grid.py` — Cell, Building, Victim, Grid data structures + A* pathfinding
- `env/seismic.py` — Seismic decay model, aftershocks, black swan events
- `env/environment.py` — `UrbanDisasterEnv` (Gymnasium-style: reset/step/render)
- `env/mental_map.py` — Commander's probabilistic belief state about the grid
- `agents/messages.py` — `Message` dataclass, `MessageBus` with configurable dropout, semantic vs raw modes
- `agents/field_agents.py` — `ScoutAgent`, `FirefighterAgent`, `MedicAgent` (heuristic policies)
- `agents/commander.py` — `HeuristicCommander` (rule-based) and `LLMCommander` (OpenAI/Anthropic API)
- `experiments/` — RQ1-RQ4 experiment runners + `ExperimentRunner` harness
- `analysis/` — Metrics aggregation (`metrics.py`) and plot generators (`plots.py`)

**Web UI** (`app.py`): Flask app renders ground-truth grid and commander's mental map side-by-side, stepping through simulation via `/start` and `/step` API endpoints.

## Configuration

Scenarios are YAML files in `configs/` (low/medium/high_hazard, black_swan, trap_scenario, decentralized). Key params: `grid_width/height`, `building_density`, `num_victims`, `num_fires`, `dropout_rate`, `message_mode` (semantic/raw), `controller_mode`, `commander_type` (heuristic/llm).

## Results

Experiment JSON results and plot PNGs are saved to `results/` and `results/plots/`.
