# RQ4: Strategic Horizon Analysis — Implementation Plan

**Version:** 1.1  
**Date:** March 2025  
**Status:** Planning

---

## 1. Executive Summary

This plan describes how to implement **RQ4: Strategic Horizon Analysis** as a **fully independent** module within the Urban Seismic Resilience framework. RQ4 compares **Global vs. Local Efficiency** in trap scenarios to determine whether agents exhibit **Second-Order Thinking** (strategic: ignoring nearby small rewards for distant large rewards) vs. **Greedy Steps** (local: always moving toward nearest target).

**Key constraints:**
- RQ4 must be **independent** of RQ1–RQ3 (no shared experiment/analysis packages)
- **No modifications** to existing core: `app.py`, `controller.py`, `metrics.py`, `env/`, `agents/`, `configs/`
- **Current setup** runs via `python app.py` (Flask UI at http://127.0.0.1:5003) — RQ4 does not disturb this
- **Separate entry point**: `python run_rq4.py` (batch experiment runner, no UI)
- **Correct analysis**: Proper operationalization of Global vs Local Efficiency and Second-Order vs Greedy behavior

---

## 2. Current Setup (What Exists and How It Runs)

### 2.1 Primary Entry Point: `python app.py`

**Reality:** The project's **current operational setup** is the Flask web application, not `main.py`.

| Aspect | Detail |
|--------|--------|
| **Command** | `python app.py` |
| **URL** | http://127.0.0.1:5003 |
| **Config** | `configs/low_hazard.yaml` (loaded on `/start`) |
| **Behavior** | Step-by-step simulation: user clicks to advance one step at a time. Each step shows ground-truth grid and commander's mental map. |

**Flow:**
1. User visits `/` → `index.html` template
2. User clicks Start → POST `/start` → creates `SimulationController(config)` with `low_hazard.yaml`, calls `setup()`
3. User clicks Step → GET `/step` → `controller.run_step()` → env advances, returns `step_data` with metrics, events, base64 images
4. When `done` → `get_evaluation_metrics()` writes `results/final_metrics.json`

**Implications for RQ4:**
- `app.py` imports only `SimulationController` and `yaml`. No experiments or analysis.
- RQ4 will add `run_rq4.py` and `rq4/` — **app.py is never touched**.
- `app.py` and `run_rq4.py` are **parallel entry points**; they share the same core (controller, env, agents) but serve different purposes: interactive UI vs. batch experiment.

### 2.2 Codebase Structure

```
urban_seismic_resilience-main-final/
├── app.py                   # CURRENT SETUP: Flask UI at port 5003
├── main.py                  # Alternative entry (broken: imports experiments/, analysis/)
├── controller.py            # SimulationController — simulation loop
├── metrics.py               # compute_final_metrics — survivor rate, mental map, LLM stats
├── configs/
│   ├── low_hazard.yaml      # Used by app.py
│   ├── trap_scenario.yaml   # RQ4 scenario (collapse at step 10)
│   ├── decentralized.yaml   # Reference for decentralized config
│   └── ...
├── env/                     # Grid, environment, seismic, mental_map
├── agents/                  # Field agents, commander, messages
├── results/                 # Output directory
└── (no experiments/ or analysis/ — RQ1–RQ3 not implemented)
```

---

## 3. RQ4 Definition and Operationalization

### 3.1 Research Question (from Slide)

| Element | Definition |
|--------|------------|
| **Metric** | Global vs. Local Efficiency |
| **Expectation** | Visualize trap scenarios to see if agents exhibit **Second-Order Thinking** (ignoring nearby small reward for distant large reward) vs **Greedy Steps** |

### 3.2 Logical Chain: Why Compare Hierarchical vs Decentralized?

**Reasoning:**

1. **Second-Order Thinking** = choosing a distant high-value target over a nearby low-value target. In disaster response: "ignore the 1 victim nearby, go to the cluster of 5 victims farther away."
2. **Greedy Steps** = always moving toward the nearest target. BFS nearest-target has no notion of "value" — only distance.
3. **To observe second-order vs greedy**, we need two conditions where the *only* difference is whether agents use global coordination or local-only decisions.
4. **Hierarchical mode** = commander assigns tasks by victim count (value), plans paths on mental map, replans when blockages reported. Agents follow tasks → **global coordination**.
5. **Decentralized mode** = no commander. Agents use `_find_nearest_target()` (BFS) → **local-only, greedy**.
6. **Conclusion:** Compare hierarchical vs decentralized in the **same scenario** (trap_scenario). Survivor rate = outcome efficiency. If hierarchical > decentralized, we infer that strategic (second-order) coordination beats greedy (local) behavior.

### 3.3 Why the Trap Scenario?

**Logic:**

1. **Trap scenario** = scenario where greedy behavior is **suboptimal** by design. Config comment: "greedy approach is suboptimal; detours save more."
2. **Mechanism:** At step 10, building_id 3 collapses. This blocks the "easy" path (e.g., a corridor agents often use). Victims may be clustered beyond the blockage.
3. **Greedy agents:** Path toward nearest victim. If nearest is beyond the blockage, they path through the corridor. At step 10, corridor collapses → path blocked. Agent hits obstacle → abandons task (or in decentralized, no task — they just keep trying nearest). BFS will eventually find an alternative, but **time is wasted** and victims decay.
4. **Strategic agents:** Commander receives scout reports of blockage. Mental map updated. Commander plans **detour** around blocked corridor. Agents follow new path → reach victims faster.
5. **Hypothesis:** In trap scenario, hierarchical (strategic) should outperform decentralized (greedy) because:
   - Commander prioritizes high-victim clusters (value-based) over nearest single victim (distance-based)
   - Commander replans when blockages are reported
   - Greedy agents waste steps on blocked paths.

### 3.4 Concept → Measurement Mapping

| Concept | Operationalization | How Measured in RQ4 |
|---------|--------------------|---------------------|
| **Global Efficiency** | Survivor rate when agents are coordinated by a commander | Hierarchical condition: `controller_mode: hierarchical`, `commander_type: heuristic` |
| **Local Efficiency** | Survivor rate when agents act on nearest-target only | Decentralized condition: `controller_mode: decentralized` |
| **Second-Order Thinking** | Choosing distant high-value target over nearby low-value | Commander sorts victims by count; agents follow assigned paths; replan on blockage |
| **Greedy Steps** | Always moving toward nearest target | Decentralized agents use `_find_nearest_target()`; may waste steps on blocked path |

---

## 4. Simulation Logic (Causal Chain)

### 4.1 One Step: What Happens

**Controller.run_step()** (controller.py):

```
1. Commander phase (if hierarchical)
   - get_commander_observation() → zones, agent_positions
   - message_bus.receive_all() → reports from scouts/medics/firefighters
   - commander.decide(obs, reports, env) → task assignments
   - message_bus.send(cmd) for each task

2. Field agent phase
   For each agent:
   - observe(env) → local observation
   - receive(aid) → task assignments (if any)
   - decide(obs, messages, env) → action
     - If has task with target: _follow_task() (move along path, or rescue if adjacent to victims in danger)
     - Else: _autonomous_action() → _find_nearest_target() or rescue/extinguish if adjacent
   - Send reports (scouts always; medics/firefighters on rescue)

3. Environment step
   - env.step(actions) → execute moves, rescues, extinguish; fire spread; victim decay; black swan check
   - step_count incremented; done if max_steps or all victims accounted for

4. Log trajectory
   - Append step_data (step, actions, metrics, events) to controller.trajectory
```

**Decentralized mode:** Step 1 is skipped (`mode == 'decentralized'` → `self.commander` is None). Agents always use `_autonomous_action()`.

### 4.2 Trap Scenario: Collapse at Step 10

**Environment.step()** (env/environment.py):

- Before agent actions: `step_count` is incremented in `step()`.
- Order: (1) execute actions, (2) aftershock check, (3) black swan check, (4) fire spread, (5) victim decay.

**Black swan** (seismic.get_black_swan_events(step)): when `step == 10`, returns `{step: 10, type: 'collapse', building_id: 3}`.

**_apply_black_swan(bs):**
- Sets building 3 integrity=0, collapsed=True
- For each cell of building 3: hazard=DEBRIS, blocked=True
- `apply_spillover(b)` — blocks adjacent road cells (spillover)

**Effect:** After step 10, the corridor through/near building 3 is blocked. Agents pathfinding through that area will hit blocked cells. `_follow_task` checks `next_cell.blocked` and abandons task; `shortest_path` uses graph that excludes blocked nodes, so paths avoid them once known.

**Key insight:** Scouts report blocked roads to commander. Mental map updates. Commander's `_compute_path_on_mental_map` excludes blocked cells. So hierarchical agents get replanned paths. Decentralized agents use `env.grid.shortest_path` — grid graph is updated when cells become blocked, so they *eventually* get alternative paths. But they may have been pathing toward the corridor and waste 1+ steps discovering the block.

---

## 5. Architecture

### 5.1 Independence Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT SETUP (UNCHANGED)                                  │
│  python app.py → Flask UI at :5003 → low_hazard.yaml → step-by-step viz       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     RQ4 STANDALONE MODULE (NEW)                               │
│  python run_rq4.py → batch experiment → trap_scenario → rq4_results.json     │
│  Zero dependency on RQ1–RQ3; app.py untouched                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   run_rq4.py  ──────►  rq4/experiment.py                                      │
│       │                         │                                             │
│       │                         │  SimulationController (same as app.py)       │
│       │                         │  compute_final_metrics                       │
│       │                         │  configs/trap_scenario.yaml                  │
│       │                         ▼                                             │
│       │                  rq4_results.json                                     │
│       │                         │                                             │
│       └────────────────────────► rq4/analysis.py                              │
│                                        │                                      │
│                                        ▼                                      │
│                               Bar chart + trap frames                          │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     SHARED CORE (UNCHANGED)                                   │
│  controller.py, metrics.py, env/, agents/, configs/                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 File Structure to Create

| Path | Purpose |
|------|---------|
| `run_rq4.py` | Standalone CLI. Parses `--seeds`, `--visualize`, `--save-frames`, `--results-dir`. Calls rq4 experiment and analysis. |
| `rq4/__init__.py` | Package init |
| `rq4/experiment.py` | `run_rq4_experiment(num_seeds, results_dir, save_frames)` — runs hierarchical vs decentralized, writes `rq4_results.json` |
| `rq4/analysis.py` | `run_rq4_analysis(results_path, plots_dir)` — bar chart + optional trap frames |

---

## 6. Detailed Implementation Specification

### 6.1 Experiment Logic (`rq4/experiment.py`)

#### Conditions

| Condition | controller_mode | commander_type | Behavior |
|-----------|-----------------|----------------|-----------|
| hierarchical | hierarchical | heuristic | Commander assigns tasks; agents follow or opportunistic rescue |
| decentralized | decentralized | (ignored) | No commander; agents use nearest-target only |

**Logic:** `controller_mode` is the only behavioral switch. When `decentralized`, `_create_agents()` never creates a commander (`self.commander = None`). `run_step()` skips commander phase.

#### Config Handling

**Reasoning:** Use same trap scenario for both conditions. Only override `controller_mode` (and `commander_type` for clarity in decentralized).

```python
with open('configs/trap_scenario.yaml') as f:
    base_config = yaml.safe_load(f)

config = copy.deepcopy(base_config)
config['controller_mode'] = 'hierarchical'  # or 'decentralized'
if condition == 'decentralized':
    config['commander_type'] = 'none'
```

#### Seed Strategy

**Logic:** For fair comparison, both conditions must see the **same initial grid** (same buildings, victims, fires). The grid is generated in `env.reset()` using `config['seed']`. So we use the **same seed** for both conditions at each seed index.

- `base_seed = base_config.get('seed', 123)`
- For seed_idx in 0..num_seeds-1: `seed = base_seed + seed_idx`
- Run hierarchical with seed=123, then decentralized with seed=123 (same grid). Repeat for seed=124, 125, ...

**Important:** Grid layout, victim placement, fire placement are deterministic given seed. So we get identical layouts per seed across conditions.

#### Quiet Run

**Logic:** `controller.run_step()` prints every step. For batch runs (e.g., 5 seeds × 2 conditions × 50 steps = 500 steps), this floods the console. We should suppress stdout.

**Approach:** Use `contextlib.redirect_stdout(io.StringIO())` when calling `ctrl.run(verbose=False)`. No controller changes. `verbose=False` already suppresses the periodic summary prints in `run()`; the step-level prints come from `run_step()` which is always called. Redirecting stdout captures those.

#### Output: `results/rq4_results.json`

```json
{
  "experiment": "RQ4",
  "metric": "Global vs Local Efficiency",
  "scenario": "trap_scenario",
  "conditions": {
    "hierarchical": {
      "survivor_rate_mean": 0.xx,
      "survivor_rate_std": 0.xx,
      "rescued_mean": xx,
      "per_seed": [
        {
          "seed": 123,
          "survivor_rate": 0.xx,
          "rescued": n,
          "total_victims": m,
          "dead": k,
          "simulation_steps": s
        }
      ]
    },
    "decentralized": { ... }
  },
  "metadata": {
    "num_seeds": 5,
    "trap_collapse_step": 10,
    "config_path": "configs/trap_scenario.yaml"
  }
}
```

**Logic:** `compute_final_metrics(controller)` returns `survivor_rate`, `mental_map_fidelity`, `commander_llm_stats`. We use `survivor_rate` (rescued/total_victims) as the efficiency metric.

#### Optional: Trap Frames

**Logic:** To visualize "Second-Order vs Greedy" as in the slide, we need images at key moments. The trap scenario has a collapse at step 10. So we capture:
- Step 9 (pre-collapse)
- Step 10 (collapse just applied)
- Step 11 (post-collapse; agents may be replanning)
- Step 20 (mid-simulation)
- Final step

**Implementation:** Use `controller.run_step()` in a loop. After each step, check if step in {9,10,11,20} or done. If so, call `env.render(save_path=...)`. For hierarchical, also `env.render_mental_map(commander.mental_map, save_path=...)`. Save to `results/plots/rq4_trap_frames/{condition}_step{NN:02d}.png`.

**Note:** We must run a **single** simulation per condition (e.g., seed 0) with step-by-step control to capture frames. The batch experiment uses `ctrl.run()` which runs to completion; for frames we need a custom loop that runs step-by-step and renders at specific steps.

### 6.2 Analysis Logic (`rq4/analysis.py`)

#### Bar Chart: Global vs Local Efficiency

**Logic:** Primary RQ4 output is a bar chart comparing survivor rate across conditions.

- **Input:** `rq4_results.json`
- **Output:** `results/plots/rq4_global_vs_local_efficiency.png`
- **Chart:** Two bars: "Hierarchical (Global)" and "Decentralized (Local)". Y-axis: survivor rate (0–1). Error bars: ± std. Title: "RQ4: Strategic Horizon — Global vs Local Efficiency (Trap Scenario)"

**Interpretation:** If hierarchical bar is higher, we conclude that global (strategic) coordination outperforms local (greedy) behavior in trap scenarios.

#### Trap Scenario Frames (when save_frames=True)

- Frames saved to `results/plots/rq4_trap_frames/`
- Filenames: `{condition}_step{NN:02d}.png`, `{condition}_mental_step{NN:02d}.png` (hierarchical only for mental)

### 6.3 Entry Point (`run_rq4.py`)

**Logic:** Single entry point for RQ4. Two modes:
1. **Experiment + plot:** Run experiment, then analysis (default)
2. **Visualize only:** Skip experiment, only run analysis (requires existing `rq4_results.json`)

Flags: `--seeds`, `--visualize`, `--save-frames`, `--results-dir`.

---

## 7. Experiment Execution Flow (Step-by-Step Logic)

### 7.1 Run Order

```
For each condition in [hierarchical, decentralized]:
  For each seed_idx in 0..num_seeds-1:
    seed = base_seed + seed_idx
    config = load trap_scenario, override controller_mode, set seed
    ctrl = SimulationController(config)
    ctrl.setup()                    # env.reset(), create agents, init mental map
    with redirect_stdout(...):
      ctrl.run(verbose=False)        # run until done or max_steps
    metrics = compute_final_metrics(ctrl)
    append to per_seed: survivor_rate, rescued, total_victims, dead, steps
  aggregate: mean, std, per_seed
  store in results['conditions'][condition]

Write results to rq4_results.json
Call run_rq4_analysis(results_path, plots_dir)
```

### 7.2 Why This Order?

1. **Condition order:** Run hierarchical first, then decentralized. No functional difference; both use same seeds.
2. **Seed order:** Sequential. Ensures reproducibility (same seed → same grid).
3. **Aggregation:** Mean and std across seeds give us a point estimate and variance for each condition.

### 7.3 Trap Frames (when save_frames=True)

**Logic:** Frames require step-by-step execution. We run **one** simulation per condition (e.g., seed 0), but instead of `ctrl.run()` we use a loop:

```python
for step in range(max_steps):
  step_data = ctrl.run_step()
  if step_data['step'] in [9, 10, 11, 20] or step_data['done']:
    env.render(save_path=...)
    if commander and mental_map:
      env.render_mental_map(..., save_path=...)
  if step_data['done']:
    break
```

---

## 8. Behavioral Metrics (Optional Enhancement)

To strengthen the "Second-Order vs Greedy" narrative, extract from trajectory:

| Metric | Source | Interpretation |
|--------|--------|----------------|
| Task-following ratio | Count steps where medic/firefighter action came from task vs autonomous | Higher = more strategic coordination |
| Path abandonments | Count "Obstacle at X, abandoning task" in trajectory/logs | Strategic agents replan when blocked |
| Rescues before vs after step 10 | Sum rescues in steps 0–9 vs 10–end | Greedy may stall post-collapse |

Add to `rq4_results.json` under `conditions[*].behavioral` if implemented.

---

## 9. Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| Current setup unchanged | `python app.py` | Flask UI at :5003, low_hazard, step-by-step |
| Run experiment | `python run_rq4.py --seeds 3` | `rq4_results.json` + bar chart |
| Run with frames | `python run_rq4.py --seeds 3 --save-frames` | Same + `rq4_trap_frames/` |
| Visualize only | `python run_rq4.py --visualize` | Regenerate plots from existing JSON |
| Independence | No import of experiments.rq1/rq2/rq3 or analysis.plots | RQ4 runs without RQ1–RQ3 |
| No core changes | Diff app.py, controller.py, metrics.py, env/, agents/, configs/ | No modifications |

---

## 10. Implementation Order

1. **Create `rq4/` package**
   - `rq4/__init__.py`
   - `rq4/experiment.py` (full experiment logic)
   - `rq4/analysis.py` (bar chart + optional frames)

2. **Create `run_rq4.py`**
   - CLI parsing, directory setup, call experiment + analysis

3. **Test**
   - `python run_rq4.py --seeds 2` (quick run)
   - `python run_rq4.py --visualize`
   - `python run_rq4.py --seeds 2 --save-frames`
   - Verify `python app.py` still works

4. **Optional**
   - Behavioral metrics from trajectory
   - Summary document for trap frames

---

## 11. Files Summary

### To Create

| File | Lines (est.) | Description |
|------|--------------|-------------|
| `run_rq4.py` | ~55 | CLI entry point |
| `rq4/__init__.py` | ~5 | Package init |
| `rq4/experiment.py` | ~130 | Experiment runner |
| `rq4/analysis.py` | ~90 | Bar chart + frame handling |

### Unchanged

- `app.py` (current setup)
- `main.py`
- `controller.py`
- `metrics.py`
- `env/*`
- `agents/*`
- `configs/*`

---

## 12. References

| Resource | Path | Purpose |
|----------|------|---------|
| Current setup | `app.py` | Flask UI, low_hazard |
| Trap scenario | `configs/trap_scenario.yaml` | RQ4 scenario |
| Decentralized ref | `configs/decentralized.yaml` | controller_mode |
| Controller | `controller.py` | run_step, run, mode check |
| Metrics | `metrics.py` | compute_final_metrics |
| Field agents | `agents/field_agents.py` | _find_nearest_target, _follow_task |
| Commander | `agents/commander.py` | HeuristicCommander victim prioritization |
| Environment | `env/environment.py` | step, render, render_mental_map, _apply_black_swan |
| Seismic | `env/seismic.py` | get_black_swan_events |
