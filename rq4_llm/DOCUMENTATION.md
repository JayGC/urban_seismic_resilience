# RQ4 LLM: Complete Logic and Code Documentation

This document provides an exhaustive explanation of the RQ4 Strategic Horizon Analysis experiment using the LLM Commander. It covers the research question, experiment design, configuration, simulation flow, agent logic, commander behaviour, analysis pipeline, and results interpretation. All explanations are written in detailed paragraph form.

---

## 1. Research Question and Motivation

The RQ4 LLM experiment addresses a fundamental question in multi-agent coordination for urban disaster response: **Do agents exhibit Second-Order Thinking (the ability to ignore a nearby small reward in favour of a distant larger reward) or do they default to Greedy Steps (always choosing the nearest target at each moment)?** This distinction matters because in complex disaster scenarios, the locally optimal choice may lead agents into traps—for example, a path that appears short but becomes blocked by a collapse, whereas a slightly longer route would have succeeded. The metric used to evaluate this is **Global vs. Local Efficiency**: we compare how well a globally coordinated system (with an LLM Commander making strategic assignments) performs against a purely local system (agents acting autonomously and greedily) in a deliberately designed trap scenario.

The experiment is part of a broader urban seismic resilience simulation framework. The framework models an earthquake-stricken city with buildings, roads, fires, collapsed structures, and civilian victims. Field agents—scouts, firefighters, and medics—must explore, extinguish fires, and rescue victims. The key variable is whether these agents receive strategic task assignments from a central LLM Commander (hierarchical mode) or act entirely on their own using a nearest-target heuristic (decentralized mode). By comparing survivor rates and rescue counts between these two conditions in a scenario that includes a predictable building collapse (the "trap"), we can infer whether Second-Order Thinking emerges under hierarchical coordination or whether Greedy Steps dominate.

---

## 2. Experiment Design and Conditions

The experiment compares exactly two conditions, each run for a configurable number of random seeds (default three, often reduced to two for cost or time reasons). The two conditions are **hierarchical** and **decentralized**. In the hierarchical condition, the simulation uses `controller_mode: hierarchical` and `commander_type: llm`. A central LLM Commander observes the global state (zone summaries, agent reports, and a mental map of the grid), issues task assignments to free agents, and those agents follow the assigned targets. In the decentralized condition, the simulation uses `controller_mode: decentralized` and `commander_type: none`. No commander exists; the controller skips the commander phase entirely. Field agents never receive task assignments and therefore always fall back to their autonomous behaviour, which is implemented as a greedy nearest-target policy.

For each condition and each seed, the experiment runs the full simulation until either the maximum number of steps (50 in the trap scenario) is reached or the environment signals termination. At the end of each run, the experiment extracts metrics: survivor rate (rescued / total victims), total rescued count, rescued count at the moment of the trap collapse (step 10), and rescued count after the collapse. These metrics are aggregated across seeds (mean and standard deviation) and stored in a JSON results file. The trap collapse at step 10 is the critical event: it blocks a path that agents might have been using. Agents following a commander-assigned path planned on a mental map that accounts for blockages could route around the collapse; agents using local greedy pathfinding may waste steps discovering the block or get stuck.

---

## 3. Trap Scenario Configuration

The trap scenario is defined in `configs/trap_scenario_llm.yaml`. The grid is 50×50 cells with a building density of 0.35. There are 40 victims and 15 fires. Victim health decays at 2.5 per step if not rescued; fire spread probability is 0.06. The seismic model includes an initial earthquake with epicenter at (10, 10) and magnitude 7.0, plus two aftershocks at steps 6 and 18 with different epicenters. The defining feature is a **black swan event** at step 10: a building with ID 3 collapses. This collapse blocks roads adjacent to that building. Agents pathfinding through that area will encounter blocked cells; the grid graph is updated to exclude blocked nodes, so pathfinding eventually finds alternative routes, but agents may have already committed to a path that becomes invalid.

The agent composition is 3 scouts, 3 firefighters, and 4 medics. Scouts have an observation radius of 2; firefighters and medics have radius 1. Message dropout is set to 0 (all messages are delivered), and message mode is semantic (structured reports with metadata). The configuration is loaded as a base and then modified per condition: for hierarchical, `controller_mode` and `commander_type` remain as specified; for decentralized, `controller_mode` is set to `decentralized` and `commander_type` to `none`. The seed is incremented per run (base_seed + seed_idx) so that each condition-seed combination uses a different random initialization while remaining comparable across conditions for the same seed index.

---

## 4. Simulation Controller and Step Loop

The simulation is orchestrated by `SimulationController` in `controller.py`. The controller holds the environment (`UrbanDisasterEnv`), the message bus (`MessageBus`), the field agents dictionary, and optionally a commander. The mode (`hierarchical` or `decentralized`) is read from the config and determines whether a commander is created. In hierarchical mode, an `LLMCommander` is instantiated with the list of agent IDs, API key, model name, and provider; the commander is also given a direct reference to the field agents dictionary so it can inspect which agents are busy. In decentralized mode, `self.commander` remains `None`.

Each simulation step (`run_step`) proceeds in four phases. **Phase 1 (Commander phase):** Only if the mode is hierarchical and a commander exists, the controller fetches the commander observation from the environment (`get_commander_observation`), receives all pending messages from the bus, calls `commander.decide()`, and sends any resulting task-assignment messages through the bus. **Phase 2 (Field agent phase):** For each field agent, the controller obtains the agent's local observation, retrieves messages addressed to that agent from the bus, and calls `agent.decide(obs, incoming, env)`. The agent returns an action and optional outgoing messages (reports, rescue confirmations). All outgoing messages are sent through the bus. **Phase 3 (Environment step):** The controller passes the collected actions to `env.step(actions)`, which updates the grid (victim decay, fire spread, collapse checks), applies the actions, and returns the new observation, rewards, and done flag. **Phase 4 (Trajectory logging):** The controller appends a step record to the trajectory, including step number, actions, agent positions, metrics, rewards, events, and message statistics.

The `run` method iterates `run_step` up to `max_steps` times or until the environment signals done. At the end, it computes final metrics via `compute_final_metrics` and stores them in `self.final_metrics` for the experiment to read.

---

## 5. Hierarchical Mode: LLM Commander Logic

In hierarchical mode, the LLM Commander is the strategic decision-maker. It is implemented as `LLMCommander` in `agents/commander.py`, extending the base `CommanderAgent`. The commander maintains a mental map—a probabilistic or belief-based representation of the grid that is initialized from the actual grid's static structure (buildings, roads) but does not initially know about hazards, victims, or blockages. The mental map is updated from field agent reports: scouts send reports with exact coordinates of fires, victims, blocked roads, and collapsed buildings; medics and firefighters send reports when they rescue victims. The commander uses this mental map to refine targets and compute paths that avoid known blockages.

Each step, the commander's `decide` method is invoked with the commander observation, the list of reports from the bus, and the environment. The method first updates the mental map by processing all reports (extracting findings and updating cells). It then builds a prompt for the LLM. The prompt includes: the current step number; zone summaries (noisy estimates of victims, fires, blocked roads, and collapsed buildings per zone); known victims in burning buildings and collapsed buildings from the mental map; known fire locations; the list of free agents with their positions and types; and recent agent reports. The prompt instructs the LLM to respond with a JSON array of task assignments, each specifying `agent_id`, `task_type` (search_zone, rescue_victims, extinguish_fire, move_to), `target_x`, `target_y`, and `reason`. It also lists priorities: assign firefighters to burning buildings with victims first, then medics to collapsed buildings, then remaining firefighters to fires, then remaining medics, then scouts to unexplored zones. The LLM is told to consider blocked roads and to assign only free agents.

The commander calls the LLM API (Triton API at UCSD, or another OpenAI-compatible endpoint) with this prompt. The raw response is parsed to extract the JSON array. Each assignment is converted into a `Message` of type `TASK_ASSIGNMENT` with metadata including `target_pos`, `task_type`, and for search_zone tasks, `zone_bounds`. Assignments for agents that are not in the free-agents list are discarded. The commander then refines each command: for rescue_victims tasks, it snaps the target to the nearest known victim in a collapsed (or burning) building from the mental map; for extinguish_fire tasks, it snaps to the nearest fire-with-victims or fire-only location. This refinement ensures that zone-level targets from the LLM are mapped to exact grid coordinates. Finally, the commander attaches paths to each command by calling `_compute_path_on_mental_map`, which runs A* on a weighted graph derived from the mental map. Explored cells have cost 1; unexplored cells have cost 2 (exploration penalty); known blocked cells are excluded. The resulting path is stored in the command metadata so that agents can follow it step-by-step without recomputing.

---

## 6. Decentralized Mode: No Commander, Greedy Agents

In decentralized mode, the controller never creates a commander. The commander phase is skipped: `commander_commands` remains empty, and no task-assignment messages are ever sent. When each field agent calls `decide`, the `incoming` messages list is empty (or contains only reports from previous steps that are not task assignments). The agent's loop over messages finds no `TASK_ASSIGNMENT` messages, so `current_task` is never set. The agent then checks: if it has a task with a target, it would follow the task; but since it does not, it always executes `_autonomous_action(obs, env)`.

The autonomous behaviour is what defines "greedy" in this experiment. For medics and firefighters, `_autonomous_action` first checks whether the agent is adjacent to victims in danger (in the local observation grid). If so, it immediately returns a rescue action. Otherwise, it calls `_find_nearest_target(env, target_type)` to locate the nearest fire or victim. For firefighters, the priority is: adjacent victims in danger → adjacent fire → nearest fire → nearest victim. For medics: adjacent victims in danger → nearest victim. The `_find_nearest_target` method uses breadth-first search (BFS) on the grid graph starting from the agent's position. BFS visits cells in order of graph distance, so the first cell that has an adjacent fire or victim (in danger) is the nearest. The method returns the road cell adjacent to that target. The agent then calls `_move_toward(target, env)`, which uses A* pathfinding on the ground-truth grid to compute a shortest path and returns a move action for the first step along that path. If no target is found, the agent performs a random move. Scouts, when autonomous, use `_find_nearest_unexplored` (BFS for the nearest unexplored road cell) and move toward it; if none exists, they scan or random-walk.

Thus, decentralized agents always choose the nearest reachable target at each step. They do not plan around future blockages; they discover blockages only when they attempt to move into a blocked cell. The grid graph is updated when cells become blocked (e.g., at the trap collapse), so pathfinding will eventually route around them, but the agent may have been heading toward a corridor that is now blocked, losing time. This is the essence of Greedy Steps: myopic, local optimisation at every decision point.

---

## 7. Field Agent Logic in Detail

### 7.1 Base FieldAgent and the Decide Flow

Every field agent extends `FieldAgent` in `agents/field_agents.py`. The base class defines `decide(obs, messages, env)`, which is the main entry point each step. The agent first processes incoming messages. For each message of type `TASK_ASSIGNMENT`, if the agent is not already busy (`current_task` is None), it accepts the task by calling `_accept_task`. This sets `current_task` (with `type` and `target_pos`) and `path`. If the message includes a precomputed path in metadata, that path is used; otherwise, the agent computes a path locally via `env.grid.shortest_path` or, if no path exists (e.g., target unreachable), `_path_to_nearest_reachable` to get as close as possible.

The agent then chooses its action. If it has a task with a target and a non-empty path, it calls `_follow_task`. Otherwise, it calls `_autonomous_action`. After choosing the action, scouts always generate a report (with findings) and append it to outgoing messages. Medics and firefighters, when performing a rescue, append a rescue report with `rescued_at` metadata so the commander can update the mental map. The action is returned along with the outgoing messages.

### 7.2 Following a Task (_follow_task)

When following a task, the agent first checks if the path is exhausted (length ≤ 1). If so, it has arrived; it performs the task action (rescue, extinguish, or scan depending on task type), clears the task and path, and returns. Before moving, it checks for opportunistic rescue: if the agent is a medic or firefighter and is adjacent to a cell with victims in danger, it stops and rescues immediately, ignoring the path for that step. This allows agents to rescue victims they encounter en route. Next, it checks whether the next cell on the path is blocked. If the next cell is a blocked road, the agent abandons the task (clears `current_task` and `path`) and returns noop. This allows the commander to reassign with updated information. Otherwise, the agent moves one step along the path.

### 7.3 Scout Autonomous Action

Scouts override `_autonomous_action`. They call `_find_nearest_unexplored(env)`, which performs BFS from the agent's position to find the nearest road cell that has not been explored. If found, they move toward it via `_move_toward`. If the current cell is unexplored, they scan. Otherwise, they random-walk. Scouts also override `_follow_task` for `search_zone` tasks: when assigned a zone, they systematically explore all reachable road cells within the zone bounds, replanning when they hit obstacles, until the zone is fully explored.

### 7.4 Firefighter Autonomous Action

Firefighters override `_autonomous_action` with a strict priority order. First, they check the local observation for victims in danger (adjacent). If present, they rescue. Second, they check for adjacent fire; if present, they extinguish. Third, they call `_find_nearest_target(env, 'fire')` and move toward it. Fourth, they call `_find_nearest_target(env, 'victim')` and move toward it. Fifth, if no target is found, they random-walk.

### 7.5 Medic Autonomous Action

Medics override `_autonomous_action` with a simpler priority. First, they check for adjacent victims in danger and rescue if present. Second, they call `_find_nearest_target(env, 'victim')` and move toward it. Third, they random-walk. For victims, `_find_nearest_target` only considers cells that are "in danger" (burning or collapsed building), so medics target actual victims needing rescue, not safe victims.

### 7.6 _find_nearest_target and _move_toward

The `_find_nearest_target(env, target_type)` method implements the greedy nearest-target logic. It uses BFS on `env.grid.graph` (the traversable graph of road cells, excluding blocked ones). The queue starts with the agent's position. For each dequeued cell, it checks all cells within a 3×3 neighbourhood. For `target_type == 'victim'`, it looks for a cell with unrescued victims (health > 0) in a cell that is in danger (`env.grid.is_cell_in_danger`). For `target_type == 'fire'`, it looks for a cell with fire hazard. The first such cell found corresponds to the nearest target (BFS order). The method returns the road cell `current` (the cell from which the target is adjacent), not the target cell itself, because agents need to path to a road cell adjacent to a building or hazard to perform rescue or extinguish.

The `_move_toward(target, env)` method uses `env.grid.shortest_path(self.position, target)` (A* on the grid graph) to get a path. If the path has at least two steps, it returns a move action for the second step (the first step toward the target). Otherwise, it returns noop.

---

## 8. Mental Map and Path Computation

The mental map (`MentalMap` in `env/mental_map.py`) is the commander's belief state. It is initialized with the same width and height as the grid and is populated from the grid's static structure (buildings, cell types). Dynamic state (hazards, victims, blockages) is unknown until agents report. When a scout or other agent sends a report with findings, the commander's `update_mental_map` processes it: for each fire, victim, blocked road, or collapsed building at exact coordinates, the mental map updates the corresponding cell. The mental map also tracks which cells have been explored.

The commander's `_compute_path_on_mental_map(start, goal)` builds a weighted graph from the mental map. For each road cell, it adds a node. Known blocked cells are excluded. Edges connect adjacent cells; the edge weight is 1 for explored cells and 2 for unexplored cells (to favour known paths). A* is run on this graph to find a path from start to goal. If the mental map has learned about the trap collapse (from scout reports), the path will avoid blocked cells. The resulting path is attached to each task-assignment message so that agents can follow it without needing global knowledge.

---

## 9. Experiment Runner Logic

The experiment is implemented in `rq4_llm/experiment.py`. The function `run_rq4_llm_experiment` takes `num_seeds`, `results_dir`, `save_frames`, and `log_file` as arguments. It loads the trap scenario config. It initializes a results dictionary with experiment metadata and an empty `conditions` dict. For each condition (`hierarchical`, `decentralized`), it creates a copy of the config, sets `controller_mode` and `commander_type` appropriately, and for each seed index, creates a new config with the seed, instantiates `SimulationController`, calls `setup()`, and runs the simulation. For the first hierarchical seed only, if `log_file` is provided, stdout is redirected to the log file so that LLM prompts and responses are captured; for all other runs, stdout is discarded to a sink to avoid clutter.

After each run, the experiment extracts `final_metrics` from the controller. It computes `rescued_at_collapse` by reading the trajectory at step 9 (0-indexed) or the step before the collapse; `rescued_post_collapse` is `rescued - rescued_at_collapse`. These values are stored per seed. After all seeds for a condition, the experiment computes survivor_rate_mean, survivor_rate_std, rescued_mean, rescued_post_collapse_mean, and rescued_post_collapse_std, and stores them in `results['conditions'][condition_name]`. Finally, the results are written to `results/rq4_llm_results.json`.

If `save_frames` is True, the experiment runs additional simulations for each condition (with the base seed) and saves render frames at steps 9, 10, 11, and 20, plus the final step. For hierarchical mode, it also saves mental-map renders at those steps.

---

## 10. Analysis and Visualization

The analysis module (`rq4_llm/analysis.py`) provides `run_rq4_llm_analysis(results_path, plots_dir)`. It reads the results JSON file and extracts the conditions. For each condition, it gets `survivor_rate_mean` and `survivor_rate_std`. It creates a bar chart with two bars: one for Hierarchical (LLM) / Second-Order Thinking and one for Decentralized / Greedy Steps. The bars show survivor rate with error bars (standard deviation). The chart is saved as `rq4_llm_second_order_vs_greedy.png` in the plots directory. The plot is generated using matplotlib with a non-interactive backend (Agg) so it can run in headless environments.

---

## 11. Findings Generation

The `generate_findings.py` script reads `results/rq4_llm_results.json` and updates `rq4_llm/FINDINGS.md` with the actual results. It extracts survivor rates and rescued counts for both conditions. It then determines the conclusion based on a simple rule: if survivor_rate (hierarchical) > survivor_rate (decentralized), the conclusion is that agents exhibit Second-Order Thinking when guided by the LLM Commander; if the difference is within 5%, the conclusion is mixed; otherwise, Greedy Steps dominate. The script writes a formatted markdown document with the research question, experiment design, results table, and the chosen conclusion. This is typically run after the experiment completes, either manually or as part of the `run_rq4_llm.py` main flow (which catches exceptions so that a failed FINDINGS update does not abort the run).

---

## 12. Results Structure and Interpretation

The results JSON file has the following structure. The top level includes `experiment`, `metric`, `commander`, `scenario`, `conditions`, and `metadata`. Under `conditions`, each condition has `survivor_rate_mean`, `survivor_rate_std`, `rescued_mean`, `rescued_post_collapse_mean`, `rescued_post_collapse_std`, and `per_seed` (a list of per-seed records with seed, survivor_rate, rescued, rescued_at_collapse, rescued_post_collapse, total_victims, dead, alive_unrescued, simulation_steps).

In the example results from the codebase, hierarchical achieved survivor_rate_mean ≈ 20.9% and rescued_mean ≈ 4.5, while decentralized achieved survivor_rate_mean ≈ 47.0% and rescued_mean ≈ 9.5. Decentralized also had higher rescued_post_collapse_mean (5.5 vs 2.5). This indicates that in this particular trap scenario, Greedy Steps dominated: the decentralized agents performed better than the hierarchical agents. Possible explanations include: the LLM Commander may have made suboptimal assignments; the mental map may have been incomplete or delayed in learning about the collapse; the overhead of coordination (waiting for assignments, following paths that become invalid) may have cost more than the benefit of strategic planning; or the scenario may have been structured such that greedy local choices happened to avoid the trap. The experiment does not conclusively prove that greedy is always better—it shows that in this specific setup, with this LLM and this trap design, decentralized greedy behaviour outperformed hierarchical LLM coordination.

---

## 13. Entry Point and Usage

The main entry point is `run_rq4_llm.py` in the project root. It parses command-line arguments: `--seeds` (default 3), `--visualize` (only run analysis on existing results), `--save-frames`, `--log-file`, and `--results-dir`. If not in visualize mode, it calls `run_rq4_llm_experiment` with the given arguments, then `run_rq4_llm_analysis(results_path, plots_dir)`, and finally attempts to run `generate_findings.main()` to update FINDINGS.md. The outputs are: `results/rq4_llm_results.json`, `results/plots/rq4_llm_second_order_vs_greedy.png`, and `rq4_llm/FINDINGS.md` (if updated successfully).

---

## 14. Summary

The RQ4 LLM experiment is a controlled comparison of hierarchical (LLM Commander) vs. decentralized (greedy) coordination in an urban disaster response simulation. The trap scenario introduces a building collapse at step 10 that blocks a path, creating a situation where strategic planning could theoretically outperform local greed. The hierarchical condition uses an LLM that receives zone summaries, agent reports, and mental map state, and issues task assignments with precomputed paths. The decentralized condition has no commander; agents act autonomously using BFS to find the nearest fire or victim and A* to move toward it. The experiment runs both conditions for multiple seeds, aggregates survivor rates and rescue counts, and produces a bar chart and findings document. The codebase implements this logic across the controller, commander, field agents, mental map, experiment runner, analysis module, and findings generator, with each component playing a specific role in the end-to-end pipeline.
