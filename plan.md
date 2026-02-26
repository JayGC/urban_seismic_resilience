# Urban Seismic Resilience Framework — 2–3 Week Plan

**Team #6** · Satyam Srivastava, Jay Chaudhary, Seemandhar Jain, Adyasha Patra, Abhay Jain, Keshav Gupta

This document condenses the full technical plan into a **2–3 week** execution schedule. Scope is trimmed to deliver a working simulator, hierarchical agents, and core experiments within the time budget.

---

## Overview

| Week | Focus | Key Deliverables |
|------|--------|------------------|
| **Week 1** | Simulation & environment | OSM grid, seismic/collapse model, env API, basic viz |
| **Week 2** | Agents & communication | Commander + field agents, message protocol, LLM integration, baselines |
| **Week 3** | Experiments & wrap-up | RQ1–RQ4 runs, metrics, visualizations, plan.md and README |

---

## Week 1 — Simulation & Environment

**Goal:** Runnable environment with realistic topology and physics; no agents yet.

### Day 1–2: Setup and topology

- **Tech stack:** Python, Gymnasium-style env API, configs (YAML/JSON).
- **Repo layout:** `env/`, `agents/`, `experiments/`, `configs/`, `analysis/`.
- **OSM integration:**
  - Pick one city/bbox; use `osmnx` (or similar) for buildings + roads.
  - Discretize to a grid (e.g. 10 m cells); mark road vs building cells; build adjacency graph.
- **Data structures:** Cell (road/building, blocked, hazards, victims), Building (id, integrity 0–100, collapse state), dynamic graph updates when cells block.

### Day 3–4: Seismic and collapse

- **Seismic model:** Epicenter (x,y), magnitude M; impact decay I(r) = I_0 * exp(-kr).
- **Integrity:** Initialize per building; reduce with main shock + simple aftershock schedule.
- **Collapse:** Probabilistic failure when integrity < threshold; mark building collapsed.
- **Spillover:** Road cells adjacent to tall collapsed buildings become blocked; update graph.

### Day 5–7: Environment API and validation

- **Env API:** `reset(scenario_config)`, `step(actions)`, `render()`; support multiple scenarios (e.g. low/medium hazard).
- **Smoke tests:** Integrity decreases, collapses and spillover block roads, step loop runs for 20+ steps.
- **Minimal viz:** 2D grid (e.g. matplotlib) for debugging.

**Week 1 exit:** Stable env that can be stepped with placeholder/no-op "agents"; config-driven scenarios.

---

## Week 2 — Agents & Communication

**Goal:** Hierarchical control (Commander + field agents), message protocol, LLM commander, baselines.

### Day 1–2: Agent interface and hierarchy

- **Observations:**
  - Field agents: 3×3 local grid, own status, recent messages.
  - Commander: coarse zone summaries + compressed text reports only (no full grid).
- **Actions:** Field — move, scan/extinguish/treat, send_report; Commander — task assignments (move_to, search_zone, prioritize_fire/victims).
- **Two-level hierarchy:** Commander → field agents (Scout, Firefighter, Medic). Defer Zone Coordinators (Alpha/Beta/Gamma) unless time allows.
- **Message schema:** Typed, short templates (e.g. "Medic M2 at (x,y): 3 victims, 1 fire"); configurable max length.

### Day 3–4: Communication and dropout

- **Message bus:** Central queue; agents push reports; commander pulls each step.
- **Dropout:** Per-message drop with probability 0 / 0.2 / 0.5 for RQ2.
- **Modes:** Semantic (natural language) vs raw (state codes); toggled via config.

### Day 5–6: LLM and baselines

- **Commander LLM:** System prompt (role, constraints, action format); context = scenario summary + agent reports; call GPT/Claude; parse JSON commands; fallback to heuristic if parse fails.
- **Field agents:** Start with heuristic policies (greedy nearest-victim/nearest-fire, A* or BFS for paths) to stay within token budget.
- **A* tool:** Pathfinding on current grid; expose to commander or use in heuristic planner.
- **Decentralized baseline:** Same env, no commander; agents act on local info only (for RQ1 comparison).

### Day 7: Trap and black-swan configs

- **Trap scenarios:** Configs where greedy path is bad; detour saves more civilians (for RQ4).
- **Black swan:** One or two scripted events (e.g. sudden collapse/fire) to test re-planning (RQ3).

**Week 2 exit:** Hierarchical and decentralized controllers running; semantic/raw + dropout wired; scenario configs for RQs.

---

## Week 3 — Experiments & Wrap-Up

**Goal:** Run RQ1–RQ4 (reduced but complete), compute metrics, document and visualize.

### Day 1–2: Experiment harness and RQ1–RQ2

- **Harness:** Config-driven runner; multiple seeds per condition; save logs (trajectories, events) to `results/`.
- **RQ1 — Resilience under load:**
  - 2–3 task-density levels (e.g. low/medium/high victim and hazard count).
  - Compare hierarchical (LLM commander) vs decentralized; metric: civilians saved per minute, survival rate, mean agent idle time.
- **RQ2 — Semantic resilience:**
  - Semantic vs raw messaging at 0%, 20%, 50% dropout.
  - Metrics: swarm coherence (e.g. conflicting assignments), commander mental map vs ground truth (map fidelity).

### Day 3–4: RQ3–RQ4 and metrics

- **RQ3 — Black swans:** Run black-swan scenarios; measure re-planning latency (time from event to new effective plan); compare hierarchical vs heuristic commander.
- **RQ4 — Strategic horizon:** Run trap scenarios; record global vs local efficiency; note sacrificial moves (locally worse, globally better); compare LLM commander vs greedy.
- **Central metrics:** Survival rate (rescued/total), map fidelity, agent idle time — ensure all logged and aggregated.

### Day 5–6: Visualization and documentation

- **Plots:** Success vs task density (RQ1); coherence vs dropout (RQ2); re-planning latency (RQ3); global vs local efficiency (RQ4).
- **Viz:** Episode replay or key-frame views (2D grid + agents + hazards).
- **README:** How to install, run env, run experiments, reproduce results.
- **plan.md:** This document; optionally add a short "what we cut" section (e.g. Zone Coordinators, full LLM field agents).

### Day 7: Buffer and polish

- **Token check:** Approximate 6M budget (10 agents × 150 tokens × 20 steps × 200 runs); reduce episodes or context if over.
- **Repo cleanup:** Remove dead code; ensure configs and scripts reproduce reported numbers.

**Week 3 exit:** Results for all four RQs, visualizations, README and plan.md in repo.

---

## Scope Adjustments (2–3 weeks)

- **In:** OSM-based grid, seismic decay, integrity, collapse + spillover, two-level hierarchy (Commander + field), semantic/raw + dropout, heuristic field agents, LLM commander, A* support, RQ1–RQ4 in reduced form.
- **Out or deferred:** Zone Coordinators (Alpha/Beta/Gamma), LLM-driven field agents, large-scale ablation sweeps, multiple cities.
- **Compute:** Keep ~200 runs × 20 steps × 10 agents × 150 tokens; trim runs or context if needed to stay near $90.

---

*Urban Seismic Resilience: A Hierarchical Multi-Agent Coordination Framework — Team #6*
