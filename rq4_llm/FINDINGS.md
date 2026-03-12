# RQ4: Strategic Horizon Analysis — Research Question Answer

## Research Question

**RQ4: Strategic Horizon Analysis**

- **Metric:** Global vs. Local Efficiency
- **Expectation:** Visualize specific "Trap Scenarios" to see if agents exhibit **Second-Order Thinking** (ignoring a nearby small reward for a distant large reward) vs. **Greedy Steps**.

## Experiment Design

We compare two conditions in the trap scenario:

1. **Hierarchical (LLM Commander)** — Global/strategic coordination. The LLM Commander receives zone summaries, agent reports, and mental map state, then issues task assignments. Agents can follow distant high-value targets assigned by the commander (Second-Order Thinking).

2. **Decentralized** — Local/greedy coordination. No commander. Agents act autonomously, choosing the nearest victim or fire at each step (Greedy Steps).

The trap scenario features a building collapse at step 10 that blocks a path. This creates a situation where greedy local behaviour may be suboptimal—agents heading toward the nearest victim may get blocked—while strategic coordination could route agents around the collapse.

## Results

| Condition | Survivor Rate (mean ± std) | Rescued (mean) | Post-Collapse Rescues (mean) |
|-----------|----------------------------|----------------|------------------------------|
| Hierarchical (LLM) | 12.46% ± 6.37% | 2.7 | 2.1 |
| Decentralized (Greedy) | 48.55% ± 16.30% | 9.4 | 6.0 |

## Answer to the Research Question

**Do agents exhibit Second-Order Thinking vs. Greedy Steps in the trap scenario?**

**Greedy Steps dominate.** The decentralized condition achieved a higher survivor rate than hierarchical (LLM). Decentralized agents adapt quickly without coordination overhead; in this scenario, greedy local behaviour was sufficient or more effective.
