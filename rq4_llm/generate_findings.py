"""
Update FINDINGS.md with actual results from rq4_llm_results.json.
Run after: python run_rq4_llm.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'rq4_llm_results.json'
)
FINDINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FINDINGS.md')


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Results not found: {RESULTS_PATH}. Run python run_rq4_llm.py first.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    conditions = results.get('conditions', {})
    h = conditions.get('hierarchical', {})
    d = conditions.get('decentralized', {})

    sr_h = h.get('survivor_rate_mean', 0)
    std_h = h.get('survivor_rate_std', 0)
    rescued_h = h.get('rescued_mean', 0)
    post_h = h.get('rescued_post_collapse_mean', 0)

    sr_d = d.get('survivor_rate_mean', 0)
    std_d = d.get('survivor_rate_std', 0)
    rescued_d = d.get('rescued_mean', 0)
    post_d = d.get('rescued_post_collapse_mean', 0)

    # Determine conclusion
    if sr_h > sr_d:
        conclusion = (
            "**Agents exhibit Second-Order Thinking when guided by the LLM Commander.** "
            "The hierarchical (LLM) condition achieved a higher survivor rate than decentralized (greedy). "
            "Strategic coordination—assigning agents to distant high-value targets and routing around the collapse—"
            "outperformed greedy local behaviour."
        )
    elif abs(sr_h - sr_d) < 0.05:
        conclusion = (
            "**Mixed results.** Hierarchical (LLM) and decentralized (greedy) achieved similar survivor rates. "
            "Both approaches yield comparable outcomes in this trap scenario; "
            "the benefit of Second-Order Thinking may be scenario-dependent."
        )
    else:
        conclusion = (
            "**Greedy Steps dominate.** The decentralized condition achieved a higher survivor rate than "
            "hierarchical (LLM). Decentralized agents adapt quickly without coordination overhead; "
            "in this scenario, greedy local behaviour was sufficient or more effective."
        )

    findings = f"""# RQ4: Strategic Horizon Analysis — Research Question Answer

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
| Hierarchical (LLM) | {sr_h:.2%} ± {std_h:.2%} | {rescued_h:.1f} | {post_h:.1f} |
| Decentralized (Greedy) | {sr_d:.2%} ± {std_d:.2%} | {rescued_d:.1f} | {post_d:.1f} |

## Answer to the Research Question

**Do agents exhibit Second-Order Thinking vs. Greedy Steps in the trap scenario?**

{conclusion}
"""

    with open(FINDINGS_PATH, 'w') as f:
        f.write(findings)

    print(f"FINDINGS.md updated with results from {RESULTS_PATH}")


if __name__ == '__main__':
    main()
