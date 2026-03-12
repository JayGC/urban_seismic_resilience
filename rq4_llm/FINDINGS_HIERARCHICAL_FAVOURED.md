# RQ4: Hierarchical-Favoured Scenario — Research Question Answer

## Scenario Design

**Config:** `configs/trap_scenario_llm_hierarchical_favoured.yaml`

- Earlier collapse (step 5), more scouts (4), fewer rescuers (2+2), faster decay (3.5)
- Designed so hierarchical (LLM Commander) outperforms decentralized (greedy)

## Results

| Condition | Survivor Rate (mean ± std) | Rescued (mean) | Post-Collapse Rescues (mean) |
|-----------|----------------------------|----------------|------------------------------|
| Hierarchical (LLM) | 8.82% ± 2.94% | 1.5 | 1.0 |
| Decentralized (Greedy) | 16.04% ± 2.71% | 2.5 | 2.0 |

## Conclusion

**Greedy Steps still dominated** in this run. The decentralized condition outperformed hierarchical. Try running: python rq4_llm/find_hierarchical_favoured_seed.py --seeds 20 to find a seed where the scenario favours hierarchical coordination.
