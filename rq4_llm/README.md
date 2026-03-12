# RQ4 Strategic Horizon Analysis (LLM Commander)

## Overview

This module runs **RQ4: Strategic Horizon Analysis** with the **LLM Commander** to answer:

> Do agents exhibit **Second-Order Thinking** (ignoring a nearby small reward for a distant large reward) vs. **Greedy Steps** in trap scenarios?

- **Metric:** Global vs. Local Efficiency
- **Conditions:** Hierarchical (LLM Commander) vs. Decentralized (greedy)


## Setup

Uses `configs/trap_scenario_llm.yaml` (and `trap_scenario_llm_hierarchical_favoured.yaml` for full comparison).  
Requires API access for the LLM Commander (see `agents/commander.py`).

## Usage

### Single scenario (greedy-favoured)
```bash
python run_rq4_llm.py                    # Run experiment (3 seeds)
python run_rq4_llm.py --seeds 5          # 5 seeds
python run_rq4_llm.py --log-file out_rq4_llm.log   # Save LLM logs
python run_rq4_llm.py --visualize        # Plots from existing results
```

### Full comparison (both scenarios for final report)
```bash
python run_rq4_llm_full_comparison.py              # Run both scenarios (3 seeds each)
python run_rq4_llm_full_comparison.py --seeds 2   # Fewer seeds (faster)
python run_rq4_llm_full_comparison.py --scenario hierarchical  # Only hierarchical-favoured
python run_rq4_llm_full_comparison.py --visualize  # Plots from existing results
```


## Outputs

- `results/rq4_llm_full_comparison_report.json` — combined report
- `results/plots/rq4_llm_second_order_vs_greedy.png` — single scenario chart
- `results/plots/rq4_llm_full_comparison.png` — side-by-side comparison

## Difference from Other RQ4 Modules

| Module | Commander | Compares |
|--------|-----------|----------|
| `rq4_llm/` | **LLM** | Hierarchical (LLM) vs Decentralized |
