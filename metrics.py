"""
End-of-simulation metrics:
  1. Survivor rate — fraction of victims rescued by medic/firefighter.
  2. Mental-map fidelity — how closely the commander's mental map matches
     the ground-truth grid at simulation end.
  3. Commander LLM stats — average latency and average token usage
     (input + output) per commander response over all steps.

Usage:
    from metrics import compute_final_metrics
    result = compute_final_metrics(controller)   # returns dict
    # result is also written to  results/final_metrics.json
"""

import json
import os
from typing import Any, Dict, Optional

from env.grid import CellType, HazardType


# ── 1. Survivor Rate ────────────────────────────────────────────────────────

def _compute_survivor_rate(env) -> dict:
    """
    Compute rescue statistics from the environment at simulation end.

    Returns dict with:
      total_victims, rescued, dead, alive_unrescued, survivor_rate
    """
    m = env.get_metrics()
    return {
        "total_victims": m["total_victims"],
        "rescued": m["rescued"],
        "dead": m["dead"],
        "alive_unrescued": m["alive_unrescued"],
        "survivor_rate": m["survival_rate"],          # rescued / total_victims
    }


# ── 2. Mental-Map Fidelity ──────────────────────────────────────────────────

def _compute_mental_map_fidelity(mental_map, ground_truth_grid) -> dict:
    """
    Direct cell-by-cell diff between commander's mental map and ground-truth grid.

    Compares EVERY cell (not just explored ones):
      - If mental map hasn't explored a cell, that's a miss.
      - If it has, we check whether hazard, blockage, and victim info are correct.

    Returns raw counts and a single overall accuracy = matching_cells / total_cells.
    """
    if mental_map is None:
        return {"fidelity": None, "note": "no mental map available"}

    total_cells = len(ground_truth_grid.cells)

    # ── Ground-truth counts ──
    gt_fires = 0
    gt_debris = 0
    gt_blocked = 0
    gt_victim_cells = 0

    # ── What the mental map got right ──
    fires_correct = 0       # cell is FIRE in both GT and MM
    debris_correct = 0      # cell is DEBRIS in both GT and MM
    blocked_correct = 0     # road cell blocked in both GT and MM
    victim_cells_correct = 0  # victim presence matches in both

    # ── What the mental map missed (GT positive, MM negative or unexplored) ──
    fires_missed = 0
    debris_missed = 0
    blocked_missed = 0
    victim_cells_missed = 0

    # ── What the mental map got wrong (MM positive, GT negative) ──
    fires_false_pos = 0
    debris_false_pos = 0
    blocked_false_pos = 0
    victim_cells_false_pos = 0

    # ── Cell-level match (every attribute matches) ──
    cells_fully_correct = 0
    cells_explored = 0

    for pos, gt_cell in ground_truth_grid.cells.items():
        mm_cell = mental_map.cells.get(pos)
        explored = mm_cell is not None and mm_cell.explored

        if explored:
            cells_explored += 1

        gt_hazard = gt_cell.hazard
        mm_hazard = (mm_cell.hazard if (explored and mm_cell.hazard is not None)
                     else HazardType.NONE)

        gt_is_fire = gt_hazard == HazardType.FIRE
        gt_is_debris = gt_hazard == HazardType.DEBRIS
        mm_is_fire = mm_hazard == HazardType.FIRE
        mm_is_debris = mm_hazard == HazardType.DEBRIS

        is_gt_blocked = (gt_cell.cell_type == CellType.ROAD and gt_cell.blocked)
        is_mm_blocked = (explored and gt_cell.cell_type == CellType.ROAD
                         and mm_cell.blocked is True)

        gt_has_victims = any(not v.rescued and v.health > 0 for v in gt_cell.victims)
        mm_has_victims = explored and len(mm_cell.victims) > 0

        # --- fire ---
        if gt_is_fire:
            gt_fires += 1
            if mm_is_fire:
                fires_correct += 1
            else:
                fires_missed += 1
        elif mm_is_fire:
            fires_false_pos += 1

        # --- debris ---
        if gt_is_debris:
            gt_debris += 1
            if mm_is_debris:
                debris_correct += 1
            else:
                debris_missed += 1
        elif mm_is_debris:
            debris_false_pos += 1

        # --- blocked roads ---
        if is_gt_blocked:
            gt_blocked += 1
            if is_mm_blocked:
                blocked_correct += 1
            else:
                blocked_missed += 1
        elif is_mm_blocked:
            blocked_false_pos += 1

        # --- victim presence ---
        if gt_has_victims:
            gt_victim_cells += 1
            if mm_has_victims:
                victim_cells_correct += 1
            else:
                victim_cells_missed += 1
        elif mm_has_victims:
            victim_cells_false_pos += 1

        # --- full cell match ---
        if explored:
            hazard_ok = (mm_hazard == gt_hazard)
            block_ok = (is_mm_blocked == is_gt_blocked) if gt_cell.cell_type == CellType.ROAD else True
            victim_ok = (mm_has_victims == gt_has_victims)
            if hazard_ok and block_ok and victim_ok:
                cells_fully_correct += 1

    explored_fraction = cells_explored / total_cells if total_cells else 0.0
    cell_accuracy = cells_fully_correct / total_cells if total_cells else 0.0

    def _safe_div(a, b):
        return round(a / b, 4) if b else 0.0

    return {
        "total_cells": total_cells,
        "cells_explored": cells_explored,
        "explored_fraction": round(explored_fraction, 4),
        # Direct cell-level accuracy
        "cells_fully_correct": cells_fully_correct,
        "cell_accuracy": round(cell_accuracy, 4),
        # Fire
        "gt_fires": gt_fires,
        "fires_found": fires_correct,
        "fires_missed": fires_missed,
        # "fires_false_positive": fires_false_pos,
        "fire_recall": _safe_div(fires_correct, gt_fires),
        # Debris
        "gt_debris": gt_debris,
        "debris_found": debris_correct,
        "debris_missed": debris_missed,
        # "debris_false_positive": debris_false_pos,
        "debris_recall": _safe_div(debris_correct, gt_debris),
        # Blocked roads
        "gt_blocked_roads": gt_blocked,
        "blocked_found": blocked_correct,
        "blocked_missed": blocked_missed,
        # "blocked_false_positive": blocked_false_pos,
        "blockage_recall": _safe_div(blocked_correct, gt_blocked),
        # Victim cells
        "gt_victim_cells": gt_victim_cells,
        "victim_cells_found": victim_cells_correct,
        "victim_cells_missed": victim_cells_missed,
        # "victim_cells_false_positive": victim_cells_false_pos,
        "victim_cell_recall": _safe_div(victim_cells_correct, gt_victim_cells),
    }


# ── 3. Commander LLM Stats ──────────────────────────────────────────────────

def _compute_commander_llm_stats(commander) -> dict:
    """
    Aggregate latency and token usage from the commander's call_log.

    Works for LLMCommander (which keeps a per-call log).
    For HeuristicCommander (no LLM calls), returns zeroes.
    """
    call_log = getattr(commander, "call_log", [])
    n = len(call_log)

    if n == 0:
        return {
            "num_llm_calls": 0,
            "avg_latency_seconds": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "per_call_log": [],
        }

    total_latency = sum(c.get("latency_seconds", 0.0) for c in call_log)
    total_input = sum(c.get("input_tokens", 0) for c in call_log)
    total_output = sum(c.get("output_tokens", 0) for c in call_log)
    total_tok = sum(c.get("total_tokens", 0) for c in call_log)

    return {
        "num_llm_calls": n,
        "avg_latency_seconds": round(total_latency / n, 4),
        "avg_input_tokens": round(total_input / n, 2),
        "avg_output_tokens": round(total_output / n, 2),
        "avg_total_tokens": round(total_tok / n, 2),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tok,
        "per_call_log": call_log,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def compute_final_metrics(controller, output_path: str = "results/final_metrics.json") -> dict:
    """
    Compute all three end-of-simulation metrics and write them to a JSON file.

    Args:
        controller: SimulationController instance (after simulation has run).
        output_path: Where to write the JSON output.

    Returns:
        Dictionary with keys: survivor_rate, mental_map_fidelity, commander_llm_stats
    """
    result: Dict[str, Any] = {}

    # 1. Survivor rate
    result["survivor_rate"] = _compute_survivor_rate(controller.env)

    # 2. Mental-map fidelity
    mental_map = controller.commander.mental_map if controller.commander else None
    result["mental_map_fidelity"] = _compute_mental_map_fidelity(
        mental_map, controller.env.grid
    )

    # 3. Commander LLM stats
    result["commander_llm_stats"] = _compute_commander_llm_stats(
        controller.commander
    ) if controller.commander else _compute_commander_llm_stats(None)

    # Metadata
    result["simulation_steps"] = controller.step_count
    result["max_steps"] = controller.config.get("max_steps", "N/A")

    # Write to JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n📊 Final metrics written to {output_path}")

    return result
