"""Measure convergence and action stability on seeded 10x10/11x11 river games."""

import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import (ActionAbstraction, BettingContext, SolveRequest,
                             SolverConfig, solve)

BUDGETS_BY_WIDTH = {10: (100, 300, 500), 11: (100, 300, 500, 1000)}


def main():
    source = json.loads(Path("experiments/river_range_width_extension.json").read_text())
    cases = []
    for raw in source["scenarios"]:
        budgets = BUDGETS_BY_WIDTH[raw["range_width_per_seat"]]
        ranges = tuple(WeightedRange(
            tuple(tuple(hand) for hand in row["hands"]), tuple(row["weights"])
        ) for row in raw["ranges"])
        base = SolveRequest(
            board=tuple(raw["board"]), street="river", pot_before_street=raw["pot"],
            stacks_at_street_start=tuple(raw["stacks"]),
            betting=BettingContext(button=1, first_actor=0, current_player=0),
            ranges=ranges, abstraction=ActionAbstraction.shark_style_postflop(),
            config=SolverConfig(iterations=budgets[0],
                                max_joint_deals=raw["compatible_joint_deals"]),
            acting_hand=ranges[0].hands[0],
            range_source="seeded_river_range_width_extension",
        )
        runs = []
        for budget in budgets:
            request = replace(base, config=replace(base.config, iterations=budget))
            started = perf_counter()
            result = solve(request)
            runs.append({
                "iteration_budget": budget,
                "iterations": result.iterations,
                "nash_conv_chips": result.nash_conv_chips,
                "latency_ms": result.latency_ms,
                "wall_latency_ms": (perf_counter() - started) * 1000,
                "selected_hand_actions": [{
                    "kind": row.action.kind, "amount": row.action.amount,
                    "frequency": row.frequency, "ev_chips": row.ev_chips,
                } for row in result.selected_hand.actions],
            })
            print(raw["range_width_per_seat"], budget,
                  "NashConv", round(result.nash_conv_chips, 6),
                  "latency_ms", round(runs[-1]["wall_latency_ms"], 1), flush=True)
        reference = {(row["kind"], row["amount"]): row
                     for row in runs[-1]["selected_hand_actions"]}
        reference_budget = runs[-1]["iterations"]
        comparisons = []
        for run in runs[:-1]:
            current = {(row["kind"], row["amount"]): row
                       for row in run["selected_hand_actions"]}
            keys = tuple(reference)
            best_ev = max(reference[key]["ev_chips"] for key in keys)
            current_value_on_reference = sum(
                current[key]["frequency"] * reference[key]["ev_chips"] for key in keys
            )
            comparisons.append({
                "iterations": run["iterations"],
                f"action_ev_mae_chips_vs_{reference_budget}": sum(
                    abs(current[key]["ev_chips"] - reference[key]["ev_chips"])
                    for key in keys
                ) / len(keys),
                f"strategy_l1_vs_{reference_budget}": sum(
                    abs(current[key]["frequency"] - reference[key]["frequency"])
                    for key in keys
                ),
                f"policy_regret_chips_on_{reference_budget}_evs": best_ev - current_value_on_reference,
            })
        cases.append({
            "range_width_per_seat": raw["range_width_per_seat"],
            "compatible_joint_deals": raw["compatible_joint_deals"],
            "board": raw["board"], "pot": raw["pot"], "stacks": raw["stacks"],
            "ranges": raw["ranges"], "hero_hand": raw["ranges"][0]["hands"][0],
            "runs": runs,
            "reference_iterations": reference_budget,
            "comparisons_vs_finite_reference": comparisons,
        })

    record = {
        "experiment": "deepen_exact_river_range_width_extension",
        "date": date.today().isoformat(),
        "source_revision": source["revision"],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "budgets_by_width": BUDGETS_BY_WIDTH, "scenarios": cases,
        "methodology": (
            "Repeat each exact seeded 10x10/11x11 Shark-inspired river game at 100, 300, "
            "and 500 CFR+ iterations for 10x10; 100/300/500/1,000 for 11x11. Compare "
            "selected-combo action values and strategy against the deepest finite run for "
            "each width."
        ),
        "limitations": [
            "One seed/position per range width; no population inference.",
            "The 500-iteration point is finite, not ground truth.",
            "Comparison covers one hero combo per supplied weighted range.",
            "Exact NashConv is for each finite supplied-range restricted game only.",
        ],
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print("comparison", case["range_width_per_seat"],
              json.dumps(case["comparisons_vs_finite_reference"]))
    print(path)


if __name__ == "__main__":
    main()
