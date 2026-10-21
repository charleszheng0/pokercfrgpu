"""Deepen the highest-NashConv position from the held-out river suite."""

import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve


def main():
    source = json.loads(Path("experiments/postflop_multiboard_validation.json").read_text())
    candidate = max((row for row in source["positions"] if row["street"] == "river"),
                    key=lambda row: row["nash_conv_chips"])
    base = SolveRequest(
        board=tuple(candidate["board"]), street="river",
        pot_before_street=candidate["pot"],
        stacks_at_street_start=tuple(candidate["stacks"]),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=tuple(WeightedRange(tuple(tuple(hand) for hand in row["hands"]),
                                   tuple(row["weights"]))
                     for row in candidate["ranges"]),
        abstraction=ActionAbstraction(),
        config=SolverConfig(iterations=200, max_joint_deals=4),
        acting_hand=tuple(candidate["hero_hand"]),
        range_source="seeded_held_out_profile",
    )
    rows = []
    for iterations in (200, 500, 1000):
        request = replace(base, config=replace(base.config, iterations=iterations))
        started = perf_counter()
        result = solve(request)
        rows.append({
            "iterations": result.iterations,
            "nash_conv_chips": result.nash_conv_chips,
            "latency_ms": result.latency_ms,
            "wall_latency_ms": (perf_counter() - started) * 1000,
            "selected_hand_actions": [
                {"kind": item.action.kind, "amount": item.action.amount,
                 "frequency": item.frequency, "ev_chips": item.ev_chips}
                for item in result.selected_hand.actions
            ],
        })
    reference = rows[-1]
    reference_rows = {(r["kind"], r["amount"]): r
                      for r in reference["selected_hand_actions"]}
    comparisons = []
    for row in rows[:-1]:
        current = {(r["kind"], r["amount"]): r
                   for r in row["selected_hand_actions"]}
        keys = tuple(reference_rows)
        best_ev = max(reference_rows[key]["ev_chips"] for key in keys)
        policy_ev = sum(current[key]["frequency"] * reference_rows[key]["ev_chips"]
                        for key in keys)
        comparisons.append({
            "iterations": row["iterations"],
            "action_ev_mae_chips": sum(abs(current[k]["ev_chips"]
                                             - reference_rows[k]["ev_chips"])
                                        for k in keys) / len(keys),
            "strategy_l1": sum(abs(current[k]["frequency"]
                                    - reference_rows[k]["frequency"])
                               for k in keys),
            "root_policy_regret_chips": best_ev - policy_ev,
        })
    record = {
        "experiment": "deepen_highest_nashconv_heldout_river_case",
        "date": date.today().isoformat(), "source_position_id": candidate["position_id"],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "position": {
            "board": candidate["board"], "pot": candidate["pot"],
            "stacks": candidate["stacks"], "ranges": candidate["ranges"],
            "hero_hand": candidate["hero_hand"],
            "compatible_joint_deals": candidate["joint_deals"],
        },
        "runs": rows,
        "same_game_comparisons_vs_1000": comparisons,
        "limitations": [
            "Only deepens one selected synthetic river fixture.",
            "The 1,000-iteration result is a finite-iteration reference, not ground truth.",
            "No real opponent range inference or external solver comparison.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["iterations"], "NashConv", round(row["nash_conv_chips"], 6),
              "ms", round(row["wall_latency_ms"], 1), flush=True)
    print("comparisons", comparisons)
    print(output)


if __name__ == "__main__":
    main()
