"""Compare CFR updates on identical held-out river/turn/cutoff fixtures."""
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve

SOURCE = Path(__file__).with_name("postflop_multiboard_validation.json")
VARIANTS = ("cfr", "cfr+", "dcfr")


def request_from_row(row, variant):
    return SolveRequest(
        board=tuple(row["board"]), street=row["street"],
        pot_before_street=row["pot"], stacks_at_street_start=tuple(row["stacks"]),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=tuple(WeightedRange(tuple(tuple(hand) for hand in seat["hands"]),
                                   tuple(seat["weights"])) for seat in row["ranges"]),
        abstraction=ActionAbstraction(),
        config=SolverConfig(iterations=row["iterations"], max_joint_deals=4,
                            continuation_mode=row["mode"], variant=variant),
        acting_hand=tuple(row["hero_hand"]), range_source="seeded_held_out_profile",
    )


def main():
    source = json.loads(SOURCE.read_text())
    rows = []
    modes = ("river_full_tree", "turn_full_tree", "turn_showdown_cutoff",
             "flop_showdown_cutoff")
    for mode in modes:
        street = mode.split("_", 1)[0]
        expected_mode = "full_tree" if mode.endswith("full_tree") else "showdown_cutoff"
        fixtures = [row for row in source["positions"]
                    if row["street"] == street and row["mode"] == expected_mode]
        for fixture in fixtures:
            for variant in VARIANTS:
                request = request_from_row(fixture, variant)
                started = perf_counter()
                result = solve(request)
                wall = (perf_counter() - started) * 1000
                rows.append({
                    "position_id": fixture["position_id"], "street": fixture["street"],
                    "mode": fixture["mode"], "variant": variant,
                    "iterations": result.iterations,
                    "joint_deals": result.joint_deals,
                    "nash_conv_chips": result.nash_conv_chips,
                    "solver_latency_ms": result.latency_ms,
                    "wall_latency_ms": wall,
                    "selected_actions": [
                        {"kind": item.action.kind, "amount": item.action.amount,
                         "frequency": item.frequency, "ev_chips": item.ev_chips}
                        for item in result.selected_hand.actions
                    ] if result.selected_hand else [],
                })
                print(fixture["position_id"], variant,
                      "NashConv", round(result.nash_conv_chips, 6),
                      "ms", round(wall, 1), flush=True)
    groups = {}
    for mode in modes:
        street = mode.split("_", 1)[0]
        expected_mode = "full_tree" if mode.endswith("full_tree") else "showdown_cutoff"
        for variant in VARIANTS:
            selected = [row for row in rows if row["street"] == street
                        and row["mode"] == expected_mode and row["variant"] == variant]
            groups[f"{mode}:{variant}"] = {
                "positions": len(selected),
                "nash_conv_median_chips": median(row["nash_conv_chips"] for row in selected),
                "nash_conv_max_chips": max(row["nash_conv_chips"] for row in selected),
                "latency_median_ms": median(row["wall_latency_ms"] for row in selected),
            }
    record = {
        "experiment": "heldout_postflop_cfr_variant_comparison",
        "date": date.today().isoformat(), "source_seed": source["seed"],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": "Same three seeded fixtures per street/mode and identical action trees, exact chance, ranges and budgets. Exact NashConv is measured within each finite game. Each condition is one run per position; runtimes are not distributions.",
        "groups": groups, "runs": rows,
        "limitations": [
            "Tiny synthetic ranges only; no broad-range or live range inference.",
            "Cutoff modes omit later betting and are separate games from full trees.",
            "Single run per fixture/variant; runtime noise and seed uncertainty are not estimated.",
            "NashConv values compare variants only within the same fixture and game.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
