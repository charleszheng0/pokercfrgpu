"""Compare vanilla CFR, CFR+, and DCFR on one fixed tiny river game."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from poker_cpu.engine import Action
from poker_cpu.river_solver import RiverCFR, RiverGame, nash_conv

BOARD = "As Kh Qd 7c 2s"
RANGES = (
    (("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")),
    (("Jd", "Td"), ("8h", "8s"), ("4c", "4d")),
)


def run(checkpoints):
    rows = []
    for variant in ("cfr", "cfr+", "dcfr"):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        solver = RiverCFR(game, variant=variant)
        cumulative_train_seconds = 0.0
        previous = 0
        for target in checkpoints:
            train_started = time.perf_counter()
            solver.train(target - previous)
            cumulative_train_seconds += time.perf_counter() - train_started
            previous = target
            analysis_started = time.perf_counter()
            report = solver.analyze_root(chosen_action=Action("check"))
            analysis_seconds = time.perf_counter() - analysis_started
            rows.append({
                "variant": variant,
                "iterations": target,
                "cumulative_train_seconds": cumulative_train_seconds,
                "analysis_seconds_including_exact_nash_conv": analysis_seconds,
                "nash_conv_chips": report["nash_conv"],
                "seat0_ev_chips": game.expected_value(solver.average, solver.average),
                "root_action_frequencies": {
                    str(action): frequency
                    for action, frequency in report["action_frequencies"].items()
                },
                "root_conditional_action_evs_chips": {
                    str(action): ev
                    for action, ev in report["conditional_action_evs"].items()
                },
                "range_average_check_gap_chips": report["range_average_action_gap"],
            })
    for variant in ("cfr", "cfr+", "dcfr"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        reference = next(
            row["root_conditional_action_evs_chips"]
            for row in variant_rows if row["iterations"] == max(checkpoints)
        )
        for row in variant_rows:
            errors = [
                abs(row["root_conditional_action_evs_chips"][action] - reference[action])
                for action in reference
            ]
            row["mean_action_ev_abs_diff_vs_deepest_chips"] = sum(errors) / len(errors)
            row["max_action_ev_abs_diff_vs_deepest_chips"] = max(errors)
    return {
        "experiment": "fixed_tiny_range_river_cfr_update_comparison",
        "board": BOARD,
        "ranges": RANGES,
        "pot_chips": 8,
        "effective_stack_chips": 10,
        "abstraction": "min/half-pot/pot/all-in; one raise maximum",
        "compatible_joint_deals": len(RiverGame(BOARD, RANGES).deals),
        "checkpoints": rows,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[20, 100, 500])
    parser.add_argument("--output", type=Path, default=Path("experiments/river_variants.json"))
    args = parser.parse_args()
    if not args.checkpoints or args.checkpoints != sorted(set(args.checkpoints)) or min(args.checkpoints) < 1:
        parser.error("checkpoints must be sorted, unique, positive integers")
    result = run(args.checkpoints)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["checkpoints"]:
        print(json.dumps({key: row[key] for key in (
            "variant", "iterations", "cumulative_train_seconds", "nash_conv_chips",
            "seat0_ev_chips", "range_average_check_gap_chips"
        )}))
