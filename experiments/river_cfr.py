"""Reproducible tiny-range Hold'em river solve with exact best responses."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from poker_cpu.river_solver import RiverCFR, RiverGame, nash_conv

BOARD = "As Kh Qd 7c 2s"
RANGES = (
    (("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")),
    (("Jd", "Td"), ("8h", "8s"), ("4c", "4d")),
)


def run(checkpoints):
    game = RiverGame(BOARD, RANGES, pot=8, stack=10)
    solver = RiverCFR(game)
    rows = []
    start = time.perf_counter()
    for target in checkpoints:
        solver.train(target - solver.iterations)
        rows.append(
            {
                "iterations": target,
                "nash_conv": nash_conv(game, solver.average, solver.average),
                "seat0_ev": game.expected_value(solver.average, solver.average),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
    values = solver.root_hand_values()
    return {
        "experiment": "tiny_range_holdem_river_cfr",
        "board": BOARD,
        "ranges": RANGES,
        "pot": game.pot,
        "effective_stack": game.stack,
        "action_abstraction": "min/half-pot/pot/effective-all-in; one raise",
        "joint_deals": len(game.deals),
        "checkpoints": rows,
        "root_conditional_ev": [
            {"player": p, "hand": list(hand), "ev": ev}
            for p, player in enumerate(values)
            for hand, ev in player.items()
        ],
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
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/river_cfr.json")
    )
    args = parser.parse_args()
    if (
        not args.checkpoints
        or args.checkpoints != sorted(set(args.checkpoints))
        or min(args.checkpoints) < 1
    ):
        parser.error("checkpoints must be sorted, unique, positive integers")
    result = run(args.checkpoints)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "root_conditional_ev"}, indent=2
        )
    )
