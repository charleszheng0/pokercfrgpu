"""Record exact Kuhn CFR convergence against infoset-constrained best responses."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from small_games.kuhn_reference import CFR, nash_conv


def run(checkpoints):
    solver = CFR()
    rows = []
    start = time.perf_counter()
    for target in checkpoints:
        solver.train(target - solver.iterations)
        nc, ev, br0, min0 = nash_conv(solver.average(), solver.average())
        rows.append(
            {
                "iterations": target,
                "nash_conv": nc,
                "seat0_ev": ev,
                "best_response_seat0_ev": br0,
                "worst_seat0_ev": min0,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
    return {
        "experiment": "kuhn_synchronous_cfr",
        "checkpoints": rows,
        "known_game_value": -1 / 18,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=[100, 1000, 10000]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/kuhn_cfr.json")
    )
    args = parser.parse_args()
    if args.checkpoints != sorted(set(args.checkpoints)) or min(args.checkpoints) < 1:
        parser.error("checkpoints must be sorted, unique, positive integers")
    result = run(args.checkpoints)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
