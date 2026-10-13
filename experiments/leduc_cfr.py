"""Save exact exploitability checkpoints for synchronous Leduc CFR."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from small_games.compare_cfr_nfsp import SynchronousCFR, nash_conv


def run(checkpoints):
    solver = SynchronousCFR()
    started = time.perf_counter()
    rows = []
    for target in checkpoints:
        solver.train(target - solver.iterations)
        nc, br0, br1, ev = nash_conv(solver.average, solver.average)
        rows.append(
            {
                "iterations": target,
                "nash_conv": nc,
                "exploitability_nash_conv_over_2": nc / 2,
                "seat0_ev": ev,
                "best_response_seat0_ev": br0,
                "best_response_seat1_ev": br1,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return {
        "experiment": "leduc_synchronous_cfr",
        "checkpoints": rows,
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
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[20, 100, 500])
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/leduc_cfr.json")
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
    print(json.dumps(result, indent=2))
