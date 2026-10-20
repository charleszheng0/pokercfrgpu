"""Reproduce a fixed-game CFR+ iteration convergence sweep.

Example: python3 -m experiments.river_cfr_convergence --output experiments/river_cfr_convergence.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from experiments.river_range_scaling import (
    BIG_BLIND, BOARD, POT, STACKS, run_condition,
)

DEFAULT_BUDGETS = (20, 100, 500, 2000)

def run_sweep(*, seed=20261020, combos_per_range=4, budgets=DEFAULT_BUDGETS):
    rows = []
    for iterations in budgets:
        started = time.perf_counter()
        row = run_condition(combos_per_range, seed=seed, iterations=iterations)
        row["wall_seconds"] = time.perf_counter() - started
        rows.append(row)
    return {
        "experiment": "fixed_river_cfr_plus_iteration_sweep",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "seed": seed,
        "same_game": all(row["ranges"] == rows[0]["ranges"] for row in rows),
        "combos_per_range": combos_per_range,
        "board": BOARD,
        "pot_chips": POT,
        "effective_stacks_chips": STACKS,
        "big_blind_chips": BIG_BLIND,
        "variant": "cfr+",
        "iterations": list(budgets),
        "action_abstraction": "shared solver default: half-pot, pot, all-in; one raise",
        "results": rows,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--combos-per-range", type=int, default=4)
    parser.add_argument("--iterations", default="20,100,500,2000")
    parser.add_argument("--output", default="experiments/river_cfr_convergence.json")
    args = parser.parse_args()
    budgets = tuple(int(value) for value in args.iterations.split(","))
    record = run_sweep(seed=args.seed, combos_per_range=args.combos_per_range, budgets=budgets)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)
    for row in record["results"]:
        print(row["iterations"], "iterations", row["nash_conv_chips"],
              "chips NashConv", row["latency_ms"], "ms")

if __name__ == "__main__":
    main()
