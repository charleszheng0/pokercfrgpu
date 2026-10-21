"""Quality-targeted retry of the 30 seeded small-tree validation positions."""
import json
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.postflop_validation_batch import generate_requests
from poker_cpu.solve import SolverConfig, solve

CAPS = {
    "river_root": 2000,
    "river_facing_bet": 2000,
    "turn_full_tree": 100,
    "turn_showdown_cutoff": 500,
    "flop_showdown_cutoff": 500,
}
TARGET = 0.1


def run_adaptive(*, shark_style=False):
    started = perf_counter()
    rows = []
    for case in generate_requests(shark_style=shark_style):
        base = case["request"]
        cap = CAPS[case["profile"]]
        request = replace(base, config=replace(
            base.config, iterations=cap, target_nash_conv_chips=TARGET))
        t0 = perf_counter()
        result = solve(request)
        wall_ms = (perf_counter() - t0) * 1000
        rows.append({
            "scenario_id": case["scenario_id"], "profile": case["profile"],
            "max_iterations": cap, "target_nash_conv_chips": TARGET,
            "actual_iterations": result.iterations,
            "status": result.status, "validation_status": result.validation_status,
            "nash_conv_chips": result.nash_conv_chips,
            "target_met": result.nash_conv_chips is not None and result.nash_conv_chips <= TARGET,
            "latency_ms": result.latency_ms, "wall_latency_ms": wall_ms,
            "joint_deals": result.joint_deals,
            "actions": [
                {"kind": action.action.kind, "amount": action.action.amount,
                 "frequency": action.frequency, "ev_chips": action.ev_chips,
                 "ev_loss_chips": action.ev_loss_chips}
                for action in (result.selected_hand.actions if result.selected_hand else ())
            ],
            "convergence_trace": result.convergence_trace,
            "request": case["scenario_id"],
        })
        print(case["scenario_id"], "iters", result.iterations,
              "NashConv", (round(result.nash_conv_chips, 5)
                            if result.nash_conv_chips is not None else None),
              "ms", round(wall_ms, 1), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"]].append(row)
    summaries = {}
    for profile, group in groups.items():
        times = sorted(row["wall_latency_ms"] for row in group)
        summaries[profile] = {
            "positions": len(group),
            "target_met": sum(row["target_met"] for row in group),
            "target_pass_rate": sum(row["target_met"] for row in group) / len(group),
            "max_iterations_used": max(row["actual_iterations"] for row in group),
            "median_iterations": median(row["actual_iterations"] for row in group),
            "median_nash_conv_chips": median(row["nash_conv_chips"] for row in group),
            "worst_nash_conv_chips": max(row["nash_conv_chips"] for row in group),
            "median_latency_ms": median(times),
            "p95_latency_ms_nearest_rank": times[(95 * len(times) + 99) // 100 - 1],
        }
    record = {
        "experiment": "adaptive_budget_30_position_postflop_batch",
        "action_tree_style": "shark_2_0_inspired" if shark_style else "default_restricted",
        "date": date.today().isoformat(), "seed": 20261021,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "total_wall_seconds": perf_counter() - started,
        "methodology": f"Same 30 deterministic positions as postflop_validation_batch.json, with shark_style={shark_style}. Solver's built-in exact-NashConv target check requests <=0.1 chips. Iteration ceilings: river 2,000, full turn 100, turn/flop cutoff 500. The solver begins with a halfway checkpoint then checks every 10% of cap; it is not warm-started between budgets.",
        "profiles": summaries, "scenarios": rows,
        "limitations": [
            "Synthetic ranges contain only one or two exact combos per player.",
            "Cutoff trees omit all future betting; target applies only to those finite cutoff games.",
            "No preflop, broad ranges, range inference, or external-solver comparison.",
            "One deterministic batch; no statistical uncertainty estimate.",
            "Exact restricted-game NashConv is not full-game exploitability or abstraction error.",
        ],
    }
    return record


def main():
    record = run_adaptive()
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["profiles"], indent=2))
    print("total seconds", round(record["total_wall_seconds"], 1))
    print(path)


if __name__ == "__main__":
    main()
