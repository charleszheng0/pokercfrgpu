"""Validate 20 postflop positions reached after prior abstract actions."""
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

from experiments.postflop_validation_batch import generate_history_requests, _request_dict
from poker_cpu.solve import SolverConfig, solve

CAPS = {
    "river_facing_raise": 2000,
    "turn_full_tree_facing_bet": 100,
    "turn_cutoff_facing_bet": 500,
    "flop_cutoff_facing_bet": 500,
    "flop_cutoff_after_check": 500,
}


def _run(request, *, cap, target=None):
    config = replace(request.config, iterations=cap, target_nash_conv_chips=target)
    result = solve(replace(request, config=config))
    return {
        "status": result.status, "reason": result.reason,
        "iterations": result.iterations, "nash_conv_chips": result.nash_conv_chips,
        "target_met": target is not None and result.nash_conv_chips is not None
                     and result.nash_conv_chips <= target,
        "latency_ms": result.latency_ms,
        "joint_deals": result.joint_deals,
        "actions": [
            {"kind": item.action.kind, "amount": item.action.amount,
             "frequency": item.frequency, "ev_chips": item.ev_chips,
             "ev_loss_chips": item.ev_loss_chips}
            for item in (result.selected_hand.actions if result.selected_hand else ())
        ],
    }


def main():
    started = perf_counter()
    rows = []
    for case in generate_history_requests():
        request = case["request"]
        fixed_started = perf_counter()
        fixed = _run(request, cap=request.config.iterations)
        fixed["wall_latency_ms"] = (perf_counter() - fixed_started) * 1000
        adaptive_started = perf_counter()
        adaptive = _run(request, cap=CAPS[case["profile"]], target=0.1)
        adaptive["wall_latency_ms"] = (perf_counter() - adaptive_started) * 1000
        rows.append({
            "scenario_id": case["scenario_id"], "profile": case["profile"],
            "request": _request_dict(request), "fixed_budget": fixed,
            "adaptive_target_0_1": adaptive,
        })
        print(case["scenario_id"], "fixed", fixed["status"],
              round(fixed["nash_conv_chips"] or 0, 5),
              "target", adaptive["target_met"], adaptive["iterations"],
              round(adaptive["nash_conv_chips"] or 0, 5), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"]].append(row)
    summaries = {}
    for profile, group in groups.items():
        latencies = sorted(row["adaptive_target_0_1"]["wall_latency_ms"] for row in group)
        residuals = [row["adaptive_target_0_1"]["nash_conv_chips"] for row in group]
        summaries[profile] = {
            "positions": len(group),
            "supported": sum(row["adaptive_target_0_1"]["status"] == "supported_restricted_game" for row in group),
            "target_passed": sum(row["adaptive_target_0_1"]["target_met"] for row in group),
            "target_pass_rate": sum(row["adaptive_target_0_1"]["target_met"] for row in group) / len(group),
            "nash_conv_median": median(residuals), "nash_conv_max": max(residuals),
            "latency_median_ms": median(latencies),
            "latency_p95_ms_nearest_rank": latencies[(95 * len(latencies) + 99) // 100 - 1],
        }
    record = {
        "experiment": "20_position_postflop_history_validation",
        "date": date.today().isoformat(), "seed": 20261022,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "total_wall_seconds": perf_counter() - started,
        "methodology": "Twenty deterministic synthetic states across river facing a raise, full turn facing a bet, turn/flop cutoff facing a bet, and flop cutoff after an opponent check. Four positions each. River/full-turn use the Shark-inspired tree; cutoff retains its fixed tree. Each state is solved once at fixed initial budget and once with exact <=0.1-chip restricted NashConv target and per-profile iteration ceiling.",
        "target_caps": CAPS, "profiles": summaries, "scenarios": rows,
        "limitations": [
            "At most two exact combos per seat; ranges are synthetic.",
            "Flop/turn cutoffs omit all future betting.",
            "No preflop or action-likelihood range inference.",
            "NashConv is exact only inside each finite tree and says nothing about action abstraction error.",
            "Four fixtures per history profile are a coverage smoke, not a production pass-rate estimate.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print("total seconds", round(record["total_wall_seconds"], 1), output)


if __name__ == "__main__":
    main()
