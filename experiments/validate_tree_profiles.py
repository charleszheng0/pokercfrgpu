"""Run and persist the declared restricted postflop tree-profile suite."""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.solve import solve
from poker_cpu.tree_profiles import supported_tree_profiles


def result_row(profile):
    started = perf_counter()
    result = solve(profile.request)
    wall_ms = (perf_counter() - started) * 1000
    return {
        "profile_id": profile.profile_id,
        "note": profile.note,
        "status": result.status,
        "reason": result.reason,
        "validation_status": result.validation_status,
        "iterations": result.iterations,
        "joint_deals": result.joint_deals,
        "nash_conv_chips": result.nash_conv_chips,
        "latency_ms": result.latency_ms,
        "wall_latency_ms": wall_ms,
        "root_actions": [
            {"kind": action.kind, "amount": action.amount}
            for action in result.actions
        ],
        "per_hand": [
            {"hand": row.hand, "marginal_probability": row.marginal_probability,
             "actions": [
                 {"kind": item.action.kind, "amount": item.action.amount,
                  "frequency": item.frequency, "ev_chips": item.ev_chips,
                  "ev_bb": item.ev_bb, "ev_loss_chips": item.ev_loss_chips}
                 for item in row.actions
             ]}
            for row in result.per_hand
        ],
        "aggregate_actions": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev_chips": row.ev_chips,
             "ev_loss_chips": row.ev_loss_chips}
            for row in result.aggregate_actions
        ],
        "assumptions": result.assumptions,
        "convergence_trace": result.convergence_trace,
    }


def main():
    rows = [result_row(profile) for profile in supported_tree_profiles()]
    record = {
        "experiment": "declared_restricted_postflop_tree_profile_validation",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version,
        "scope": "seven declared one-deal CPU profiles; exact NashConv only for each stated finite game",
        "profiles": rows,
        "limitations": [
            "One-deal smoke profiles are not broad-range validation.",
            "Full flop profile is experimental, one CFR iteration, and not a product recommendation.",
            "Showdown-cutoff profiles omit later-street betting and require the fixed action abstraction.",
            "Preflop is unsupported; no profile result generalizes to a different request.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["profile_id"], row["status"], row["validation_status"],
              "iters", row["iterations"], "NashConv", row["nash_conv_chips"],
              "ms", round(row["wall_latency_ms"], 2))
    print(output)


if __name__ == "__main__":
    main()
