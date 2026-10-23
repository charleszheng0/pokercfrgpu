"""Reproduce Scenario Lab's public-tree preview timings and menu sizes.

Run: python3 -m experiments.profile_tree_preview
This benchmarks preview generation only, not CFR convergence or solve latency.
"""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from trainer.tree_preview import preview_betting_tree


def main():
    rows = []
    for tree in ("default", "shark_2_0_inspired"):
        for spot in ("opening", "facing_bet", "facing_raise"):
            payload = {
                "board": "As Kh Qd 7c 2s", "pot": 20,
                "stacks": [100, 100], "betting_tree": tree, "spot": spot,
            }
            times = []
            for _ in range(50):
                started = perf_counter()
                result = preview_betting_tree(payload)
                times.append((perf_counter() - started) * 1000)
            ordered = sorted(times)
            rows.append({
                "tree": tree, "spot": spot,
                "median_ms": median(times), "p95_ms": ordered[47],
                "public_decision_states": result["current_street_public_decisions"],
                "public_action_edges": result["current_street_public_action_edges"],
                "legal_actions": result["legal_actions"],
                "prior_bet_targets": result["prior_bet_targets"],
                "prior_raise_targets": result["prior_raise_targets"],
            })
    record = {
        "experiment": "scenario_lab_public_tree_preview",
        "date": date.today().isoformat(),
        "revision_before_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version,
        "method": "Fifty sequential in-process calls per river tree/spot after module import; p95 is sorted sample 48/50. No HTTP, JavaScript, CFR, or hand evaluation included.",
        "inputs": {"board": "As Kh Qd 7c 2s", "pot": 20,
                   "stacks": [100, 100]},
        "rows": rows,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["tree"], row["spot"], row["public_decision_states"],
              round(row["median_ms"], 3), round(row["p95_ms"], 3))
    print(output)


if __name__ == "__main__":
    main()
