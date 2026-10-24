"""Held-out finite-game quality/latency check for three postflop profiles.

The deeper result is a finite-iteration reference, not a GTO oracle. Flop uses
a no-later-betting cutoff and must not be compared with full-street flop play.
"""

import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from experiments.postflop_validation_batch import _one_request, _request_dict
from poker_cpu.solve import solve

import random


SEED = 20261023
OUT = Path(__file__).with_name("scenario_milestone_validation.json")
PROFILES = (
    ("river_root", "river", "full_tree", 2, 200, 1000),
    ("turn_full_tree", "turn", "full_tree", 1, 20, 100),
    ("flop_showdown_cutoff", "flop", "showdown_cutoff", 2, 20, 100),
)


def action_key(row):
    return (row.action.kind, row.action.amount)


def run():
    rng = random.Random(SEED)
    rows = []
    for profile, street, mode, combos, shallow_budget, deeper_budget in PROFILES:
        case = _one_request(rng, profile, street, mode, shallow_budget,
                            False, combos, 0, shark_style=True)
        request = case["request"]
        results = []
        for iterations in (shallow_budget, deeper_budget):
            current = replace(request, config=replace(request.config, iterations=iterations))
            started = perf_counter()
            report = solve(current)
            results.append((report, (perf_counter() - started) * 1000))
        (shallow, shallow_ms), (deeper, deeper_ms) = results
        if shallow.status != "supported_restricted_game" or deeper.status != "supported_restricted_game":
            raise RuntimeError(f"Unsupported fixture {profile}: {shallow.reason}; {deeper.reason}")
        a = {action_key(item): item for item in shallow.selected_hand.actions}
        b = {action_key(item): item for item in deeper.selected_hand.actions}
        if a.keys() != b.keys():
            raise RuntimeError(f"Action mismatch for {profile}")
        modal = max(a, key=lambda key: a[key].frequency)
        best_deeper_ev = max(item.ev_chips for item in b.values())
        rows.append({
            "profile": profile,
            "request": _request_dict(request),
            "shallow_iterations": shallow.iterations,
            "deeper_iterations": deeper.iterations,
            "shallow_nash_conv_chips": shallow.nash_conv_chips,
            "deeper_nash_conv_chips": deeper.nash_conv_chips,
            "shallow_wall_ms": shallow_ms,
            "deeper_wall_ms": deeper_ms,
            "shallow_vs_deeper_action_ev_mae_chips": sum(
                abs(a[key].ev_chips - b[key].ev_chips) for key in a) / len(a),
            "shallow_vs_deeper_frequency_l1": sum(
                abs(a[key].frequency - b[key].frequency) for key in a),
            "shallow_modal_action_regret_on_deeper_evs_chips": best_deeper_ev - b[modal].ev_chips,
            "actions": [{"kind": key[0], "amount": key[1],
                         "shallow_frequency": a[key].frequency,
                         "deeper_frequency": b[key].frequency,
                         "shallow_ev_chips": a[key].ev_chips,
                         "deeper_ev_chips": b[key].ev_chips} for key in a],
        })
    output = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "platform": platform.platform(), "python": sys.version,
        "method": "One new seeded board/range per profile; CFR+ shallow versus deeper finite-iteration solve on identical game. Exact restricted NashConv is computed for each profile. Action regret uses deeper-profile conditional action EVs, not an equilibrium oracle.",
        "limitations": "One fixture/profile, sparse ranges. Flop showdown cutoff omits turn/river betting. No external solver parity or full-game GTO distance measured.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    result = run()
    for row in result["rows"]:
        print(row["profile"], "NashConv", row["shallow_nash_conv_chips"],
              "->", row["deeper_nash_conv_chips"], "wall ms",
              round(row["shallow_wall_ms"]), "->", round(row["deeper_wall_ms"]))
