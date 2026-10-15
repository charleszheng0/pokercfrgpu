"""Compare exact and chance-sampled CPU CFR on fixed tiny postflop games.

Run from the repository root: python3 -m experiments.multistreet_scale
This is a research comparison, not an app quality certification.
"""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.multi_street import MultiStreetGame, _seven_strength
from poker_cpu.river_solver import RiverCFR


def run(street, ranges, stack, variant, iterations, seed=None):
    board = ("As", "Kh", "Qd") if street == "flop" else ("As", "Kh", "Qd", "7c")
    _seven_strength.cache_clear()
    started = perf_counter()
    game = MultiStreetGame(board, ranges, pot=8, stack=stack, button=1,
                           bet_fractions=(), max_raises=0)
    root = game.initial_state()
    solver = RiverCFR(game, variant=variant, root_state=root,
                      chance_sampling=seed is not None,
                      sampling_seed=seed or 0)
    search_started = perf_counter()
    solver.train(iterations)
    search_ms = (perf_counter() - search_started) * 1000
    report = solver.analyze_root(player=0, private_hand=ranges[0][0])
    total_ms = (perf_counter() - started) * 1000
    return {
        "street": street, "board": board, "ranges": ranges,
        "pot": 8, "stack": stack, "bet_fractions": [], "max_raises": 0,
        "variant": variant, "chance_sampling": seed is not None,
        "seed": seed, "iterations": iterations, "joint_deals": len(game.deals),
        "search_ms": search_ms, "total_ms": total_ms,
        "nash_conv_chips": report["nash_conv"],
        "rank_cache": {"hits": _seven_strength.cache_info().hits,
                       "misses": _seven_strength.cache_info().misses},
        "per_hand": [
            {"hand": row["hand"],
             "actions": [{"kind": action.kind, "amount": action.amount,
                          "frequency": row["frequency"][action],
                          "conditional_ev_chips": row["action_evs"][action]}
                         for action in report["actions"]]}
            for row in report["per_hand"]
        ],
    }


def max_ev_delta(candidate, reference):
    reference_rows = {tuple(row["hand"]): row for row in reference["per_hand"]}
    return max(
        abs(action["conditional_ev_chips"] - reference_action["conditional_ev_chips"])
        for row in candidate["per_hand"]
        for action, reference_action in zip(row["actions"],
                                            reference_rows[tuple(row["hand"])]["actions"])
    )


def main():
    flop_ranges = ((("Jc", "Tc"),), (("9h", "9s"),))
    turn_ranges = ((("Jc", "Tc"), ("8h", "8s")),
                   (("9h", "9s"), ("Ac", "Ad")))
    flop_reference = run("flop", flop_ranges, 1, "cfr+", 20)
    turn_reference = run("turn", turn_ranges, 4, "cfr+", 300)
    turn_budget = run("turn", turn_ranges, 4, "cfr+", 100)
    sampled = [run(street, ranges, stack, "cfr", 1000, seed)
               for street, ranges, stack in (("flop", flop_ranges, 1),
                                             ("turn", turn_ranges, 4))
               for seed in (17, 23, 41)]
    for row in sampled:
        row["max_action_ev_delta_to_deeper_exact_reference_chips"] = max_ev_delta(
            row, flop_reference if row["street"] == "flop" else turn_reference)
    turn_budget["max_action_ev_delta_to_deeper_exact_reference_chips"] = max_ev_delta(
        turn_budget, turn_reference)
    record = {
        "experiment": "multistreet_exact_collapse_and_chance_sampling",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": "Sequential single-process cold rank-cache runs; each run clears only the global board/hand rank cache. Per-game state caches are new. Total time includes setup, search and exact best-response analysis. Each sampled seed uses the true public-card distribution with an exact all-in payoff collapse. EV deltas compare each candidate's average continuation policy with a deeper exact CFR+ profile, so they are descriptive and not true equilibrium errors.",
        "references": {"flop_exact_20": flop_reference,
                       "turn_exact_300": turn_reference,
                       "turn_exact_100": turn_budget},
        "sampled": sampled,
        "limitations": ["one flop hand pair and one four-deal turn position",
                        "no full-game exploitability", "sampled CFR uses vanilla CFR while exact uses CFR+",
                        "one timing per condition", "flop exact 20 is itself approximate"],
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    for label, row in record["references"].items():
        print(label, row["iterations"], round(row["nash_conv_chips"], 6),
              round(row["total_ms"], 2))
    for row in sampled:
        print(row["street"], "sampled", row["seed"],
              round(row["nash_conv_chips"], 6), round(row["total_ms"], 2),
              round(row["max_action_ev_delta_to_deeper_exact_reference_chips"], 6))
    print(path)


if __name__ == "__main__":
    main()
