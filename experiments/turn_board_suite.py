"""Fixed-board exact turn CFR+ quality suite; run from repo root.

python3 -m experiments.turn_board_suite
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


RANGES = ((("Jc", "Tc"), ("8h", "8s")),
          (("9h", "9s"), ("Ac", "Ad")))
BOARDS = (("2c", "7h", "Jd", "Qs"),
          ("9c", "6d", "3s", "Th"))


def solve_board(board, iterations):
    _seven_strength.cache_clear()
    start = perf_counter()
    game = MultiStreetGame(board, RANGES, pot=8, stack=4, button=1,
                           bet_fractions=(), max_raises=0)
    root = game.initial_state()
    solver = RiverCFR(game, variant="cfr+", root_state=root)
    solver.train(iterations)
    report = solver.analyze_root(player=0)
    elapsed = (perf_counter() - start) * 1000
    per_hand = {}
    for row in report["per_hand"]:
        per_hand[" ".join(row["hand"])] = [
            {"kind": action.kind, "amount": action.amount,
             "frequency": row["frequency"][action],
             "conditional_ev_chips": row["action_evs"][action]}
            for action in report["actions"]
        ]
    return {"iterations": iterations, "joint_deals": len(game.deals),
            "latency_ms": elapsed, "nash_conv_chips": report["nash_conv"],
            "per_hand": per_hand}


def comparison(candidate, reference):
    details = {}
    for hand, actions in candidate["per_hand"].items():
        reference_actions = reference["per_hand"][hand]
        chosen = max(range(len(actions)), key=lambda i: actions[i]["frequency"])
        details[hand] = {
            "modal_action": actions[chosen]["kind"] + " " + str(actions[chosen]["amount"]),
            "modal_action_loss_under_deeper_profile_chips":
                max(row["conditional_ev_chips"] for row in reference_actions)
                - reference_actions[chosen]["conditional_ev_chips"],
            "max_abs_action_ev_delta_chips": max(
                abs(row["conditional_ev_chips"] - reference_actions[i]["conditional_ev_chips"])
                for i, row in enumerate(actions)),
        }
    return details


def main():
    results = []
    for board in BOARDS:
        candidate = solve_board(board, 100)
        reference = solve_board(board, 300)
        results.append({"board": board, "candidate": candidate,
                        "deeper_reference": reference,
                        "comparison": comparison(candidate, reference)})
    record = {
        "experiment": "fixed_turn_board_exact_reference_suite",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "config": {"ranges": RANGES, "boards_selected_before_solving": BOARDS,
                   "pot": 8, "stack": 4, "button": 1,
                   "bet_fractions": [], "max_raises": 0,
                   "variant": "synchronous CFR+",
                   "candidate_iterations": 100,
                   "deeper_reference_iterations": 300},
        "methodology": "Exact public chance and final finite-game best response for both profiles. One cold-rank-cache run per board/budget. Candidate modal action is scored under the deeper profile's root conditional action EVs; both profiles are approximate and this is not full-game action regret.",
        "results": results,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    for row in results:
        print(row["board"], "candidate NashConv",
              round(row["candidate"]["nash_conv_chips"], 6),
              "reference NashConv", round(row["deeper_reference"]["nash_conv_chips"], 6),
              "latencies ms", round(row["candidate"]["latency_ms"], 2),
              round(row["deeper_reference"]["latency_ms"], 2),
              "comparison", row["comparison"], flush=True)
    print(path)


if __name__ == "__main__":
    main()
