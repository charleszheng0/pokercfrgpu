"""Reproduce the explicit-range Solver Lab and chance-transition measurements."""

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from poker_cpu.cards import DECK
from poker_cpu.public_chance import public_transition
from poker_cpu.river_solver import WeightedRange
from trainer.server import analyze_river_position


BASE_REQUEST = {
    "board": "As Kh Qd 7c 2s",
    "hero_hand": "Jc Tc",
    "pot": 20,
    "stacks": [100, 100],
    "ranges": [
        {"hands": [["Jc", "Tc"], ["9c", "8c"], ["4c", "4d"]], "weights": [1, 1, 1]},
        {"hands": [["Ac", "Ad"], ["Jd", "Jh"], ["8h", "8s"]], "weights": [0.5, 1, 0.5]},
    ],
}


def main():
    runs = []
    for name, iterations, first_actor, actions in (
        ("first_river_action", 500, 0, []),
        ("after_hero_check_opponent_bet_10", 500, 0,
         [{"kind": "check", "amount": 0}, {"kind": "bet", "amount": 10}]),
        ("after_hero_check_opponent_bet_10_deeper", 2000, 0,
         [{"kind": "check", "amount": 0}, {"kind": "bet", "amount": 10}]),
        ("after_opponent_check", 500, 1, [{"kind": "check", "amount": 0}]),
    ):
        request = {**BASE_REQUEST, "iterations": iterations,
                   "first_actor": first_actor, "river_actions": actions}
        runs.append({"name": name, "request": request,
                     "response": analyze_river_position(request)})
    ranges = tuple(
        WeightedRange(tuple(tuple(hand) for hand in item["hands"]), tuple(item["weights"]))
        for item in BASE_REQUEST["ranges"]
    )
    board = ("As", "Kh", "Qd")
    started = time.perf_counter()
    outcomes = [public_transition(board, (card,), ranges) for card in DECK if card not in board]
    chance_ms = (time.perf_counter() - started) * 1000
    record = {
        "experiment": "solver_lab_matrix_post_action_and_public_chance",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty; current Solver Lab and range-inference work not committed",
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "methodology": (
            "Direct analyze_river_position calls on four explicitly supplied river states, "
            "500 or 2000 CFR+ iterations with exact finite-game NashConv at half/final "
            "checkpoints. Post-action requests deliberately reuse the root manual ranges "
            "without action conditioning; they test root-state mechanics, not advice quality. "
            "The chance check enumerates all turn cards from one weighted three-hand-per-seat "
            "flop state and sums exact card-removal probabilities. No random sampling."
        ),
        "runs": runs,
        "chance_check": {
            "board": board, "outcomes": len(outcomes),
            "probability_sum": sum(row.probability for row in outcomes),
            "elapsed_ms": chance_ms,
        },
    }
    path = Path(__file__).with_name("solver_lab_milestone_v2.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)
    for row in runs:
        result = row["response"]
        print(row["name"], result["iterations"],
              round(result["nash_conv"], 6), round(result["latency_ms"], 2))
    print("turn chance sum", record["chance_check"]["probability_sum"],
          "elapsed ms", round(chance_ms, 3))


if __name__ == "__main__":
    main()
