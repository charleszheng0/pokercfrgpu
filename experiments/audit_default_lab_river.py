"""Reproduce the default Scenario Lab result and verify its jam EV identity."""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.cards import evaluate
from poker_cpu.engine import Action
from poker_cpu.multi_street import MultiStreetGame
from poker_cpu.ranges import canonical_hand, preset_range
from poker_cpu.river_solver import RiverCFR

BOARD = ("As", "Kh", "Qd", "7c", "2s")
HERO = canonical_hand(("Jc", "Tc"))
POT = 20
STACK = 100
CHECKPOINTS = (500, 2000, 5000)


def main():
    hero_range = preset_range("river_demo_hero")
    opponent_range = preset_range("river_demo_opponent")
    game = MultiStreetGame(
        BOARD, (hero_range.hands, opponent_range.hands), pot=POT, stack=STACK,
        button=1, big_blind=2,
        range_weights=(hero_range.weights, opponent_range.weights),
    )
    root = game.initial_state(0)
    solver = RiverCFR(game, variant="cfr+", root_player=0, root_state=root)
    jam = Action("bet", STACK)
    call = Action("call")
    jam_response = game.advance(root, jam)
    hero_score = evaluate(HERO + BOARD)
    opponent_scores = {
        " ".join(hand): evaluate(hand + BOARD)
        for hand in opponent_range.hands
    }
    assert all(hero_score > score for score in opponent_scores.values())

    started = perf_counter()
    runs = []
    for budget in CHECKPOINTS:
        while solver.iterations < budget:
            solver.step()
        report = solver.analyze_root(private_hand=HERO)
        hand = next(row for row in report["per_hand"] if row["hand"] == HERO)
        compatible = [(hand0, hand1, weight) for hand0, hand1, weight in game.deals
                      if hand0 == HERO]
        mass = sum(weight for _, _, weight in compatible)
        call_probability = sum(
            weight * solver.average(jam_response, (hand0, hand1)).get(call, 0.0)
            for hand0, hand1, weight in compatible
        ) / mass
        jam_decision_ev = hand["action_evs"][jam] + POT / 2
        payoff_identity = POT + STACK * call_probability
        assert abs(jam_decision_ev - payoff_identity) < 1e-9
        actions = [{
            "kind": action.kind, "amount": action.amount,
            "frequency": hand["frequency"][action],
            "pot_centered_ev_chips": hand["action_evs"][action],
            "decision_ev_chips": hand["action_evs"][action] + POT / 2,
        } for action in report["actions"]]
        runs.append({
            "iterations": solver.iterations,
            "elapsed_wall_ms": (perf_counter() - started) * 1000,
            "nash_conv_chips": report["nash_conv"],
            "actions": actions,
            "jam_call_probability_given_hero": call_probability,
            "jam_ev_from_independent_payoff_identity": payoff_identity,
        })
        print(budget, "NashConv", round(report["nash_conv"], 6),
              "jam freq", round(hand["frequency"][jam], 6),
              "jam EV", round(jam_decision_ev, 6),
              "conditional call probability", round(call_probability, 6), flush=True)

    record = {
        "experiment": "default_scenario_lab_river_ev_strategy_audit",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "position": {
            "board": BOARD, "hero_hand": HERO, "pot": POT, "stack": STACK,
            "hero_range": {"hands": hero_range.hands, "weights": hero_range.weights},
            "opponent_range": {"hands": opponent_range.hands,
                               "weights": opponent_range.weights},
            "compatible_joint_deals": len(game.deals),
            "hero_hand_rank": hero_score,
            "opponent_hand_ranks": opponent_scores,
            "abstraction": "default half-pot/pot/minimum/all-in, one raise per street",
        },
        "runs": runs,
        "payoff_check": (
            "Hero's Broadway straight beats every entered opponent combo. "
            "When hero jams 100 into a 20-chip pot, opponent fold yields +20 decision EV "
            "and opponent call yields +120; therefore jam EV = 20 + 100 * conditional "
            "opponent call probability. This identity is asserted at each checkpoint."
        ),
        "limitations": [
            "The two three-combo ranges are illustrative assumptions, not inferred from play.",
            "NashConv is exact only for the supplied nine-deal restricted betting tree.",
            "This audit verifies a payoff identity and convergence trend, not full-game GTO.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
