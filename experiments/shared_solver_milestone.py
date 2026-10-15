"""Reproduce the unified river and turn CPU solver integration benchmark."""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

from poker_cpu.one_street_solver import OneStreetGame
from poker_cpu.river_solver import RiverCFR, RiverGame, WeightedRange
from poker_cpu.solve import (ActionAbstraction, BettingContext, SolveRequest,
                             SolverConfig, solve)


RANGES = (
    WeightedRange((("Jc", "Tc"), ("8h", "8s")), (1, 1)),
    WeightedRange((("9h", "9s"), ("Ac", "Ad")), (1, 1)),
)


def rows(actions):
    return [{"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev_chips": row.ev_chips}
            for row in actions]


def main():
    board = ("As", "Kh", "Qd", "7c", "2s")
    old_game = RiverGame(board, tuple(belief.hands for belief in RANGES),
                         pot=8, stack=10)
    old = RiverCFR(old_game, variant="cfr+").train(50).analyze_root(
        private_hand=("Jc", "Tc"))
    river = solve(SolveRequest(
        board=board, street="river", pot_before_street=8,
        stacks_at_street_start=(10, 10),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=RANGES, config=SolverConfig(iterations=50),
        acting_hand=("Jc", "Tc"),
    ))
    old_hand = next(row for row in old["per_hand"] if row["hand"] == ("Jc", "Tc"))
    river_max_frequency_difference = max(
        abs(row.frequency - old_hand["frequency"][row.action])
        for row in river.selected_hand.actions
    )
    river_max_ev_difference = max(
        abs(row.ev_chips - old_hand["action_evs"][row.action])
        for row in river.selected_hand.actions
    )
    turn_ranges = (
        WeightedRange((("8h", "8s"),), (1,)),
        WeightedRange((("9h", "9s"),), (1,)),
    )
    turn_board = board[:4]
    turn = solve(SolveRequest(
        board=turn_board, street="turn", pot_before_street=8,
        stacks_at_street_start=(4, 4),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=turn_ranges, abstraction=ActionAbstraction((), 1),
        config=SolverConfig(iterations=10), acting_hand=("8h", "8s"),
    ))
    one_street_game = OneStreetGame(
        turn_board, tuple(belief.hands for belief in turn_ranges), pot=8, stack=4)
    one_street = RiverCFR(one_street_game, variant="cfr+").train(10).analyze_root(
        private_hand=("8h", "8s"))
    turn_variants = []
    for variant in ("cfr", "cfr+", "dcfr"):
        result = solve(SolveRequest(
            board=turn_board, street="turn", pot_before_street=8,
            stacks_at_street_start=(4, 4),
            betting=BettingContext(button=1, first_actor=0, current_player=0),
            ranges=turn_ranges, abstraction=ActionAbstraction((), 1),
            config=SolverConfig(variant=variant, iterations=100),
            acting_hand=("8h", "8s"),
        ))
        turn_variants.append({"variant": variant,
                              "iterations": result.iterations,
                              "nash_conv_chips": result.nash_conv_chips,
                              "latency_ms": result.latency_ms})
    flop_ranges = (
        WeightedRange((("Jc", "Tc"),), (1,)),
        WeightedRange((("9h", "9s"),), (1,)),
    )
    flop = solve(SolveRequest(
        board=board[:3], street="flop", pot_before_street=8,
        stacks_at_street_start=(1, 1),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=flop_ranges, abstraction=ActionAbstraction((), 0),
        config=SolverConfig(iterations=1, allow_experimental_flop=True),
        acting_hand=("Jc", "Tc"),
    ))
    record = {
        "experiment": "shared_postflop_solver_milestone",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty; shared solver milestone not committed",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version,
        "methodology": "One deterministic river reference comparison at 50 CFR+ iterations, one one-deal turn comparison at 10 iterations, a three-variant turn comparison at 100 iterations, and a one-iteration one-deal flop stress check. Exact chance; no sampling. Timings include solver setup, search, and exact restricted-game best responses. One run per condition, no uncertainty estimate.",
        "river": {
            "board": board, "ranges": [list(belief.hands) for belief in RANGES],
            "iterations": river.iterations, "joint_deals": river.joint_deals,
            "shared_latency_ms": river.latency_ms,
            "reference_nash_conv_chips": old["nash_conv"],
            "shared_nash_conv_chips": river.nash_conv_chips,
            "max_hero_frequency_difference": river_max_frequency_difference,
            "max_hero_conditional_ev_difference_chips": river_max_ev_difference,
            "per_hand_count": len(river.per_hand),
            "hero_actions": rows(river.selected_hand.actions),
        },
        "turn": {
            "board": turn_board,
            "ranges": [list(belief.hands) for belief in turn_ranges],
            "stack": 4, "pot": 8, "iterations": turn.iterations,
            "joint_deals": turn.joint_deals,
            "shared_latency_ms": turn.latency_ms,
            "shared_nash_conv_chips": turn.nash_conv_chips,
            "one_street_nash_conv_chips": one_street["nash_conv"],
            "hero_actions": rows(turn.selected_hand.actions),
            "one_street_hero_action_evs": [
                {"kind": action.kind, "amount": action.amount, "ev_chips": ev}
                for action, ev in one_street["per_hand"][0]["action_evs"].items()
            ],
            "continuation_difference": "Shared solver includes river betting; one-street reference does not",
        },
        "turn_variants": turn_variants,
        "flop_experimental": {
            "board": board[:3], "joint_deals": flop.joint_deals,
            "effective_stack": 1, "iterations": flop.iterations,
            "nash_conv_chips": flop.nash_conv_chips,
            "latency_ms": flop.latency_ms,
            "app_status": "disabled: one iteration is not a useful convergence result",
        },
    }
    path = Path(__file__).with_name("shared_solver_milestone.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)
    print("river reference differences", river_max_frequency_difference,
          river_max_ev_difference, "NashConv", river.nash_conv_chips)
    print("turn NashConv", turn.nash_conv_chips,
          "latency_ms", round(turn.latency_ms, 2))
    print("turn variants", [(row["variant"], round(row["nash_conv_chips"], 6))
                             for row in turn_variants])
    print("flop one-iteration NashConv", flop.nash_conv_chips,
          "latency_ms", round(flop.latency_ms, 2))


if __name__ == "__main__":
    main()
