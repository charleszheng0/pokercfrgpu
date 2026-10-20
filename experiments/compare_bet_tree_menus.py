"""Compare baseline and Shark-inspired action menus on one fixed river game.

Run with ``python -m experiments.compare_bet_tree_menus``. This is a small
branching/convergence probe, not an external solver comparison or strength
claim. The saved JSON captures the exact request and both actual solve outputs.
"""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from poker_cpu.multi_street import _seven_strength
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import (
    ActionAbstraction,
    BettingContext,
    SolveRequest,
    SolverConfig,
    solve,
)


def one_run(abstraction, iterations):
    _seven_strength.cache_clear()
    request = SolveRequest(
        board=("As", "Kh", "Qd", "7c", "2s"),
        street="river",
        pot_before_street=20,
        stacks_at_street_start=(40, 40),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=(
            WeightedRange((("Jc", "Tc"), ("8h", "8s")), (1.0, 1.0)),
            WeightedRange((("9h", "9s"), ("Ac", "Ad")), (1.0, 1.0)),
        ),
        abstraction=abstraction,
        config=SolverConfig(iterations=iterations, max_joint_deals=16),
        acting_hand=("Jc", "Tc"),
        range_source="fixed_equal_weight_fixture",
    )
    started = perf_counter()
    result = solve(request)
    wall_ms = (perf_counter() - started) * 1000
    return {
        "status": result.status,
        "iterations": result.iterations,
        "compatible_joint_deals": result.joint_deals,
        "action_count": len(result.actions),
        "actions": [
            {"kind": action.kind, "amount": action.amount}
            for action in result.actions
        ],
        "restricted_nash_conv_chips": result.nash_conv_chips,
        "solver_latency_ms": result.latency_ms,
        "wall_latency_ms": wall_ms,
        "selected_hand": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev_chips": row.ev_chips}
            for row in result.selected_hand.actions
        ] if result.selected_hand else None,
        "request": {
            "board": request.board,
            "pot_chips": request.pot_before_street,
            "stacks": request.stacks_at_street_start,
            "ranges": [
                {"hands": range_.hands, "weights": range_.weights}
                for range_ in request.ranges
            ],
            "iterations": iterations,
        },
    }


def main():
    iterations = 100
    variants = {
        "baseline": ActionAbstraction(),
        "shark_inspired": ActionAbstraction.shark_style_postflop(),
    }
    repeats = {name: [one_run(abstraction, iterations) for _ in range(3)]
               for name, abstraction in variants.items()}
    record = {
        "experiment": "same_game_bet_tree_menu_probe",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "methodology": (
            "One fixed river request, equal-weight explicit two-combo ranges, "
            "exact joint deals, 100 CFR+ iterations, exact finite-game NashConv; "
            "three sequential runs per menu with rank cache cleared before each. "
            "This compares two abstractions, so NashConv values are not expected "
            "to certify the same game or imply full no-limit quality."
        ),
        "summary": {
            name: {
                "root_actions": runs[-1]["actions"],
                "root_action_count": runs[-1]["action_count"],
                "median_wall_latency_ms": median(row["wall_latency_ms"] for row in runs),
                "nash_conv_chips_by_run": [row["restricted_nash_conv_chips"] for row in runs],
                "selected_hand_action_frequencies": runs[-1]["selected_hand"],
            }
            for name, runs in repeats.items()
        },
        "runs": repeats,
        "limitations": [
            "single river position and tiny ranges",
            "different abstractions define different finite games",
            "100 iterations is a latency/tree probe, not a convergence gate",
            "no external solver comparison",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for name, summary in record["summary"].items():
        print(name, summary["root_action_count"],
              round(summary["median_wall_latency_ms"], 2),
              [round(value, 6) for value in summary["nash_conv_chips_by_run"]])
    print(output)


if __name__ == "__main__":
    main()
