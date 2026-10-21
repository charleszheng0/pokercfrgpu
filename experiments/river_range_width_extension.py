"""Probe exact river CFR+ traversal beyond the app's 8-combo input ceiling."""

import json
import platform
import random
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from poker_cpu.cards import DECK
from poker_cpu.engine import Action
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import (ActionAbstraction, BettingContext, SolveRequest,
                             SolverConfig, solve)

SEED = 20261024
WIDTHS = (10, 11)
ITERATIONS = 100


def requests(seed=SEED):
    rng = random.Random(seed)
    for width in WIDTHS:
        # Five board cards plus four cards per range entry. Eleven entries fit
        # in 52 cards (49 selected); twelve do not fit this disjoint fixture.
        cards = rng.sample(tuple(DECK), 5 + 4 * width)
        board = tuple(cards[:5])
        hands0 = tuple(tuple(cards[5 + 2 * i:7 + 2 * i]) for i in range(width))
        start1 = 5 + 2 * width
        hands1 = tuple(tuple(cards[start1 + 2 * i:start1 + 2 * i + 2])
                        for i in range(width))
        weights0 = tuple(rng.randint(1, 10) for _ in range(width))
        weights1 = tuple(rng.randint(1, 10) for _ in range(width))
        yield SolveRequest(
            board=board, street="river", pot_before_street=12,
            stacks_at_street_start=(16, 16),
            betting=BettingContext(button=1, first_actor=0, current_player=0),
            ranges=(WeightedRange(hands0, weights0), WeightedRange(hands1, weights1)),
            abstraction=ActionAbstraction.shark_style_postflop(),
            config=SolverConfig(iterations=ITERATIONS,
                                max_joint_deals=width * width),
            acting_hand=hands0[0], range_source="seeded_river_range_width_extension",
        )


def main():
    rows = []
    for width, request in zip(WIDTHS, requests()):
        started = perf_counter()
        result = solve(request)
        wall_ms = (perf_counter() - started) * 1000
        rows.append({
            "range_width_per_seat": width,
            "compatible_joint_deals": result.joint_deals,
            "board": request.board,
            "pot": request.pot_before_street,
            "stacks": request.stacks_at_street_start,
            "ranges": [{"hands": row.hands, "weights": row.weights}
                       for row in request.ranges],
            "iterations": result.iterations,
            "status": result.status,
            "reason": result.reason,
            "nash_conv_chips": result.nash_conv_chips,
            "latency_ms": result.latency_ms,
            "wall_latency_ms": wall_ms,
            "selected_hand_actions": [{
                "kind": row.action.kind, "amount": row.action.amount,
                "frequency": row.frequency, "ev_chips": row.ev_chips,
            } for row in (result.selected_hand.actions if result.selected_hand else ())],
        })
        print(width, "combos/seat", result.status,
              "deals", result.joint_deals, "iterations", result.iterations,
              "NashConv", (round(result.nash_conv_chips, 6)
                           if result.nash_conv_chips is not None else None),
              "latency_ms", round(wall_ms, 1), flush=True)
    record = {
        "experiment": "exact_river_range_width_extension",
        "date": date.today().isoformat(), "seed": SEED,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": (
            "One seeded root river game each at 10x10 and 11x11 exact weighted combos, "
            "Shark-inspired action abstraction, exact deal traversal, 100 CFR+ iterations. "
            "All board and private cards are sampled without replacement. This probes solver "
            "core range width beyond the Scenario Lab's current eight-entry-per-seat limit; "
            "one fixture per width does not estimate general performance."
        ),
        "iterations": ITERATIONS,
        "scenarios": rows,
        "summary": {
            str(row["range_width_per_seat"]): {
                "compatible_joint_deals": row["compatible_joint_deals"],
                "latency_ms": row["wall_latency_ms"],
                "nash_conv_chips": row["nash_conv_chips"],
            } for row in rows
        },
        "limitations": [
            "One sampled root position per width; not a reliable latency distribution.",
            "100 iterations are a scale probe, not a convergence claim.",
            "Only river, exact sparse combos, and a restricted betting tree are covered.",
            "Eleven combos per seat is nowhere near 1,326 combinations per player.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("median latency ms", median(row["wall_latency_ms"] for row in rows))
    print(output)


if __name__ == "__main__":
    main()
