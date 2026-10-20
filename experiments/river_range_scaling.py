"""Measure exact-combo river CFR cost as explicit ranges grow.

The benchmark deliberately does not alter solver limits or regret math. Sizes
through the current 64-compatible-deal API cap use ``solve()``; larger sizes
are candidate-only direct ``RiverGame``/``RiverCFR`` runs with identical rules.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import date
from itertools import combinations
from pathlib import Path
from random import Random
from time import perf_counter

from poker_cpu.cards import DECK
from poker_cpu.ranges import canonical_hand
from poker_cpu.river_solver import RiverCFR, RiverGame, RiverState, WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve


BOARD = ("As", "Kh", "Qd", "7c", "2s")
RANGE_SIZES = (1, 2, 4, 6, 8, 10, 12)
GENERATION_SEED = 20261020
ITERATIONS = 20
POT = 20
STACKS = (40, 40)
BIG_BLIND = 2
MAX_DEALS = 64


def concrete_ranges(size: int, seed: int = GENERATION_SEED):
    """Return deterministic unique combos with board blockers removed."""
    if type(size) is not int or size < 1:
        raise ValueError("range size must be a positive integer")
    available = [canonical_hand(hand) for hand in combinations(DECK, 2)
                 if not set(hand) & set(BOARD)]
    # Separate shuffled slices produce different ranges while preserving
    # common prefixes across the size sweep. Retry only when the chosen ranges
    # have no compatible deals at all; card-removal collisions otherwise stay
    # in the benchmark and are counted rather than silently removed.
    split = len(available) // 2
    if size > split:
        raise ValueError("requested range size exceeds available concrete combos")
    for attempt in range(100):
        shuffled = available.copy()
        Random(seed + attempt).shuffle(shuffled)
        ranges = (shuffled[:size], shuffled[split:split + size])
        if compatible_joint_deals(ranges):
            return ranges
    raise RuntimeError("could not sample ranges with a compatible joint deal")


def compatible_joint_deals(ranges):
    return sum(not (set(hand0) & set(hand1))
               for hand0 in ranges[0] for hand1 in ranges[1])


def action_json(action):
    return {"kind": action.kind, "amount": action.amount}


def _shared_request(ranges):
    beliefs = tuple(WeightedRange(tuple(hands), (1.0,) * len(hands))
                    for hands in ranges)
    return SolveRequest(
        board=BOARD, street="river", pot_before_street=POT,
        stacks_at_street_start=STACKS,
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=beliefs,
        acting_hand=next(hand for hand in ranges[0]
                         if any(not (set(hand) & set(other)) for other in ranges[1])),
        big_blind=BIG_BLIND,
        abstraction=ActionAbstraction(),
        config=SolverConfig(variant="cfr+", iterations=ITERATIONS,
                            max_joint_deals=MAX_DEALS),
    )


def _format_result(result):
    return {
        "status": result.status,
        "reason": result.reason,
        "validation_status": result.validation_status,
        "iterations": result.iterations,
        "compatible_joint_deals": result.joint_deals,
        "nash_conv_chips": result.nash_conv_chips,
        "latency_ms": result.latency_ms,
        "search_latency_ms": result.search_latency_ms,
        "root_actions": [action_json(action) for action in result.actions],
        "aggregate_action_frequencies": [
            {**action_json(row.action), "frequency": row.frequency}
            for row in result.aggregate_actions
        ],
        "aggregate_conditional_action_evs_chips": [
            {**action_json(row.action), "ev_chips": row.ev_chips}
            for row in result.aggregate_actions
        ],
        "selected_hand": (" ".join(result.selected_hand.hand)
                          if result.selected_hand else None),
        "selected_hand_frequencies": ([
            {**action_json(row.action), "frequency": row.frequency}
            for row in result.selected_hand.actions
        ] if result.selected_hand else []),
        "selected_hand_action_evs_chips": ([
            {**action_json(row.action), "ev_chips": row.ev_chips}
            for row in result.selected_hand.actions
        ] if result.selected_hand else []),
    }


def _run_direct_candidate(ranges, iterations):
    beliefs = tuple(WeightedRange(tuple(hands), (1.0,) * len(hands))
                    for hands in ranges)
    game = RiverGame(BOARD, tuple(belief.hands for belief in beliefs),
                     pot=POT, stack=min(STACKS), big_blind=BIG_BLIND,
                     range_weights=tuple(belief.weights for belief in beliefs))
    solver = RiverCFR(game, variant="cfr+")
    started = perf_counter()
    solver.train(iterations)
    selected_hand = next(hand for hand in ranges[0]
                         if any(not (set(hand) & set(other)) for other in ranges[1]))
    report = solver.analyze_root(private_hand=selected_hand)
    elapsed_ms = (perf_counter() - started) * 1000
    return {
        "status": "candidate_direct_solver",
        "reason": "Direct RiverGame/RiverCFR bypasses the shared API deal cap; not production-supported.",
        "validation_status": report["validation_status"],
        "iterations": report["iterations"],
        "compatible_joint_deals": len(game.deals),
        "nash_conv_chips": report["nash_conv"],
        "latency_ms": elapsed_ms,
        "search_latency_ms": None,
        "root_actions": [action_json(action) for action in report["actions"]],
        "aggregate_action_frequencies": [
            {**action_json(action), "frequency": report["action_frequencies"][action]}
            for action in report["actions"]
        ],
        "aggregate_conditional_action_evs_chips": [
            {**action_json(action), "ev_chips": report["conditional_action_evs"][action]}
            for action in report["actions"]
        ],
        "selected_hand": " ".join(selected_hand),
        "selected_hand_frequencies": [
            {**action_json(action), "frequency": report["per_hand"][0]["frequency"][action]}
            for action in report["actions"]
        ],
        "selected_hand_action_evs_chips": [
            {**action_json(action), "ev_chips": report["per_hand"][0]["action_evs"][action]}
            for action in report["actions"]
        ],
    }


def run_condition(size: int, *, seed: int = GENERATION_SEED,
                  iterations: int = ITERATIONS):
    """Run one deterministic condition; used by CLI and harness smoke test."""
    ranges = concrete_ranges(size, seed)
    deal_count = compatible_joint_deals(ranges)
    record = {
        "range_combos_per_player": size,
        "compatible_joint_deals_expected": deal_count,
        "ranges": [[" ".join(hand) for hand in seat] for seat in ranges],
        "mode": "shared_api" if deal_count <= MAX_DEALS else "direct_candidate",
    }
    try:
        if record["mode"] == "shared_api":
            request = _shared_request(ranges)
            if iterations != ITERATIONS:
                request = SolveRequest(**{**request.__dict__, "config": SolverConfig(
                    variant="cfr+", iterations=iterations,
                    max_joint_deals=MAX_DEALS)})
            output = _format_result(solve(request))
        else:
            # Candidate direct runs use the same configured CFR+ iteration
            # count without relaxing the public SolveRequest/API limits.
            output = _run_direct_candidate(ranges, iterations)
        return {**record, **output, "error": None}
    except Exception as exc:  # Record a failed size and continue the batch.
        return {**record, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run_batch(sizes=RANGE_SIZES, *, seed=GENERATION_SEED, iterations=ITERATIONS):
    return [run_condition(size, seed=seed, iterations=iterations) for size in sizes]


def main():
    rows = run_batch()
    record = {
        "experiment": "river_exact_combo_range_scaling",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "seed": GENERATION_SEED,
        "board": BOARD,
        "pot_chips": POT,
        "effective_stacks_chips": STACKS,
        "big_blind_chips": BIG_BLIND,
        "variant": "cfr+",
        "iterations": ITERATIONS,
        "action_abstraction": {"bet_fractions": (0.5, 1.0), "max_raises": 1},
        "shared_api_deal_cap": MAX_DEALS,
        "methodology": (
            "Each player receives equally weighted exact concrete combos. Range prefixes are deterministic from the saved seed. "
            "Compatible joint deals exclude card collisions. Shared API results are used at or below 64 deals; larger rows call the same RiverGame/RiverCFR implementation directly and are candidate-only. "
            "Each row uses one exact CFR+ run and exact finite-game best-response/NashConv analysis; timings are single observations, not distributions."
        ),
        "results": rows,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)
    for row in rows:
        print(row["range_combos_per_player"], "combos/seat",
              row["compatible_joint_deals_expected"], "deals",
              row.get("iterations", 0), "iterations",
              row.get("nash_conv_chips"), "chips NashConv",
              row.get("latency_ms"), "ms", row.get("error"))


if __name__ == "__main__":
    main()
