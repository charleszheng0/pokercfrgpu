"""Held-out seeded board/range validation for small shared-API postflop trees."""

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
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve


SEED = 20261021
POSITION_COUNT = 3


def make_request(rng, street, mode, iterations):
    board_count = {"flop": 3, "turn": 4, "river": 5}[street]
    cards = rng.sample(DECK, board_count + 8 if street == "river" else board_count + 4)
    board = tuple(cards[:board_count])
    hero_hands = ((tuple(cards[board_count:board_count + 2]),
                   tuple(cards[board_count + 2:board_count + 4]))
                  if street == "river" else
                  (tuple(cards[board_count:board_count + 2]),))
    opp_start = board_count + (4 if street == "river" else 2)
    opponent_hands = ((tuple(cards[opp_start:opp_start + 2]),
                       tuple(cards[opp_start + 2:opp_start + 4]))
                      if street == "river" else
                      (tuple(cards[opp_start:opp_start + 2]),))
    hero_weights = (0.65, 0.35) if len(hero_hands) == 2 else (1.0,)
    opp_weights = (0.4, 0.6) if len(opponent_hands) == 2 else (1.0,)
    return SolveRequest(
        board=board, street=street, pot_before_street=12,
        stacks_at_street_start=((10, 10) if street == "river" else
                                (4, 4) if mode == "full_tree" else (10, 10)),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=(WeightedRange(hero_hands, hero_weights),
                WeightedRange(opponent_hands, opp_weights)),
        abstraction=ActionAbstraction(),
        config=SolverConfig(
            iterations=iterations,
            max_joint_deals=4 if street == "river" else 1,
            continuation_mode=mode,
        ),
        acting_hand=hero_hands[0], range_source="seeded_held_out_profile",
    )


def solve_one(position_id, request):
    started = perf_counter()
    result = solve(request)
    wall_ms = (perf_counter() - started) * 1000
    return {
        "position_id": position_id,
        "status": result.status,
        "reason": result.reason,
        "validation_status": result.validation_status,
        "street": request.street,
        "mode": request.config.continuation_mode,
        "board": request.board,
        "hero_hand": request.acting_hand,
        "ranges": [{"hands": belief.hands, "weights": belief.weights}
                   for belief in request.ranges],
        "pot": request.pot_before_street,
        "stacks": request.stacks_at_street_start,
        "iterations": result.iterations,
        "joint_deals": result.joint_deals,
        "nash_conv_chips": result.nash_conv_chips,
        "latency_ms": result.latency_ms,
        "wall_latency_ms": wall_ms,
        "action_frequencies": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency}
            for row in result.selected_hand.actions
        ] if result.selected_hand else [],
        "action_evs": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "ev_chips": row.ev_chips, "ev_loss_chips": row.ev_loss_chips}
            for row in result.selected_hand.actions
        ] if result.selected_hand else [],
        "assumptions": result.assumptions,
    }


def main():
    rng = random.Random(SEED)
    cases = (
        ("river", "full_tree", 200),
        ("turn", "full_tree", 100),
        ("turn", "showdown_cutoff", 100),
        ("flop", "showdown_cutoff", 100),
    )
    rows = []
    for street, mode, iterations in cases:
        for position_id in range(POSITION_COUNT):
            request = make_request(rng, street, mode, iterations)
            row = solve_one(f"{street}_{mode}_{position_id}", request)
            if row["status"] != "supported_restricted_game":
                raise RuntimeError(f"{row['position_id']} failed: {row['reason']}")
            rows.append(row)
            print(row["position_id"], "NashConv", round(row["nash_conv_chips"], 6),
                  "ms", round(row["wall_latency_ms"], 2),
                  "deals", row["joint_deals"], flush=True)
    groups = {}
    for street, mode, _ in cases:
        group = [row for row in rows if row["street"] == street and row["mode"] == mode]
        groups[f"{street}_{mode}"] = {
            "positions": len(group),
            "nash_conv_chips": [row["nash_conv_chips"] for row in group],
            "nash_conv_median_chips": median(row["nash_conv_chips"] for row in group),
            "nash_conv_max_chips": max(row["nash_conv_chips"] for row in group),
            "wall_latency_ms": [row["wall_latency_ms"] for row in group],
            "wall_latency_median_ms": median(row["wall_latency_ms"] for row in group),
            "compatible_deals": [row["joint_deals"] for row in group],
        }
    record = {
        "experiment": "seeded_held_out_postflop_tree_validation",
        "date": date.today().isoformat(), "seed": SEED,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": (
            "Three distinct seeded boards/range fixtures per solver mode. River uses two weighted "
            "combos per player and exact joint deals; turn/flop profiles use one combo per player. "
            "River 200 iterations; full turn and showdown-cutoff turn/flop 100 iterations. "
            "Exact restricted-game NashConv and conditional hero action EVs are recorded."
        ),
        "groups": groups,
        "positions": rows,
        "limitations": [
            "Only three small synthetic positions per mode; no real betting-history range inference.",
            "Cutoff modes omit future betting; their NashConv is for a different game than full-tree mode.",
            "No external solver labels; this suite tests reproducible support and cross-board sensitivity, not equilibrium accuracy.",
            "Per-position latency is one CPU sample and is not a percentile estimate.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
