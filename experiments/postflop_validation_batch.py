"""Deterministic 30-position quality/latency batch for constrained postflop trees.

This is a finite-game validation suite, not a full-range blueprint generator.
"""
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from poker_cpu.cards import DECK
from poker_cpu.engine import Action
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve

SEED = 20261021
CASES_PER_PROFILE = 6
PROFILES = (
    ("river_root", "river", "full_tree", 500, False, 2),
    ("river_facing_bet", "river", "full_tree", 500, True, 2),
    ("turn_full_tree", "turn", "full_tree", 100, False, 1),
    ("turn_showdown_cutoff", "turn", "showdown_cutoff", 100, False, 2),
    ("flop_showdown_cutoff", "flop", "showdown_cutoff", 100, False, 2),
)


def _one_request(rng, profile_id, street, mode, iterations, facing_bet,
                 combos_per_seat, index, shark_style=False):
    board_count = {"flop": 3, "turn": 4, "river": 5}[street]
    deal = rng.sample(tuple(DECK), board_count + 8)
    board = tuple(deal[:board_count])
    seat0 = (tuple(deal[board_count:board_count + 2]),
             tuple(deal[board_count + 2:board_count + 4]))
    seat1 = (tuple(deal[board_count + 4:board_count + 6]),
             tuple(deal[board_count + 6:board_count + 8]))
    seat0 = seat0[:combos_per_seat]
    seat1 = seat1[:combos_per_seat]
    weights0 = tuple(rng.randint(1, 9) for _ in seat0)
    weights1 = tuple(rng.randint(1, 9) for _ in seat1)
    if profile_id == "turn_full_tree":
        # Previous measurement: broad stack/action trees made one solve take
        # 86 seconds. Keep this exact full-turn profile to the tested 4-chip cap.
        pot, effective = rng.choice((8, 12)), 4
    elif profile_id.endswith("showdown_cutoff"):
        pot, effective = rng.choice((8, 12, 16)), rng.choice((4, 8))
    else:
        pot = rng.choice((8, 12, 16, 20))
        effective = rng.choice((8, 12, 16))
    stacks = (effective, effective + rng.choice((0, 0, 4)))
    facing_bet_size = max(2, min(effective, int(pot / 3))) if shark_style else max(
        2, min(effective, int(pot * 0.5)))
    actions = (Action("bet", facing_bet_size),) if facing_bet else ()
    actor = 1 if facing_bet else 0
    request = SolveRequest(
        board=board, street=street, pot_before_street=pot,
        stacks_at_street_start=stacks,
        betting=BettingContext(button=1, first_actor=0, current_player=actor,
                               street_actions=actions),
        ranges=(WeightedRange(seat0, weights0), WeightedRange(seat1, weights1)),
        abstraction=(ActionAbstraction.shark_style_postflop()
                     if shark_style and mode == "full_tree" else ActionAbstraction()),
        config=SolverConfig(iterations=iterations, max_joint_deals=4,
                            continuation_mode=mode),
        acting_hand=(seat1[0] if facing_bet else seat0[0]),
        range_source="seeded_batch_assumption",
    )
    return {"scenario_id": f"{profile_id}_{index:02d}", "profile": profile_id,
            "request": request}


def generate_requests(seed=SEED, *, shark_style=False):
    rng = random.Random(seed)
    cases = []
    for profile_id, street, mode, iterations, facing_bet, combos in PROFILES:
        for index in range(CASES_PER_PROFILE):
            cases.append(_one_request(rng, profile_id, street, mode, iterations,
                                      facing_bet, combos, index, shark_style))
    return cases


def _request_dict(request):
    return {
        "board": request.board, "street": request.street,
        "pot": request.pot_before_street, "stacks": request.stacks_at_street_start,
        "first_actor": request.betting.first_actor,
        "current_player": request.betting.current_player,
        "history": [(a.kind, a.amount) for a in request.betting.street_actions],
        "ranges": [{"hands": r.hands, "weights": r.weights} for r in request.ranges],
        "iterations": request.config.iterations,
        "variant": request.config.variant,
        "continuation_mode": request.config.continuation_mode,
        "abstraction": {
            "bet_fractions": request.abstraction.bet_fractions,
            "raise_fractions": request.abstraction.raise_fractions,
            "bet_fractions_by_street": request.abstraction.bet_fractions_by_street,
            "raise_fractions_by_street": request.abstraction.raise_fractions_by_street,
            "max_raises": request.abstraction.max_raises,
            "include_min_size": request.abstraction.include_min_size,
        },
    }


def run_batch(seed=SEED, *, shark_style=False):
    started = perf_counter()
    output_rows = []
    for case in generate_requests(seed, shark_style=shark_style):
        request = case["request"]
        t0 = perf_counter()
        result = solve(request)
        wall_ms = (perf_counter() - t0) * 1000
        actions = result.selected_hand.actions if result.selected_hand else ()
        output_rows.append({
            "scenario_id": case["scenario_id"], "profile": case["profile"],
            "request": _request_dict(request), "status": result.status,
            "reason": result.reason, "validation_status": result.validation_status,
            "iterations": result.iterations, "joint_deals": result.joint_deals,
            "nash_conv_chips": result.nash_conv_chips,
            "latency_ms": result.latency_ms, "wall_latency_ms": wall_ms,
            "selected_hand_actions": [
                {"kind": row.action.kind, "amount": row.action.amount,
                 "frequency": row.frequency, "ev_chips": row.ev_chips,
                 "ev_loss_chips": row.ev_loss_chips}
                for row in actions
            ],
            "assumptions": result.assumptions,
            "residual_gate_0_1_chips_passed": (
                result.status == "supported_restricted_game"
                and result.nash_conv_chips is not None
                and result.nash_conv_chips <= 0.1
            ),
        })
        print(case["scenario_id"], result.status,
              "NashConv", (round(result.nash_conv_chips, 5)
                            if result.nash_conv_chips is not None else None),
              "ms", round(wall_ms, 1), flush=True)
    groups = {}
    by_profile = defaultdict(list)
    for row in output_rows:
        by_profile[row["profile"]].append(row)
    for profile, rows in by_profile.items():
        latencies = sorted(row["wall_latency_ms"] for row in rows)
        residuals = [row["nash_conv_chips"] for row in rows
                     if row["nash_conv_chips"] is not None]
        groups[profile] = {
            "positions": len(rows),
            "supported": sum(row["status"] == "supported_restricted_game" for row in rows),
            "gate_pass_count": sum(row["residual_gate_0_1_chips_passed"] for row in rows),
            "gate_pass_rate": sum(row["residual_gate_0_1_chips_passed"] for row in rows) / len(rows),
            "nash_conv_median_chips": median(residuals) if residuals else None,
            "nash_conv_max_chips": max(residuals) if residuals else None,
            "latency_median_ms": median(latencies),
            "latency_p95_ms_nearest_rank": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)],
        }
    return {
        "experiment": "30_position_constrained_postflop_validation_batch",
        "action_tree_style": "shark_2_0_inspired" if shark_style else "default_restricted",
        "date": date.today().isoformat(), "seed": seed,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "total_wall_seconds": perf_counter() - started,
        "methodology": (
            f"Thirty deterministic synthetic scenarios; six per profile. Randomized board, weighted exact combos per seat, pot and effective stack. River uses two combos per seat and full restricted betting at 500 CFR+ iterations. Full turn uses one combo per seat and 4-chip effective stacks at 100 iterations, because wider trees measured 86 seconds per solve. Turn/flop showdown-cutoff use two combos per seat and 100 iterations. Six river facing-bet scenarios start after a legal {'one-third-pot' if shark_style else 'half-pot'} bet. {'River/full-turn use the Shark 2.0-inspired per-street menu; cutoff trees retain their fixed half-pot/pot/all-in menu.' if shark_style else 'All profiles use the default restricted menu.'} NashConv is exact within each supplied-range finite tree. 0.1-chip pass rate is a screening metric, not GTO validation."
        ),
        "profiles": groups, "scenarios": output_rows,
        "limitations": [
            "Only two private combos per seat; no broad 1,326-combo ranges.",
            "Turn/flop cutoff profiles omit future betting and cannot be compared as full-game advice.",
            "No preflop solving, no action-conditioned range inference, no external solver comparison.",
            "Only one deterministic batch; no uncertainty estimate or real-user state coverage.",
            "The 0.1-chip NashConv screen applies only to the exact restricted game and does not measure abstraction error.",
        ],
    }


def main():
    record = run_batch()
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["profiles"], indent=2))
    print("total seconds", round(record["total_wall_seconds"], 2))
    print(path)


if __name__ == "__main__":
    main()
