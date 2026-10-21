"""Measure exact river solve behavior as supplied ranges grow beyond two combos."""
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
from poker_cpu.solve import (ActionAbstraction, BettingContext, SolveRequest,
                             SolverConfig, solve)

SEED = 20261023
WIDTHS = (2, 3, 4, 6, 8)
POSITIONS_PER_WIDTH = {2: 4, 3: 4, 4: 4, 6: 2, 8: 2}
TARGET = 0.1


def generate_requests(seed=SEED):
    rng = random.Random(seed)
    cases = []
    for width in WIDTHS:
        for index in range(POSITIONS_PER_WIDTH[width]):
            cards = rng.sample(tuple(DECK), 5 + width * 4)
            board = tuple(cards[:5])
            hands0 = tuple(tuple(cards[5 + 2 * i:7 + 2 * i]) for i in range(width))
            start1 = 5 + 2 * width
            hands1 = tuple(tuple(cards[start1 + 2 * i:start1 + 2 * i + 2])
                            for i in range(width))
            weights0 = tuple(rng.randint(1, 10) for _ in range(width))
            weights1 = tuple(rng.randint(1, 10) for _ in range(width))
            pot = rng.choice((8, 12, 16))
            stack = rng.choice((10, 16, 24))
            facing = index % 2 == 1
            bet = max(2, min(stack, int(pot / 3)))
            request = SolveRequest(
                board=board, street="river", pot_before_street=pot,
                stacks_at_street_start=(stack, stack),
                betting=BettingContext(button=1, first_actor=0,
                                       current_player=1 if facing else 0,
                                       street_actions=(Action("bet", bet),) if facing else ()),
                ranges=(WeightedRange(hands0, weights0), WeightedRange(hands1, weights1)),
                abstraction=ActionAbstraction.shark_style_postflop(),
                config=SolverConfig(iterations=(500 if width >= 6 else 2000), max_joint_deals=64,
                                    target_nash_conv_chips=TARGET),
                acting_hand=hands1[0] if facing else hands0[0],
                range_source="seeded_weighted_range_width_validation",
            )
            cases.append({
                "scenario_id": f"river_{width}x{width}_{index:02d}",
                "range_width": width, "facing_bet": facing,
                "iteration_cap": 500 if width >= 6 else 2000,
                "request": request,
            })
    return cases


def _serialize_request(request):
    return {
        "board": request.board, "pot": request.pot_before_street,
        "stacks": request.stacks_at_street_start,
        "current_player": request.betting.current_player,
        "street_actions": [(row.kind, row.amount) for row in request.betting.street_actions],
        "ranges": [{"hands": row.hands, "weights": row.weights} for row in request.ranges],
        "action_tree": "Shark 2.0-inspired postflop menu",
        "iterations_cap": request.config.iterations,
        "target_nash_conv_chips": request.config.target_nash_conv_chips,
        "compatible_joint_deals": None,
    }


def main():
    started = perf_counter()
    rows = []
    for case in generate_requests():
        request = case["request"]
        t0 = perf_counter()
        result = solve(request)
        wall_ms = (perf_counter() - t0) * 1000
        rows.append({
            "scenario_id": case["scenario_id"], "range_width_per_seat": case["range_width"],
            "facing_bet": case["facing_bet"],
            "request": _serialize_request(request),
            "status": result.status, "reason": result.reason,
            "iterations": result.iterations, "joint_deals": result.joint_deals,
            "nash_conv_chips": result.nash_conv_chips,
            "target_met": result.nash_conv_chips is not None and result.nash_conv_chips <= TARGET,
            "latency_ms": result.latency_ms, "wall_latency_ms": wall_ms,
            "action_rows": [
                {"kind": row.action.kind, "amount": row.action.amount,
                 "frequency": row.frequency, "ev_chips": row.ev_chips,
                 "ev_loss_chips": row.ev_loss_chips}
                for row in (result.selected_hand.actions if result.selected_hand else ())
            ],
        })
        rows[-1]["request"]["compatible_joint_deals"] = result.joint_deals
        print(rows[-1]["scenario_id"], result.status,
              "deals", result.joint_deals,
              "iters", result.iterations,
              "NashConv", round(result.nash_conv_chips or 0, 5),
              "ms", round(wall_ms, 1), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["range_width_per_seat"]].append(row)
    summaries = {}
    for width, group in groups.items():
        latencies = sorted(row["wall_latency_ms"] for row in group)
        residuals = [row["nash_conv_chips"] for row in group]
        summaries[str(width)] = {
            "positions": len(group), "target_passed": sum(row["target_met"] for row in group),
            "target_pass_rate": sum(row["target_met"] for row in group) / len(group),
            "deals_min_max": [min(row["joint_deals"] for row in group),
                              max(row["joint_deals"] for row in group)],
            "nash_conv_median_max": [median(residuals), max(residuals)],
            "latency_median_ms": median(latencies),
            "latency_p95_ms_nearest_rank": latencies[(95 * len(latencies) + 99) // 100 - 1],
        }
    record = {
        "experiment": "river_range_width_shark_tree_validation",
        "date": date.today().isoformat(), "seed": SEED,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": "Sixteen seeded river scenarios at 2/3/4/6/8 exact combos per seat, balancing root and facing-bet states. Weighted ranges and board are blocker-disjoint. Shark 2.0-inspired menu, CFR+ exact traversal, 2,000-iteration cap through four combos and 500-iteration cap for six/eight combos, exact restricted-game NashConv target 0.1 chips. Four fixtures at widths 2/3/4; two at widths 6/8.",
        "target": TARGET,
        "profiles": summaries, "scenarios": rows,
        "limitations": [
            "Maximum eight combos per player; this does not solve the full 1,326-combo matrix range.",
            "Only two positions at widths six/eight; no statistical reliability estimate for wide ranges.",
            "All games are river-only and abstracted; no preflop/turn/flop continuation.",
            "No opponent range inference or external solver comparison.",
        ],
        "total_wall_seconds": perf_counter() - started,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print("total seconds", round(record["total_wall_seconds"], 1), path)


if __name__ == "__main__":
    main()
