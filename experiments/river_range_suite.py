"""Fixed-seed multi-board test of CFR+ with weighted, blocker-free ranges."""

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

from poker_cpu.cards import DECK
from poker_cpu.river_solver import RiverCFR, RiverGame, nash_conv


def make_positions(seed, count):
    rng = random.Random(seed)
    positions = []
    for index in range(count):
        cards = rng.sample(DECK, 17)
        board = tuple(cards[:5])
        range0 = (tuple(cards[5:7]), tuple(cards[7:9]), tuple(cards[9:11]))
        range1 = (tuple(cards[11:13]), tuple(cards[13:15]), tuple(cards[15:17]))
        weights0 = tuple(rng.randint(1, 3) for _ in range(3))
        weights1 = tuple(rng.randint(1, 3) for _ in range(3))
        positions.append({
            "id": index,
            "board": board,
            "ranges": (range0, range1),
            "range_weights": (weights0, weights1),
            "pot": rng.choice((6, 8, 12)),
            "stack": rng.choice((6, 10, 14)),
        })
    return positions


def run(seed=729, count=5, iterations=500):
    rows = []
    for position in make_positions(seed, count):
        game = RiverGame(
            position["board"], position["ranges"], pot=position["pot"],
            stack=position["stack"], range_weights=position["range_weights"],
        )
        solver = RiverCFR(game, variant="cfr+")
        start = time.perf_counter()
        solver.train(iterations)
        train_seconds = time.perf_counter() - start
        exploitability_start = time.perf_counter()
        report = solver.analyze_root()
        report_seconds = time.perf_counter() - exploitability_start
        rows.append({
            "position_id": position["id"],
            "board": position["board"],
            "ranges": position["ranges"],
            "range_weights": position["range_weights"],
            "pot": position["pot"],
            "effective_stack": position["stack"],
            "compatible_deals": len(game.deals),
            "iterations": iterations,
            "nash_conv_chips": report["nash_conv"],
            "seat0_ev_chips": game.expected_value(solver.average, solver.average),
            "train_seconds": train_seconds,
            "exact_report_seconds": report_seconds,
        })
    return {
        "experiment": "fixed_seed_weighted_river_range_suite_cfr_plus",
        "seed": seed,
        "positions": rows,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=729)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("experiments/river_range_suite.json"))
    args = parser.parse_args()
    if args.count < 1 or args.iterations < 1:
        parser.error("count and iterations must be positive")
    result = run(args.seed, args.count, args.iterations)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["positions"]:
        print(json.dumps({key: row[key] for key in (
            "position_id", "compatible_deals", "iterations", "nash_conv_chips",
            "train_seconds", "exact_report_seconds"
        )}))
