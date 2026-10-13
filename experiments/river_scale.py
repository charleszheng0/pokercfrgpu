"""Measure CPU river CFR cost as the explicit private-hand ranges grow."""

import json
import platform
import random
import subprocess
import time
import tracemalloc
from pathlib import Path

from poker_cpu.cards import DECK
from poker_cpu.river_solver import RiverCFR, RiverGame, nash_conv


def run():
    rng = random.Random(71)
    board = ("As", "Kh", "Qd", "7c", "2s")
    remaining = [card for card in DECK if card not in board]
    rng.shuffle(remaining)
    rows = []
    for size in (3, 6, 10):
        first = tuple(tuple(remaining[2 * i : 2 * i + 2]) for i in range(size))
        second = tuple(
            tuple(remaining[2 * size + 2 * i : 2 * size + 2 * i + 2])
            for i in range(size)
        )
        game = RiverGame(board, (first, second), pot=8, stack=10)
        tracemalloc.start()
        start = time.perf_counter()
        solver = RiverCFR(game).train(100)
        training_seconds = time.perf_counter() - start
        peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        start = time.perf_counter()
        vulnerability = nash_conv(game, solver.average, solver.average)
        response_seconds = time.perf_counter() - start
        rows.append(
            {
                "hands_per_seat": size,
                "joint_deals": len(game.deals),
                "information_sets": len(solver.regrets),
                "training_seconds_100_iterations": training_seconds,
                "exact_best_response_seconds": response_seconds,
                "nash_conv_100_iterations": vulnerability,
                "tracemalloc_peak_bytes_training": peak_bytes,
            }
        )
        print(
            f"{size} hands/seat: {training_seconds:.3f}s, {vulnerability:.3f} NashConv",
            flush=True,
        )
    return {
        "experiment": "river_range_size_cpu_scaling",
        "seed": 71,
        "board": board,
        "iterations": 100,
        "rows": rows,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }


if __name__ == "__main__":
    result = run()
    Path("experiments/river_scale.json").write_text(json.dumps(result, indent=2) + "\n")
