"""Reproducible small CPU baseline; writes the actual measured result as JSON."""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from poker_cpu.agents import HeuristicAgent, RandomAgent
from poker_cpu.engine import HoldemGame


def run(hands, seed, samples):
    gains = []
    started = time.perf_counter()
    for i in range(hands):
        seat = i % 2
        game = HoldemGame(seed=seed + i, button=i % 2)
        agents = [RandomAgent(seed + 10000 + i), RandomAgent(seed + 20000 + i)]
        agents[seat] = HeuristicAgent(seed + 30000 + i, samples=samples)
        while not game.done:
            p = game.current_player
            game.step(agents[p].act(game.observe(p)))
        gains.append(game.rewards()[seat])
    elapsed = time.perf_counter() - started
    return {
        "experiment": "heuristic_vs_random", "seed": seed, "hands": hands,
        "equity_samples_per_decision": samples,
        "mean_chips_per_hand": statistics.mean(gains),
        "standard_error": statistics.stdev(gains) / hands ** 0.5 if hands > 1 else None,
        "wins": sum(x > 0 for x in gains), "ties": sum(x == 0 for x in gains),
        "elapsed_seconds": elapsed, "python": platform.python_version(),
        "machine": platform.machine(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("experiments/baseline_seed7.json"))
    args = parser.parse_args()
    if args.hands < 2 or args.samples < 1:
        parser.error("hands must be at least 2 and samples at least 1")
    result = run(args.hands, args.seed, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
