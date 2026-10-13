"""Profile actual CPU reference operations before considering acceleration."""

import cProfile
import io
import json
import platform
import pstats
import subprocess
import time
from pathlib import Path

import torch

from poker_cpu.agents import HeuristicAgent
from poker_cpu.cards import evaluate
from poker_cpu.engine import HoldemGame
from small_games.compare_cfr_nfsp import SynchronousCFR
from small_games.encoder import encode_state
from small_games.leduc import LeducPoker
from small_games.train_nfsp import AveragePolicyNetwork


def timed(fn, repetitions):
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    return (time.perf_counter() - start) / repetitions


def run():
    torch.set_num_threads(1)
    game = HoldemGame(seed=7)
    hand = ("As", "Kd", "Qs", "Jh", "Tc", "2d", "3c")
    heuristic = HeuristicAgent(seed=7, samples=32)
    obs = game.observe(0)
    leduc_state = encode_state(LeducPoker().observe(0))
    policy = AveragePolicyNetwork().eval()
    with torch.no_grad():
        policy(leduc_state)  # Warm up dispatch.
    solver = SynchronousCFR()
    profiler = cProfile.Profile()
    cfr_started = time.perf_counter()
    profiler.enable()
    solver.train(10)
    profiler.disable()
    cfr_elapsed = time.perf_counter() - cfr_started
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(12)
    with torch.no_grad():
        inference = timed(lambda: policy(leduc_state), 5000)
    return {
        "experiment": "cpu_profile",
        "hand_evaluation_seconds": timed(lambda: evaluate(hand), 2000),
        "heuristic_decision_seconds_32_samples": timed(lambda: heuristic.act(obs), 100),
        "nfsp_policy_inference_seconds": inference,
        "leduc_cfr_step_seconds_profiled": cfr_elapsed / 10,
        "leduc_cfr_profile_top12": report.getvalue(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }


if __name__ == "__main__":
    output = Path("experiments/cpu_profile.json")
    result = run()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "leduc_cfr_profile_top12"
            },
            indent=2,
        )
    )
