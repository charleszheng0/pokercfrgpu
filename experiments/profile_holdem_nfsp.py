"""Profile saved restricted Hold'em NFSP inference on a real legal observation."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from poker_cpu.engine import HoldemGame
from poker_cpu.nfsp import HoldemNFSPAgent, encode


def measure(fn, repetitions):
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return {
        "repetitions": repetitions,
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (repetitions - 1))],
        "mean_ms": statistics.mean(samples),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/holdem_nfsp/seed11/policy_player_0.pt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/holdem_nfsp_profile.json")
    )
    args = parser.parse_args()
    if args.repetitions < 20:
        parser.error("repetitions must be at least 20")
    torch.set_num_threads(1)
    agent = HoldemNFSPAgent()
    agent.policy.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=True)
    )
    agent.policy.eval()
    agent.mode = "average"
    game = HoldemGame(stacks=(20, 20), button=0, seed=19)
    obs = game.observe(game.current_player)
    state = encode(obs)
    for _ in range(100):
        agent.act(obs, record=False)
    with torch.no_grad():
        result = {
            "experiment": "restricted_holdem_nfsp_cpu_profile",
            "model": str(args.model),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "observation": "seed-19 first preflop decision, 20-chip stacks",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "encoding": measure(lambda: encode(obs), args.repetitions),
            "policy_forward": measure(lambda: agent.policy(state), args.repetitions),
            "complete_act": measure(
                lambda: agent.act(obs, record=False), args.repetitions
            ),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
