"""Test whether replacing only NFSP faced-bet decisions changes chip return."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from experiments.holdem_nfsp import evaluate
from poker_cpu.agents import HeuristicAgent
from poker_cpu.nfsp import HoldemNFSPAgent


class TimedPolicy:
    def __init__(self, nfsp, guard=None):
        self.nfsp = nfsp
        self.guard = guard
        self.counts = Counter()
        self.elapsed = 0.0

    def act(self, obs, record=False):
        start = time.perf_counter()
        if self.guard is not None and obs["to_call"] > 0:
            action = self.guard.act(obs)
            self.counts["guard"] += 1
        else:
            action = self.nfsp.act(obs, record=False)[1]
            self.counts["nfsp"] += 1
        self.elapsed += time.perf_counter() - start
        self.counts[action.kind] += 1
        return None, action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=902000)
    parser.add_argument("--models", type=Path, default=Path("artifacts/holdem_nfsp"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/nfsp_faced_bet_ablation.json"),
    )
    args = parser.parse_args()
    if args.hands < 2:
        parser.error("hands must be at least two")
    torch.set_num_threads(1)
    path = args.models / f"seed{args.seed}" / "policy_player_0.pt"
    nfsp = HoldemNFSPAgent()
    nfsp.policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    nfsp.policy.eval()
    nfsp.mode = "average"
    results = {}
    for name, guard in (
        ("saved_nfsp", None),
        (
            "heuristic_when_facing_bet",
            HeuristicAgent(args.eval_seed + 90000, samples=32),
        ),
    ):
        policy = TimedPolicy(nfsp, guard)
        row = evaluate(policy, "heuristic", args.hands, args.eval_seed)
        row["policy_action_counts"] = dict(policy.counts)
        row["policy_decision_seconds"] = policy.elapsed
        row["mean_policy_decision_ms"] = (
            1000 * policy.elapsed / (policy.counts["guard"] + policy.counts["nfsp"])
        )
        results[name] = row
        print(
            f"{name}: {row['mean_chips_per_hand']:+.3f}±{row['standard_error']:.3f} "
            f"chips/hand, {row['mean_policy_decision_ms']:.2f} ms/decision",
            flush=True,
        )
    result = {
        "experiment": "restricted_holdem_nfsp_faced_bet_ablation",
        "training_seed": args.seed,
        "evaluation_seed": args.eval_seed,
        "hands_per_condition": args.hands,
        "opponent": "heuristic, 32 equity samples",
        "model": str(path),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "results": results,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
