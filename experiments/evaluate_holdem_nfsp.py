"""Evaluate frozen Hold'em NFSP checkpoints on a larger fixed suite."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from experiments.holdem_nfsp import evaluate
from poker_cpu.nfsp import HoldemNFSPAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=900000)
    parser.add_argument("--models", type=Path, default=Path("artifacts/holdem_nfsp"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/holdem_nfsp_fixed_eval.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "experiment": "restricted_holdem_nfsp_fixed_evaluation",
        "seeds": args.seeds,
        "hands_per_seed_per_opponent": args.hands,
        "evaluation_seed": args.eval_seed,
        "opponents": ["random", "heuristic"],
        "stack": 20,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "runs": [],
    }
    for seed in args.seeds:
        path = args.models / f"seed{seed}" / "policy_player_0.pt"
        agent = HoldemNFSPAgent()
        agent.policy.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        agent.policy.eval()
        agent.mode = "average"
        row = {
            "training_seed": seed,
            "model_path": str(path),
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evaluation": [
                evaluate(agent, opponent, args.hands, args.eval_seed)
                for opponent in ("random", "heuristic")
            ],
        }
        result["runs"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"seed {seed}: "
            + ", ".join(
                f"{x['opponent']}={x['mean_chips_per_hand']:+.2f}±{x['standard_error']:.2f}"
                for x in row["evaluation"]
            ),
            flush=True,
        )
