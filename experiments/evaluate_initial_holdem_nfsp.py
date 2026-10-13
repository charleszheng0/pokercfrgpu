"""Fixed-opponent reference for untrained Hold'em NFSP policies."""

import argparse
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/holdem_nfsp_untrained_eval.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "experiment": "restricted_holdem_nfsp_untrained_evaluation",
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
        torch.manual_seed(seed)
        agent = HoldemNFSPAgent()
        agent.policy.eval()
        agent.mode = "average"
        row = {
            "initialization_seed": seed,
            "evaluation": [
                evaluate(agent, opponent, args.hands, args.eval_seed)
                for opponent in ("random", "heuristic")
            ],
        }
        result["runs"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"untrained seed {seed}: "
            + ", ".join(
                f"{x['opponent']}={x['mean_chips_per_hand']:+.2f}±{x['standard_error']:.2f}"
                for x in row["evaluation"]
            ),
            flush=True,
        )
