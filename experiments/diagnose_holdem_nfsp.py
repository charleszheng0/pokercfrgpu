"""Record hand-level outcomes and wager choices for a fixed Hold'em NFSP suite."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import torch

from poker_cpu.agents import HeuristicAgent
from poker_cpu.engine import HoldemGame
from poker_cpu.nfsp import HoldemNFSPAgent, STACK


def run(agent, hands, seed):
    torch.manual_seed(seed)
    records = []
    for i in range(hands):
        seat = i % 2
        game = HoldemGame(stacks=(STACK, STACK), button=(i // 2) % 2, seed=seed + i)
        opponent = HeuristicAgent(seed + 10000 + i, samples=32)
        actions = []
        while not game.done:
            player = game.current_player
            obs = game.observe(player)
            action = (
                agent.act(obs, record=False)[1] if player == seat else opponent.act(obs)
            )
            if player == seat:
                risk = max(0, action.amount - obs["street_paid"][seat])
                actions.append(
                    {
                        "street": obs["street"],
                        "kind": action.kind,
                        "target": action.amount,
                        "risk": risk,
                        "all_in": risk > 0 and risk >= obs["stacks"][seat],
                    }
                )
            game.step(action)
        records.append(
            {
                "hand": i,
                "seat": seat,
                "button": game.button,
                "gain": game.rewards()[seat],
                "actions": actions,
                "terminal": game.history[-1][2] if game.history else None,
            }
        )
    gains = [record["gain"] for record in records]
    action_counts = Counter(
        action["kind"] for record in records for action in record["actions"]
    )
    all_in_hands = [
        record for record in records if any(a["all_in"] for a in record["actions"])
    ]
    large_losses = [record for record in records if record["gain"] <= -STACK // 2]
    return {
        "hands": hands,
        "mean_chips_per_hand": statistics.mean(gains),
        "standard_error": statistics.stdev(gains) / hands**0.5,
        "action_counts": dict(action_counts),
        "all_in_hands": len(all_in_hands),
        "mean_all_in_hand_gain": (
            statistics.mean(r["gain"] for r in all_in_hands) if all_in_hands else None
        ),
        "large_loss_hands": len(large_losses),
        "total_large_loss_chips": sum(r["gain"] for r in large_losses),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=901000)
    parser.add_argument("--models", type=Path, default=Path("artifacts/holdem_nfsp"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/holdem_nfsp_diagnostic.json"),
    )
    args = parser.parse_args()
    if args.hands < 2:
        parser.error("hands must be at least two")
    torch.set_num_threads(1)
    model = args.models / f"seed{args.seed}" / "policy_player_0.pt"
    torch.manual_seed(args.seed)
    initial = HoldemNFSPAgent()
    initial.policy.eval()
    initial.mode = "average"
    trained = HoldemNFSPAgent()
    trained.policy.load_state_dict(
        torch.load(model, map_location="cpu", weights_only=True)
    )
    trained.policy.eval()
    trained.mode = "average"
    result = {
        "experiment": "restricted_holdem_nfsp_hand_diagnostic",
        "training_seed": args.seed,
        "evaluation_seed": args.eval_seed,
        "opponent": "heuristic, 32 equity samples",
        "hands_per_policy": args.hands,
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "trained": run(trained, args.hands, args.eval_seed),
        "initialized": run(initial, args.hands, args.eval_seed),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for label in ("trained", "initialized"):
        row = result[label]
        print(
            f"{label}: {row['mean_chips_per_hand']:+.3f}±{row['standard_error']:.3f}, "
            f"all-in hands {row['all_in_hands']}, "
            f"large losses {row['large_loss_hands']}",
            flush=True,
        )
