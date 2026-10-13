"""Seeded NFSP learning curves measured by exact Leduc best responses."""

import argparse
import json
import platform
import random
import statistics
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import torch

from small_games.compare_cfr_nfsp import (
    information_set,
    legal,
    nash_conv,
    observation,
)
from small_games.dqn import ACTIONS
from small_games.encoder import encode_state
from small_games.leduc import LeducPoker
from small_games.train_nfsp import NFSPAgent, play_hand


def average_policy(agent):
    agent.policy_network.eval()

    @lru_cache(maxsize=None)
    def policy(state):
        with torch.no_grad():
            logits = agent.policy_network(encode_state(observation(state)))
            actions = legal(state)
            probs = torch.softmax(logits[[ACTIONS.index(a) for a in actions]], dim=0)
            return dict(zip(actions, probs.tolist()))

    return policy


def evaluate_agents(agents):
    return nash_conv(average_policy(agents[0]), average_policy(agents[1]))


def run_seed(seed, episodes, checkpoints, batch_size, artifact_root, eta, reward_scale):
    random.seed(seed)
    torch.manual_seed(seed)
    game = LeducPoker()
    agents = [
        NFSPAgent(eta=eta, reward_scale=reward_scale),
        NFSPAgent(eta=eta, reward_scale=reward_scale),
    ]
    rewards = []
    q_losses = []
    policy_losses = []
    rows = []
    started = time.perf_counter()

    for episode in range(episodes + 1):
        if episode in checkpoints:
            nc, br0, br1, ev = evaluate_agents(agents)
            rows.append(
                {
                    "episodes": episode,
                    "nash_conv": nc,
                    "exploitability_nash_conv_over_2": nc / 2,
                    "seat0_ev": ev,
                    "best_response_seat0_ev": br0,
                    "best_response_seat1_ev": br1,
                    "mean_training_reward_seat0": (
                        statistics.mean(rewards) if rewards else None
                    ),
                    "mean_q_loss": statistics.mean(q_losses) if q_losses else None,
                    "mean_policy_loss": (
                        statistics.mean(policy_losses) if policy_losses else None
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        if episode == episodes:
            break
        result = play_hand(game, agents)
        rewards.append(result[0])
        if (episode + 1) % 4 == 0:
            for agent in agents:
                q_loss = agent.train_dqn(batch_size)
                p_loss = agent.train_average_policy(batch_size)
                if q_loss is not None:
                    q_losses.append(q_loss)
                if p_loss is not None:
                    policy_losses.append(p_loss)

    output = artifact_root / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    for player, agent in enumerate(agents):
        torch.save(
            agent.policy_network.state_dict(), output / f"nfsp_player_{player}.pt"
        )
        torch.save(agent.q_network.state_dict(), output / f"nfsp_q_player_{player}.pt")
    return {
        "seed": seed,
        "snapshots": rows,
        "final_rl_buffer_sizes": [len(a.rl_buffer) for a in agents],
        "final_sl_buffer_sizes": [len(a.sl_buffer) for a in agents],
        "model_directory": str(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 1000, 3000])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/leduc_nfsp.json")
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/nfsp"))
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.batch_size < 1
        or args.reward_scale <= 0
        or not 0 <= args.eta <= 1
        or args.checkpoints != sorted(set(args.checkpoints))
        or args.checkpoints[-1] != args.episodes
        or args.checkpoints[0] != 0
    ):
        parser.error(
            "checkpoints must start at 0, end at episodes, and be sorted and unique"
        )
    torch.set_num_threads(1)
    result = {
        "experiment": "leduc_nfsp_exact_nash_conv",
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "eta": args.eta,
        "reward_scale": args.reward_scale,
        "seeds": args.seeds,
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
        row = run_seed(
            seed,
            args.episodes,
            set(args.checkpoints),
            args.batch_size,
            args.artifacts,
            args.eta,
            args.reward_scale,
        )
        result["runs"].append(row)
        print(
            f"seed {seed}: "
            + ", ".join(
                f"{x['episodes']}={x['nash_conv']:.4f}" for x in row["snapshots"]
            ),
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
