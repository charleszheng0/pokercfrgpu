"""Reproducible restricted Hold'em NFSP self-play and fixed-opponent tests."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from poker_cpu.agents import HeuristicAgent, RandomAgent
from poker_cpu.engine import HoldemGame
from poker_cpu.nfsp import HoldemNFSPAgent, STACK, play_self_play_hand


def evaluate(agent, opponent, hands, seed):
    # Pair policy-action randomness across checkpoints on the same dealt hands.
    torch.manual_seed(seed)
    gains = []
    wins = ties = 0
    for i in range(hands):
        seat = i % 2
        game = HoldemGame(stacks=(STACK, STACK), button=(i // 2) % 2, seed=seed + i)
        bot = (
            RandomAgent(seed + 10000 + i)
            if opponent == "random"
            else HeuristicAgent(seed + 10000 + i, samples=32)
        )
        while not game.done:
            p = game.current_player
            obs = game.observe(p)
            action = agent.act(obs, record=False)[1] if p == seat else bot.act(obs)
            game.step(action)
        gain = game.rewards()[seat]
        gains.append(gain)
        wins += gain > 0
        ties += gain == 0
    return {
        "opponent": opponent,
        "hands": hands,
        "mean_chips_per_hand": statistics.mean(gains),
        "standard_error": statistics.stdev(gains) / hands**0.5 if hands > 1 else None,
        "wins": wins,
        "ties": ties,
    }


def run_seed(seed, hands, eval_hands, output_root):
    random.seed(seed)
    torch.manual_seed(seed)
    agents = [HoldemNFSPAgent(), HoldemNFSPAgent()]
    rewards = []
    q_losses = []
    sl_losses = []
    started = time.perf_counter()
    for i in range(hands):
        result = play_self_play_hand(agents, seed + i, button=i % 2)
        rewards.append(result[0])
        if (i + 1) % 4 == 0:
            for agent in agents:
                q, sl = agent.update()
                if q is not None:
                    q_losses.append(q)
                if sl is not None:
                    sl_losses.append(sl)
        if (i + 1) % 1000 == 0:
            print(f"seed {seed}: {i + 1}/{hands} self-play hands", flush=True)
    training_seconds = time.perf_counter() - started
    directory = output_root / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    for player, agent in enumerate(agents):
        torch.save(agent.policy.state_dict(), directory / f"policy_player_{player}.pt")
        torch.save(agent.q.state_dict(), directory / f"q_player_{player}.pt")
    agents[0].mode = "average"
    evaluation = [
        evaluate(agents[0], opponent, eval_hands, seed + 100000)
        for opponent in ("random", "heuristic")
    ]
    return {
        "seed": seed,
        "training_hands": hands,
        "training_seconds": training_seconds,
        "mean_self_play_reward_seat0": statistics.mean(rewards),
        "mean_q_loss": statistics.mean(q_losses) if q_losses else None,
        "mean_policy_loss": statistics.mean(sl_losses) if sl_losses else None,
        "rl_buffer_sizes": [len(agent.rl) for agent in agents],
        "sl_buffer_sizes": [len(agent.sl.data) for agent in agents],
        "evaluation": evaluation,
        "model_directory": str(directory),
        "model_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.pt"))
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--hands", type=int, default=10000)
    parser.add_argument("--eval-hands", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/holdem_nfsp.json")
    )
    parser.add_argument("--models", type=Path, default=Path("artifacts/holdem_nfsp"))
    args = parser.parse_args()
    if args.hands < 1 or args.eval_hands < 2:
        parser.error("hands must be positive; eval-hands must be at least two")
    torch.set_num_threads(1)
    result = {
        "experiment": "restricted_holdem_nfsp",
        "stack": STACK,
        "blinds": [1, 2],
        "seeds": args.seeds,
        "hands_per_training_seed": args.hands,
        "evaluation_hands_per_opponent": args.eval_hands,
        "algorithm": "DQN best response plus reservoir-supervised average policy, eta=0.3",
        "observation_dim": 525,
        "action_slots": 11,
        "network_hidden_units": [128, 128],
        "eta": 0.3,
        "epsilon": 0.1,
        "learning_rate": 0.001,
        "dqn_reward_scale": 20,
        "target_update_interval": 200,
        "rl_capacity": 100000,
        "sl_capacity": 100000,
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
        row = run_seed(seed, args.hands, args.eval_hands, args.models)
        result["runs"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"seed {seed}: "
            + ", ".join(
                f"{x['opponent']}={x['mean_chips_per_hand']:+.2f}"
                for x in row["evaluation"]
            ),
            flush=True,
        )
