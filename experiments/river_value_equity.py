"""Controlled value-model feature experiment on the saved river dataset."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from poker_cpu.river_solver import RiverGame
from experiments.river_value import INPUT_DIM, encode


def equity_feature(game, player, hand, weights):
    opponent = 1 - player
    own = game.strength[hand]
    wins = total = 0.0
    for other, weight in zip(game.ranges[opponent], weights[opponent]):
        if set(other) & set(hand):
            continue
        rank = game.strength[other]
        wins += weight * (1.0 if own > rank else 0.5 if own == rank else 0.0)
        total += weight
    return wins / total if total else 0.5


def run(args):
    source = json.loads(args.dataset.read_text())
    torch.set_num_threads(1)
    train_seed = source["seed"] if args.train_seed is None else args.train_seed
    torch.manual_seed(train_seed)
    train, test = [], []
    for position in source["positions"]:
        game = RiverGame(
            position["board"],
            position["ranges"],
            pot=8,
            stack=10,
            range_weights=position["range_weights"],
        )
        weights = position["range_weights"]
        for label in position["labels"]:
            player, hand = label["player"], tuple(label["hand"])
            features = encode(game, player, hand, weights)
            features.append(equity_feature(game, player, hand, weights))
            (train if position["split"] == "train" else test).append(
                (features, label["ev"])
            )
    x_train = torch.tensor([x for x, _ in train], dtype=torch.float32)
    y_train = torch.tensor([[y / 14.0] for _, y in train], dtype=torch.float32)
    x_test = torch.tensor([x for x, _ in test], dtype=torch.float32)
    y_test = torch.tensor([[y] for _, y in test], dtype=torch.float32)
    net = nn.Sequential(
        nn.Linear(INPUT_DIM + 1, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=source["lr"])
    for _ in range(source["epochs"]):
        loss = torch.mean((net(x_train) - y_train) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    start = time.perf_counter()
    with torch.no_grad():
        output = net(x_test) * 14.0
    inference = (time.perf_counter() - start) / len(test)
    error = output - y_test
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.model_output)
    return {
        "experiment": "tiny_range_river_value_equity_feature",
        "source_dataset": str(args.dataset),
        "feature_change": "append exact showdown equity against opponent range",
        "dataset_seed": source["seed"],
        "train_seed": train_seed,
        "train_positions": source["train_positions"],
        "test_positions": source["test_positions"],
        "epochs": source["epochs"],
        "lr": source["lr"],
        "model_path": str(args.model_output),
        "metrics": {
            "held_out_mae_chips": error.abs().mean().item(),
            "held_out_rmse_chips": torch.sqrt((error**2).mean()).item(),
            "network_inference_seconds_per_hand_value": inference,
            "zero_predictor_mae_chips": source["metrics"]["zero_predictor_mae_chips"],
            "shallow_vs_reference_mae_chips": source["metrics"][
                "shallow_vs_reference_mae_chips"
            ],
        },
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("experiments/river_value.json")
    )
    parser.add_argument("--train-seed", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/river_value_equity.json")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/river_value_equity.pt")
    )
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
