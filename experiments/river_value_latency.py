"""End-to-end CPU latency of a saved river value model versus local solving."""

import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from experiments.river_value import INPUT_DIM, encode
from experiments.river_value_equity import equity_feature
from poker_cpu.river_solver import RiverCFR, RiverGame


def timed(fn, repeat):
    started = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - started) / repeat


def main():
    torch.set_num_threads(1)
    dataset = json.loads(
        Path("experiments/river_value_extended_dataset.json").read_text()
    )
    position = next(row for row in dataset["positions"] if row["split"] == "test")
    weights = position["range_weights"]
    model_path = Path("artifacts/river_value_equity_180_seed53.pt")
    net = nn.Sequential(
        nn.Linear(INPUT_DIM + 1, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    net.eval()

    def network_position():
        game = RiverGame(
            position["board"],
            position["ranges"],
            pot=8,
            stack=10,
            range_weights=weights,
        )
        features = []
        for player in (0, 1):
            for hand in game.ranges[player]:
                row = encode(game, player, hand, weights)
                row.append(equity_feature(game, player, hand, weights))
                features.append(row)
        with torch.no_grad():
            return net(torch.tensor(features, dtype=torch.float32)) * 14.0

    def shallow_position():
        game = RiverGame(
            position["board"],
            position["ranges"],
            pot=8,
            stack=10,
            range_weights=weights,
        )
        return RiverCFR(game).train(20).root_hand_values()

    network_position()
    shallow_position()
    result = {
        "experiment": "river_value_end_to_end_latency",
        "model": str(model_path),
        "dataset": "experiments/river_value_extended_dataset.json",
        "test_board": position["board"],
        "network_including_features_seconds_per_position": timed(network_position, 50),
        "shallow_20_iteration_seconds_per_position": timed(shallow_position, 20),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }
    Path("experiments/river_value_latency.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
