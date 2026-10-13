"""Extend river training positions while preserving the exact held-out boards."""

import argparse
import json
import random
from pathlib import Path

from experiments.river_value import sample_position, solve_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("experiments/river_value.json")
    )
    parser.add_argument("--extra", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1053)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/river_value_extended_dataset.json"),
    )
    args = parser.parse_args()
    if args.extra < 1:
        parser.error("extra must be positive")
    data = json.loads(args.source.read_text())
    rng = random.Random(args.seed)
    existing_boards = {tuple(row["board"]) for row in data["positions"]}
    added = []
    for i in range(args.extra):
        while True:
            board, ranges, weights = sample_position(rng)
            if tuple(board) not in existing_boards:
                existing_boards.add(tuple(board))
                break
        game, labels, nc, _ = solve_position(
            board, ranges, weights, data["reference_iterations"]
        )
        added.append(
            {
                "split": "train",
                "board": board,
                "ranges": ranges,
                "range_weights": weights,
                "labels": [
                    {"player": p, "hand": hand, "ev": value}
                    for p in (0, 1)
                    for hand, value in labels[p].items()
                ],
                "reference_nash_conv": nc,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"Additional training positions {i + 1}/{args.extra}", flush=True)
    data["positions"] = (
        [row for row in data["positions"] if row["split"] == "train"]
        + added
        + [row for row in data["positions"] if row["split"] == "test"]
    )
    data["train_positions"] += args.extra
    data["experiment"] = "tiny_range_river_value_extended_dataset"
    data["extension_source"] = str(args.source)
    data["extension_seed"] = args.seed
    data["extension_extra_positions"] = args.extra
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"Saved {data['train_positions']} training and {data['test_positions']} fixed test positions"
    )
