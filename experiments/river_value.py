"""Train a small river value model from deeper CPU CFR labels.

This is an exploratory continuation-value prototype: each label is a
hand-conditional root EV under a 200-iteration tiny-range river strategy.
It is not DeepStack's counterfactual value network or a safe re-solver.
"""

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from poker_cpu.cards import DECK
from poker_cpu.river_solver import RiverCFR, RiverGame, nash_conv

INPUT_DIM = 52 + 52 + 2 * 3 * 53 + 1 + 2


def encode(game, player, hand, weights):
    features = [float(card in game.board) for card in DECK]
    features.extend(float(card in hand) for card in DECK)
    for p in (0, 1):
        total = sum(weights[p])
        for slot, candidate in enumerate(game.ranges[p]):
            features.extend(float(card in candidate) for card in DECK)
            features.append(weights[p][slot] / total)
    features.extend((float(player), game.pot / 20, game.stack / 20))
    assert len(features) == INPUT_DIM
    return features


def sample_position(rng):
    cards = rng.sample(DECK, 17)
    board = cards[:5]
    ranges = (
        tuple(tuple(cards[5 + 2 * i : 7 + 2 * i]) for i in range(3)),
        tuple(tuple(cards[11 + 2 * i : 13 + 2 * i]) for i in range(3)),
    )
    weights = (
        tuple(rng.expovariate(1) for _ in range(3)),
        tuple(rng.expovariate(1) for _ in range(3)),
    )
    return board, ranges, weights


def solve_position(board, ranges, weights, iterations):
    game = RiverGame(board, ranges, pot=8, stack=10, range_weights=weights)
    started = time.perf_counter()
    solver = RiverCFR(game).train(iterations)
    elapsed = time.perf_counter() - started
    return (
        game,
        solver.root_hand_values(),
        nash_conv(game, solver.average, solver.average),
        elapsed,
    )


def examples_for(game, weights, values):
    for player in (0, 1):
        for hand, value in values[player].items():
            yield encode(game, player, hand, weights), value


def model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def run(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train_examples, test_examples = [], []
    raw = []
    reference_times, shallow_times, ref_nc, shallow_nc = [], [], [], []
    shallow_abs_errors = []
    positions = args.train_positions + args.test_positions
    for index in range(positions):
        board, ranges, weights = sample_position(rng)
        game, labels, nc, elapsed = solve_position(
            board, ranges, weights, args.reference_iterations
        )
        reference_times.append(elapsed)
        ref_nc.append(nc)
        split = "train" if index < args.train_positions else "test"
        examples = list(examples_for(game, weights, labels))
        (train_examples if split == "train" else test_examples).extend(examples)
        if split == "test":
            _, shallow, shallow_vulnerability, shallow_elapsed = solve_position(
                board, ranges, weights, args.shallow_iterations
            )
            shallow_times.append(shallow_elapsed)
            shallow_nc.append(shallow_vulnerability)
            for player in (0, 1):
                for hand in labels[player]:
                    shallow_abs_errors.append(
                        abs(labels[player][hand] - shallow[player][hand])
                    )
        raw.append(
            {
                "split": split,
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
        if (index + 1) % 10 == 0:
            print(f"Solved {index + 1}/{positions} public positions", flush=True)

    scale = 14.0  # Pot/2 + effective stack upper bound in this configuration.
    x_train = torch.tensor([x for x, _ in train_examples], dtype=torch.float32)
    y_train = torch.tensor(
        [[y / scale] for _, y in train_examples], dtype=torch.float32
    )
    x_test = torch.tensor([x for x, _ in test_examples], dtype=torch.float32)
    y_test = torch.tensor([[y] for _, y in test_examples], dtype=torch.float32)
    net = model()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    losses = []
    for epoch in range(args.epochs):
        predicted = net(x_train)
        loss = torch.mean((predicted - y_train) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in (0, args.epochs // 2, args.epochs - 1):
            losses.append({"epoch": epoch + 1, "train_normalized_mse": loss.item()})
    net.eval()
    started = time.perf_counter()
    with torch.no_grad():
        output = net(x_test) * scale
    inference_seconds_per_example = (time.perf_counter() - started) / len(test_examples)
    error = output - y_test
    metrics = {
        "held_out_mae_chips": error.abs().mean().item(),
        "held_out_rmse_chips": torch.sqrt((error**2).mean()).item(),
        "zero_predictor_mae_chips": y_test.abs().mean().item(),
        "shallow_vs_reference_mae_chips": sum(shallow_abs_errors)
        / len(shallow_abs_errors),
        "reference_mean_nash_conv": sum(ref_nc) / len(ref_nc),
        "test_shallow_mean_nash_conv": sum(shallow_nc) / len(shallow_nc),
        "reference_mean_solve_seconds": sum(reference_times) / len(reference_times),
        "shallow_mean_solve_seconds": sum(shallow_times) / len(shallow_times),
        "network_inference_seconds_per_hand_value": inference_seconds_per_example,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.model_output)
    return {
        "experiment": "tiny_range_river_value_model",
        "seed": args.seed,
        "train_positions": args.train_positions,
        "test_positions": args.test_positions,
        "reference_iterations": args.reference_iterations,
        "shallow_iterations": args.shallow_iterations,
        "epochs": args.epochs,
        "lr": args.lr,
        "label_definition": "hand-conditional root EV under average river CFR policy",
        "model_path": str(args.model_output),
        "training_losses": losses,
        "metrics": metrics,
        "positions": raw,
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
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--train-positions", type=int, default=60)
    parser.add_argument("--test-positions", type=int, default=20)
    parser.add_argument("--reference-iterations", type=int, default=200)
    parser.add_argument("--shallow-iterations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/river_value.json")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/river_value.pt")
    )
    args = parser.parse_args()
    if (
        min(
            args.train_positions,
            args.test_positions,
            args.reference_iterations,
            args.shallow_iterations,
            args.epochs,
        )
        < 1
        or args.shallow_iterations >= args.reference_iterations
    ):
        parser.error("Use positive counts and fewer shallow than reference iterations")
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
