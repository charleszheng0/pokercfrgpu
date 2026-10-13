"""Audit river value labels and test suit-invariant data augmentation.

The target remains the root, hand-conditional chip EV of the average strategy
from a finite-iteration tiny-range CFR solve. This is not a continuation value.
"""

import argparse
import itertools
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from experiments.river_value import INPUT_DIM, encode
from experiments.river_value_equity import equity_feature
from poker_cpu.cards import DECK
from poker_cpu.engine import HoldemGame
from poker_cpu.river_solver import RiverCFR, RiverGame, RiverState, nash_conv


def transform(cards, mapping):
    return tuple(card[0] + mapping[card[1]] for card in cards)


def build_examples(dataset, augment_per_position, seed):
    rng = random.Random(seed)
    train, test = [], []
    suit_maps = list(itertools.permutations("cdhs"))
    for row in dataset["positions"]:
        count = augment_per_position if row["split"] == "train" else 1
        mappings = [dict(zip("cdhs", perm)) for perm in suit_maps]
        if count < len(mappings):
            mappings = rng.sample(mappings, count)
        for mapping in mappings:
            board = transform(row["board"], mapping)
            ranges = tuple(
                tuple(transform(hand, mapping) for hand in player_range)
                for player_range in row["ranges"]
            )
            game = RiverGame(
                board,
                ranges,
                pot=8,
                stack=10,
                range_weights=row["range_weights"],
            )
            dest = train if row["split"] == "train" else test
            for label in row["labels"]:
                hand = transform(label["hand"], mapping)
                vector = encode(game, label["player"], hand, row["range_weights"])
                vector.append(
                    equity_feature(game, label["player"], hand, row["range_weights"])
                )
                dest.append((vector, label["ev"], row["split"], tuple(row["board"])))
    return train, test


def train_eval(train, test, seed, epochs):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = nn.Sequential(
        nn.Linear(INPUT_DIM + 1, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1),
    )
    x = torch.tensor([row[0] for row in train], dtype=torch.float32)
    y = torch.tensor([[row[1] / 14.0] for row in train], dtype=torch.float32)
    tx = torch.tensor([row[0] for row in test], dtype=torch.float32)
    ty = torch.tensor([[row[1]] for row in test], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    started = time.perf_counter()
    for _ in range(epochs):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_seconds = time.perf_counter() - started
    net.eval()
    with torch.no_grad():
        pred = net(tx) * 14.0
    error = (pred - ty).flatten()
    return {
        "test_examples": len(test),
        "held_out_mae_chips": error.abs().mean().item(),
        "held_out_rmse_chips": error.square().mean().sqrt().item(),
        "training_seconds": train_seconds,
    }


def audit_labels(dataset, recompute_positions=4):
    max_conditional_error = 0.0
    max_engine_payoff_error = 0.0
    tested_payoffs = 0
    selected = dataset["positions"][:recompute_positions]
    for row in selected:
        game = RiverGame(
            row["board"], row["ranges"], pot=8, stack=10,
            range_weights=row["range_weights"],
        )
        solver = RiverCFR(game).train(dataset["reference_iterations"])
        values = solver.root_hand_values()
        for label in row["labels"]:
            hand = tuple(label["hand"])
            max_conditional_error = max(
                max_conditional_error,
                abs(values[label["player"]][hand] - label["ev"]),
            )

        # Compare terminal payoff to actual engine stack delta for representative
        # showdown, tie, and fold outcomes that are legal in the abstract tree.
        for h0, h1, _ in game.deals:
            for winner in (0, 1):
                engine = HoldemGame(seed=1, stacks=(20, 20), blinds=(1, 2))
                engine.street = 3
                engine.board = list(game.board)
                engine.hole = [h0, h1]
                state = RiverState(paid=(2, 2), done=True, winner=winner)
                # At river root, each player has four chips of prior contribution
                # represented by the dead eight-chip pot. New street contributions
                # have already left the stacks and are included in the pot.
                engine.initial_stacks = [14, 14]
                engine.stacks = [10 - state.paid[0], 10 - state.paid[1]]
                engine.pot = 8 + sum(state.paid)
                engine.paid = list(state.paid)
                engine.street_paid = list(state.paid)
                engine._award(winner)
                payoff = game.terminal_payoff(state, (h0, h1))
                max_engine_payoff_error = max(
                    max_engine_payoff_error,
                    abs(payoff - engine.rewards()[0]),
                )
                tested_payoffs += 1
    return {
        "recomputed_positions": len(selected),
        "max_saved_label_vs_recomputed_label_abs_error_chips": max_conditional_error,
        "max_river_payoff_vs_engine_abs_error_chips": max_engine_payoff_error,
        "engine_payoff_cases": tested_payoffs,
    }


def convergence_check(dataset, positions=5):
    rows = [r for r in dataset["positions"] if r["split"] == "test"][:positions]
    metrics = []
    for row in rows:
        solves = {}
        for iterations in (20, 200, 500):
            game = RiverGame(
                row["board"], row["ranges"], pot=8, stack=10,
                range_weights=row["range_weights"],
            )
            solver = RiverCFR(game).train(iterations)
            values = solver.root_hand_values()
            solves[iterations] = (values, solver)
        mae = lambda a, b: sum(
            abs(solves[a][0][p][hand] - solves[b][0][p][hand])
            for p in (0, 1) for hand in solves[a][0][p]
        ) / sum(len(solves[a][0][p]) for p in (0, 1))
        metrics.append({
            "board": row["board"],
            "mae_20_vs_200_chips": mae(20, 200),
            "mae_200_vs_500_chips": mae(200, 500),
            "nash_conv_20": nash_conv(solves[20][1].game, solves[20][1].average, solves[20][1].average),
            "nash_conv_200": nash_conv(solves[200][1].game, solves[200][1].average, solves[200][1].average),
            "nash_conv_500": nash_conv(solves[500][1].game, solves[500][1].average, solves[500][1].average),
        })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("experiments/river_value_extended_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/river_value_diagnosis.json"))
    parser.add_argument("--augmentation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--convergence-positions", type=int, default=5)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text())
    base_train, base_test = build_examples(data, 1, args.seed)
    augmented_train, test = build_examples(data, args.augmentation, args.seed)
    train_rows = [row for row in data["positions"] if row["split"] == "train"]
    test_rows = [row for row in data["positions"] if row["split"] == "test"]
    train_boards = {tuple(row["board"]) for row in train_rows}
    test_boards = {tuple(row["board"]) for row in test_rows}
    label_values = [label["ev"] for row in data["positions"] for label in row["labels"]]
    result = {
        "experiment": "river_value_label_audit_and_suit_augmentation",
        "dataset": str(args.dataset),
        "source_reference_iterations": data["reference_iterations"],
        "target": "player-perspective root hand-conditional EV in chips under average CFR strategy",
        "train_positions": data["train_positions"],
        "test_positions": data["test_positions"],
        "data_audit": {
            "unique_train_boards": len(train_boards),
            "unique_test_boards": len(test_boards),
            "train_test_board_overlap": len(train_boards & test_boards),
            "train_hand_labels": sum(len(row["labels"]) for row in train_rows),
            "test_hand_labels": sum(len(row["labels"]) for row in test_rows),
            "target_min_chips": min(label_values),
            "target_max_chips": max(label_values),
            "test_labels_are_clustered_by_public_position": True,
        },
        "model_input_features": INPUT_DIM + 1,
        "range_schema": "3 explicit hands per player; independent exponential weights; incompatible joint hands removed and product weights normalized",
        "encoding": "board and private hand card indicators; each explicit range hand and normalized marginal weight; player id; pot and effective stack; plus exact showdown equity feature",
        "label_audit": audit_labels(data),
        "convergence": convergence_check(data, args.convergence_positions),
        "controlled_model_comparison": {
            "same_architecture_optimizer_seed_epochs_and_test_positions": True,
            "raw_training_examples": train_eval(base_train, base_test, args.seed, args.epochs),
            "suit_augmented_training_examples": train_eval(augmented_train, test, args.seed, args.epochs),
            "augmentation_per_training_position": args.augmentation,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
