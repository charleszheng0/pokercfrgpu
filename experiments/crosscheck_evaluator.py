"""Optional independent rank cross-check against treys (not a runtime dependency)."""

import argparse
import json
import random
from pathlib import Path

from poker_cpu.cards import DECK, evaluate


def run(trials, seed):
    from treys import Card, Evaluator

    rng = random.Random(seed)
    oracle = Evaluator()
    for i in range(trials):
        cards = rng.sample(DECK, 9)
        board = cards[:5]
        own = cards[5:7]
        other = cards[7:9]
        own_score = evaluate(tuple(board + own))
        other_score = evaluate(tuple(board + other))
        ours = (own_score > other_score) - (own_score < other_score)
        theirs = oracle.evaluate([Card.new(c) for c in other], [Card.new(c) for c in board])
        theirs -= oracle.evaluate([Card.new(c) for c in own], [Card.new(c) for c in board])
        their_sign = (theirs > 0) - (theirs < 0)
        if ours != their_sign:
            raise AssertionError(f"Mismatch at trial {i}: {cards}, ours={ours}, oracle={their_sign}")
    return {"trials": trials, "seed": seed, "mismatches": 0, "oracle": "treys"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, default=Path("experiments/evaluator_crosscheck_seed19.json"))
    args = parser.parse_args()
    result = run(args.trials, args.seed)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
