"""Minimal playable terminal interface for the validated CPU rules."""

import argparse

from .agents import HeuristicAgent, RandomAgent
from .engine import HoldemGame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    bot = RandomAgent(args.seed) if args.opponent == "random" else HeuristicAgent(args.seed)
    game = HoldemGame(seed=args.seed)
    while not game.done:
        p = game.current_player
        obs = game.observe(p)
        if p == 0:
            print(f"\nYour cards {obs['hole']} | board {obs['board']} | pot {obs['pot']} | stacks {obs['stacks']}")
            print("Actions:", *[f"{i}: {a.kind} {a.amount}" for i, a in enumerate(obs["legal_actions"])], sep="\n")
            try:
                action = obs["legal_actions"][int(input("Choose action: "))]
            except (ValueError, IndexError):
                print("Enter one of the listed numbers.")
                continue
        else:
            action = bot.act(obs)
            print(f"Opponent: {action.kind} {action.amount}")
        game.step(action)
    print(f"Board {tuple(game.board)} | opponent cards {game.hole[1]} | reward {game.rewards()}")


if __name__ == "__main__":
    main()
