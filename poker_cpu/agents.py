"""CPU baselines. Neither baseline is a trained or equilibrium policy."""

import random

from .cards import DECK, evaluate


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, observation):
        return self.rng.choice(observation["legal_actions"])


class HeuristicAgent:
    """Monte Carlo showdown equity with simple pot-odds and bet rules."""

    def __init__(self, seed=None, samples=96):
        self.rng = random.Random(seed)
        self.samples = samples

    def equity(self, obs):
        known = set(obs["hole"] + obs["board"])
        unseen = [c for c in DECK if c not in known]
        wins = 0.0
        count = 5 - len(obs["board"])
        for _ in range(self.samples):
            draw = self.rng.sample(unseen, 2 + count)
            board = obs["board"] + tuple(draw[2:])
            own = evaluate(obs["hole"] + board)
            other = evaluate(tuple(draw[:2]) + board)
            wins += 1.0 if own > other else 0.5 if own == other else 0.0
        return wins / self.samples

    def act(self, obs):
        legal = obs["legal_actions"]
        equity = self.equity(obs)
        owed = obs["to_call"]
        if owed:
            # Equity assumes showdown and ignores future betting and fold equity.
            if equity < owed / (obs["pot"] + owed) + 0.04:
                return next(a for a in legal if a.kind == "fold")
            raises = [a for a in legal if a.kind == "raise"]
            if equity > 0.78 and raises:
                return raises[0]
            return next(a for a in legal if a.kind == "call")
        bets = [a for a in legal if a.kind == "bet"]
        if equity > 0.72 and bets:
            return bets[0]
        return next(a for a in legal if a.kind == "check")
