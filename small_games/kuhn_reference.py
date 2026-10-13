"""Synchronous full-tree CFR and exact best responses for standard Kuhn poker.

Both players ante one chip. P0 acts first. A bet adds one chip; a call adds
one. Payoffs are net chips to P0. This module is independent of the original
teaching script in ``kuhn.py`` and gives it a measurable reference.
"""

from itertools import product

CARDS = ("J", "Q", "K")
RANK = {card: i for i, card in enumerate(CARDS)}
DEALS = tuple((a, b) for a in CARDS for b in CARDS if a != b)
ACTIONS = {"": ("c", "b"), "c": ("c", "b"), "b": ("f", "k"), "cb": ("f", "k")}
TERMINALS = ("cc", "bf", "bk", "cbf", "cbk")
INFOSETS = {
    0: tuple((0, card, h) for card in CARDS for h in ("", "cb")),
    1: tuple((1, card, h) for card in CARDS for h in ("c", "b")),
}


def actor(history):
    return 0 if history in ("", "cb") else 1


def payoff(history, deal):
    if history == "bf":
        return 1
    if history == "cbf":
        return -1
    amount = 1 if history == "cc" else 2
    return amount if RANK[deal[0]] > RANK[deal[1]] else -amount


def value(policy0, policy1):
    """Exact expected P0 net reward over six equally likely ordered deals."""

    def visit(history, deal):
        if history in TERMINALS:
            return payoff(history, deal)
        p = actor(history)
        info = (p, deal[p], history)
        probs = (policy0 if p == 0 else policy1)[info]
        return sum(
            probs[i] * visit(history + a, deal) for i, a in enumerate(ACTIONS[history])
        )

    return sum(visit("", deal) for deal in DEALS) / len(DEALS)


def pure_policies(player):
    for choices in product((0, 1), repeat=len(INFOSETS[player])):
        yield {
            info: (1.0, 0.0) if choice == 0 else (0.0, 1.0)
            for info, choice in zip(INFOSETS[player], choices)
        }


def nash_conv(policy0, policy1):
    """Exact sum of unilateral gains; no hidden-card peeking by best response."""
    baseline = value(policy0, policy1)
    br0 = max(value(pure, policy1) for pure in pure_policies(0))
    br1_p0_value = min(value(policy0, pure) for pure in pure_policies(1))
    return br0 - br1_p0_value, baseline, br0, br1_p0_value


def regret_match(regrets):
    positive = [max(0.0, x) for x in regrets]
    total = sum(positive)
    return tuple(x / total for x in positive) if total else (0.5, 0.5)


class CFR:
    def __init__(self):
        self.regrets = {
            info: [0.0, 0.0] for player in (0, 1) for info in INFOSETS[player]
        }
        self.strategy_sum = {info: [0.0, 0.0] for info in self.regrets}
        self.iterations = 0

    def step(self):
        # Freeze every information-set strategy for the whole iteration.
        policy = {info: regret_match(r) for info, r in self.regrets.items()}
        changes = {info: [0.0, 0.0] for info in self.regrets}

        def visit(history, deal, reach0, reach1):
            if history in TERMINALS:
                return payoff(history, deal)
            p = actor(history)
            info = (p, deal[p], history)
            probs = policy[info]
            own_reach = reach0 if p == 0 else reach1
            opponent_reach = reach1 if p == 0 else reach0
            for i in (0, 1):
                self.strategy_sum[info][i] += own_reach * probs[i] / len(DEALS)
            child = []
            for i, a in enumerate(ACTIONS[history]):
                child.append(
                    visit(
                        history + a,
                        deal,
                        reach0 * probs[i] if p == 0 else reach0,
                        reach1 * probs[i] if p == 1 else reach1,
                    )
                )
            node = sum(probs[i] * child[i] for i in (0, 1))
            sign = 1 if p == 0 else -1
            for i in (0, 1):
                changes[info][i] += (
                    opponent_reach * sign * (child[i] - node) / len(DEALS)
                )
            return node

        for deal in DEALS:
            visit("", deal, 1.0, 1.0)
        for info in self.regrets:
            for i in (0, 1):
                self.regrets[info][i] += changes[info][i]
        self.iterations += 1

    def train(self, iterations):
        for _ in range(iterations):
            self.step()
        return self

    def average(self):
        result = {}
        for info, sums in self.strategy_sum.items():
            total = sum(sums)
            result[info] = tuple(x / total for x in sums) if total else (0.5, 0.5)
        return result
