"""Compare a frozen NFSP average policy with tabular CFR on custom two-round Leduc.

Requirements: PyTorch and the local small_games package.
Assumed rules: deck JJQQKK, ante 1, bets 2/4 by round, up to 2 raises
per round, seat 0 acts first on both rounds, one public card after round 1.

Usage: python3 -m small_games.compare_cfr_nfsp --iterations 1500

Rules are cross-checked against small_games/leduc.py in tests/test_nfsp.py.
"""

import argparse
import math
from pathlib import Path
from dataclasses import dataclass, replace
from collections import defaultdict
from functools import lru_cache
from itertools import permutations

import torch

CARDS = "JQK"
ACTIONS = ("check", "bet", "fold", "call", "raise")
DECK = "JJQQKK"
DEALS = tuple(permutations(range(6), 3))  # 120 equally likely physical-card deals


@dataclass(frozen=True)
class State:
    cards: tuple  # two private card ranks and eventual board rank (hidden in round 0)
    round: int = 0
    current: int = 0
    pot: int = 2
    paid: tuple = (1, 1)  # Total contributed by each player
    round_paid: tuple = (0, 0)  # Contributions in current betting round
    raises: int = 0  # Number of raises in the current round
    history: tuple = ()  # (round, seat, action)
    finished: bool = False
    folded: int = -1  # player who folded, or -1 for showdown


def initial(deal):
    return State(tuple(DECK[i] for i in deal))


def legal(s):
    if s.finished:
        return ()
    owed = max(s.round_paid) - s.round_paid[s.current]
    if owed > 0:
        return ("fold", "call", "raise") if s.raises < 2 else ("fold", "call")
    return ("check", "bet") if s.raises < 2 else ("check",)


def advance(s, action):
    """Advance a hand under the explicit rules described in this file."""
    assert action in legal(s), (s, action, legal(s))
    p = s.current
    owed = max(s.round_paid) - s.round_paid[p]
    amount = 2 if s.round == 0 else 4
    paid = list(s.paid)
    round_paid = list(s.round_paid)
    history = s.history + ((s.round, p, action),)
    raises = s.raises
    pot = s.pot
    if action == "fold":
        return replace(s, history=history, finished=True, folded=p)
    if action in ("bet", "raise"):
        chips = owed + amount
        paid[p] += chips
        round_paid[p] += chips
        pot += chips
        if action == "raise":
            raises += 1
    elif action == "call":
        paid[p] += owed
        round_paid[p] += owed
        pot += owed
    # A round finishes on check-check, or on a call that matches a bet/raise.
    this_round = [a for r, _, a in history if r == s.round]
    close = action == "call" or (
        len(this_round) >= 2 and this_round[-2:] == ["check", "check"]
    )
    if close:
        if s.round == 1:
            return replace(
                s,
                paid=tuple(paid),
                round_paid=tuple(round_paid),
                pot=pot,
                raises=raises,
                history=history,
                finished=True,
            )
        return replace(
            s,
            round=1,
            current=0,
            paid=tuple(paid),
            pot=pot,
            round_paid=(0, 0),
            raises=0,
            history=history,
        )
    return replace(
        s,
        current=1 - p,
        paid=tuple(paid),
        round_paid=tuple(round_paid),
        raises=raises,
        pot=pot,
        history=history,
    )


def terminal_payoff(s, player):
    """Return net chips won by player (sum of the two players' payoffs is zero)."""
    assert s.finished
    if s.folded != -1:
        winner = 1 - s.folded
    else:
        board = s.cards[2]
        ranks = {c: i for i, c in enumerate(CARDS)}
        strengths = [(int(s.cards[i] == board), ranks[s.cards[i]]) for i in range(2)]
        winner = (
            None if strengths[0] == strengths[1] else int(strengths[1] > strengths[0])
        )
    if winner is None:
        return s.pot / 2.0 - s.paid[player]
    return (s.pot if winner == player else 0) - s.paid[player]


def information_set(s, player):
    """A player cannot see the opponent's card or undealt public card."""
    return (player, s.cards[player], s.cards[2] if s.round == 1 else None, s.history)


def observation(s):
    """Construct the format expected by the user's encoder.encode_state."""
    return {
        "private_card": s.cards[s.current],
        "public_card": s.cards[2] if s.round == 1 else None,
        "round": s.round,
        "pot": s.pot,
        "to_call": max(s.round_paid) - s.round_paid[s.current],
        "raises_remaining": 2 - s.raises,
        "current_player": s.current,
        "history": list(s.history),
    }


@lru_cache(maxsize=None)
def children(s):
    return {a: advance(s, a) for a in legal(s)}


class CFR:
    def __init__(self):
        self.regrets = {}
        self.sum_policy = {}

    def strategy(self, key, actions):
        r = self.regrets.setdefault(key, {a: 0.0 for a in actions})
        positive = [max(0.0, r[a]) for a in actions]
        z = sum(positive)
        return {
            a: positive[i] / z if z else 1.0 / len(actions)
            for i, a in enumerate(actions)
        }

    def traverse(self, s, reaches, chance):
        if s.finished:
            return terminal_payoff(s, 0)
        p = s.current
        actions = legal(s)
        key = information_set(s, p)
        strat = self.strategy(key, actions)
        accum = self.sum_policy.setdefault(key, {a: 0.0 for a in actions})
        for a in actions:
            accum[a] += chance * reaches[p] * strat[a]
        evs = {}
        node_ev = 0.0
        for a in actions:
            next_reaches = list(reaches)
            next_reaches[p] *= strat[a]
            evs[a] = self.traverse(children(s)[a], next_reaches, chance)
            node_ev += strat[a] * evs[a]
        # Every payoff is expressed from player 0's perspective.
        sign = 1 if p == 0 else -1
        for a in actions:
            self.regrets[key][a] += chance * reaches[1 - p] * sign * (evs[a] - node_ev)
        return node_ev

    def train(self, iterations):
        starts = [initial(deal) for deal in DEALS]
        for iteration in range(1, iterations + 1):
            for s in starts:
                self.traverse(s, (1.0, 1.0), 1.0 / len(starts))
            if iteration % max(1, iterations // 5) == 0:
                print(f"CFR iteration {iteration}/{iterations}", flush=True)

    def average(self, s):
        actions = legal(s)
        sums = self.sum_policy.get(information_set(s, s.current))
        if sums is None:
            return {a: 1 / len(actions) for a in actions}
        total = sum(sums.values())
        return {a: sums[a] / total if total else 1 / len(actions) for a in actions}


class SynchronousCFR(CFR):
    """Full-tree CFR with strategies frozen across all physical deals per step.

    The original ``CFR`` above updates regrets during a deal sweep. Keeping it
    available preserves previous comparisons; this class is the reference for
    standard simultaneous full-tree CFR experiments.
    """

    def __init__(self):
        super().__init__()
        self.iterations = 0

    def step(self):
        updates = {}
        strategies = {}

        def visit(state, reaches):
            if state.finished:
                return terminal_payoff(state, 0)
            player = state.current
            actions = legal(state)
            key = information_set(state, player)
            if key not in strategies:
                strategies[key] = self.strategy(key, actions)
            strat = strategies[key]
            sums = self.sum_policy.setdefault(key, {a: 0.0 for a in actions})
            for action in actions:
                sums[action] += reaches[player] * strat[action] / len(DEALS)
            child_values = {}
            node_value = 0.0
            for action in actions:
                next_reaches = list(reaches)
                next_reaches[player] *= strat[action]
                child_values[action] = visit(children(state)[action], next_reaches)
                node_value += strat[action] * child_values[action]
            changes = updates.setdefault(key, {a: 0.0 for a in actions})
            sign = 1 if player == 0 else -1
            for action in actions:
                changes[action] += (
                    reaches[1 - player]
                    * sign
                    * (child_values[action] - node_value)
                    / len(DEALS)
                )
            return node_value

        for deal in DEALS:
            visit(initial(deal), (1.0, 1.0))
        for key, changes in updates.items():
            for action, delta in changes.items():
                self.regrets[key][action] += delta
        self.iterations += 1

    def train(self, iterations):
        for _ in range(iterations):
            self.step()
        return self


class NeuralPolicy:
    def __init__(self, path):
        from .encoder import encode_state, INPUT_DIM
        from .train_nfsp import AveragePolicyNetwork

        self.encode = encode_state
        self.network = AveragePolicyNetwork()
        self.network.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        self.network.eval()
        self.cache = {}
        # Fail clearly if model / encoder dimensions do not match.
        assert self.network.net[0].in_features == INPUT_DIM

    def __call__(self, s):
        key = information_set(s, s.current)
        if key not in self.cache:
            with torch.no_grad():
                logits = self.network(self.encode(observation(s)))
                # Import action list from the project, not assumed ordering.
                from .dqn import ACTIONS as network_actions

                ids = [network_actions.index(a) for a in legal(s)]
                probs = torch.softmax(logits[ids], dim=0).tolist()
            self.cache[key] = dict(zip(legal(s), probs))
        return self.cache[key]


def profile_value(policy0, policy1):
    """Exactly sum the expected seat-0 payoff over all deals and actions."""

    @lru_cache(maxsize=None)
    def visit(s):
        if s.finished:
            return terminal_payoff(s, 0)
        policy = policy0 if s.current == 0 else policy1
        return sum(prob * visit(children(s)[a]) for a, prob in policy(s).items())

    return sum(visit(initial(d)) for d in DEALS) / len(DEALS)


def best_response(fixed_policy, br_player):
    """Exact infoset-constrained BR: group all hidden deals before maximizing."""
    by_info = defaultdict(list)

    def collect(s, opp_reach, chance):
        if s.finished:
            return
        if s.current == br_player:
            by_info[information_set(s, br_player)].append((s, opp_reach * chance))
            for nxt in children(s).values():
                collect(nxt, opp_reach, chance)
        else:
            probabilities = fixed_policy(s)
            for a, nxt in children(s).items():
                collect(nxt, opp_reach * probabilities[a], chance)

    for d in DEALS:
        collect(initial(d), 1.0, 1.0 / len(DEALS))

    selected = {}

    @lru_cache(maxsize=None)
    def continuation(s):
        if s.finished:
            return terminal_payoff(s, br_player)
        if s.current == br_player:
            return continuation(children(s)[selected[information_set(s, br_player)]])
        return sum(
            prob * continuation(children(s)[a]) for a, prob in fixed_policy(s).items()
        )

    # Every deeper information set must be solved before its ancestors.
    for key in sorted(by_info, key=lambda k: len(k[3]), reverse=True):
        nodes = by_info[key]
        moves = legal(nodes[0][0])
        scores = {
            a: sum(weight * continuation(children(s)[a]) for s, weight in nodes)
            for a in moves
        }
        selected[key] = max(moves, key=lambda a: scores[a])
        continuation.cache_clear()

    def br_policy(s):
        chosen = selected[information_set(s, s.current)]
        return {a: float(a == chosen) for a in legal(s)}

    payoff = (
        profile_value(br_policy, fixed_policy)
        if br_player == 0
        else -profile_value(fixed_policy, br_policy)
    )
    return payoff


def nash_conv(p0, p1):
    """Sum of both players' gains from unilaterally deviating to best responses."""
    baseline = profile_value(p0, p1)
    br0 = best_response(p1, 0)
    br1 = best_response(p0, 1)
    gain0, gain1 = br0 - baseline, br1 + baseline
    return gain0 + gain1, br0, br1, baseline


def main():
    parser = argparse.ArgumentParser()
    legacy_dir = Path(__file__).resolve().parents[1] / "artifacts" / "legacy"
    parser.add_argument("--model", default=legacy_dir / "nfsp_player_0.pt")
    parser.add_argument(
        "--model1",
        default=legacy_dir / "nfsp_player_1.pt",
        help="Seat-1 average-policy model",
    )
    parser.add_argument("--iterations", type=int, default=1500)
    args = parser.parse_args()

    print("Training tabular CFR on exactly the explicitly defined Leduc rules...")
    solver = CFR()
    solver.train(args.iterations)
    cfr = solver.average

    print("Loading saved NFSP average-policy network(s)...")
    n0 = NeuralPolicy(args.model)
    n1 = NeuralPolicy(args.model1)

    for label, pair in [("CFR", (cfr, cfr)), ("NFSP", (n0, n1))]:
        print(f"Computing exact best responses for {label}...", flush=True)
        nc, br0, br1, utility = nash_conv(*pair)
        print(
            f"{label}: NashConv={nc:.6f}, exploitability=NashConv/2={nc/2:.6f} chips/hand"
        )
        print(f"  Seat 0 EV={utility:+.6f}; BR0 value={br0:+.6f}; BR1 value={br1:+.6f}")

    print("\nOpening decisions (board not yet visible):")
    for card in CARDS:
        # Build any physically valid deal with this private card.
        deal = next(d for d in DEALS if DECK[d[0]] == card)
        s = initial(deal)
        print(f"  Holding {card}:")
        for name, pi in [("CFR", cfr), ("NFSP", n0)]:
            probs = pi(s)
            print(
                "   ", f"{name:4}:", "  ".join(f"{a}={probs[a]:.1%}" for a in legal(s))
            )

    print("\nRule/encoder equivalence is covered by tests/test_nfsp.py.")


if __name__ == "__main__":
    main()
