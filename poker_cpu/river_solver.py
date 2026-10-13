"""Tiny exact-range river CFR reference with a deliberately restricted tree.

This solves only a supplied river public board and small explicit private-hand
ranges. It is a CPU research reference, not a full Hold'em blueprint or a safe
continual re-solver. Wager targets are min, half-pot, pot and effective all-in;
at most one raise is allowed after the first bet.
"""

from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from math import isfinite

from .cards import evaluate, parse_cards
from .engine import Action


@dataclass(frozen=True)
class RiverState:
    current: int = 0
    paid: tuple = (0, 0)
    history: tuple = ()
    raises: int = 0
    last_raise: int = 2
    done: bool = False
    winner: int = -1  # -1 means showdown


@dataclass(frozen=True)
class WeightedRange:
    """A normalized explicit private-hand belief with blocker/action updates."""

    hands: tuple
    weights: tuple

    def __post_init__(self):
        hands = tuple(parse_cards(hand) for hand in self.hands)
        weights = tuple(float(weight) for weight in self.weights)
        if not hands or any(len(hand) != 2 for hand in hands):
            raise ValueError("A range needs at least one two-card hand")
        if len(set(hands)) != len(hands) or len(weights) != len(hands):
            raise ValueError("Range hands must be unique and match the weights")
        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Range weights must be finite and nonnegative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("Range needs positive total weight")
        object.__setattr__(self, "hands", hands)
        object.__setattr__(self, "weights", tuple(weight / total for weight in weights))

    def condition(self, dead_cards=(), action_likelihoods=None):
        """Condition on public/known cards and an observed action likelihood.

        `action_likelihoods` maps each private hand to P(observed action | hand).
        Missing hands have likelihood zero. The result is renormalized, and an
        impossible observation raises instead of silently creating a uniform range.
        """
        dead = set(parse_cards(dead_cards)) if dead_cards else set()
        has_action_observation = action_likelihoods is not None
        action_likelihoods = action_likelihoods or {}
        hands, weights = [], []
        for hand, prior in zip(self.hands, self.weights):
            if dead.intersection(hand):
                continue
            likelihood = float(
                action_likelihoods.get(hand, 0.0 if has_action_observation else 1.0)
            )
            if not isfinite(likelihood) or not 0.0 <= likelihood <= 1.0:
                raise ValueError("Action likelihoods must be finite probabilities")
            posterior = prior * likelihood
            if posterior > 0:
                hands.append(hand)
                weights.append(posterior)
        if not weights:
            raise ValueError("Observation leaves no possible private hands")
        return WeightedRange(tuple(hands), tuple(weights))


class RiverGame:
    def __init__(self, board, ranges, pot=8, stack=10, big_blind=2, range_weights=None):
        self.board = parse_cards(board)
        if len(self.board) != 5 or pot < 0 or stack < big_blind or big_blind < 1:
            raise ValueError(
                "Use a five-card board, nonnegative pot, and positive effective stack"
            )
        self.pot, self.stack, self.big_blind = pot, stack, big_blind
        if len(ranges) != 2 or not all(ranges):
            raise ValueError("Both players need nonempty private-hand ranges")
        if range_weights is None:
            range_weights = tuple((1.0,) * len(player_range) for player_range in ranges)
        if len(range_weights) != 2:
            raise ValueError("Range weights need one vector per player")
        self.range_beliefs = tuple(
            WeightedRange(tuple(player_range), tuple(range_weights[p]))
            for p, player_range in enumerate(ranges)
        )
        self.ranges = tuple(belief.hands for belief in self.range_beliefs)
        range_weights = tuple(belief.weights for belief in self.range_beliefs)
        for player_range in self.ranges:
            if any(
                len(hand) != 2 or set(hand) & set(self.board) for hand in player_range
            ):
                raise ValueError("Each private hand needs two cards outside the board")
            if len(set(player_range)) != len(player_range):
                raise ValueError("Duplicate hands in a range")
        weights = [
            (a, b, range_weights[0][i] * range_weights[1][j])
            for i, a in enumerate(self.ranges[0])
            for j, b in enumerate(self.ranges[1])
            if not set(a) & set(b) and range_weights[0][i] * range_weights[1][j] > 0
        ]
        total_weight = sum(weight for _, _, weight in weights)
        if total_weight <= 0:
            raise ValueError("Ranges have no compatible joint deals")
        self.deals = tuple((a, b, weight / total_weight) for a, b, weight in weights)
        self.strength = {
            hand: evaluate(hand + self.board) for hands in self.ranges for hand in hands
        }

    @lru_cache(maxsize=None)
    def legal(self, state):
        if state.done:
            return ()
        p = state.current
        owed = max(state.paid) - state.paid[p]
        actions = [Action("fold"), Action("call")] if owed else [Action("check")]
        highest = max(state.paid)
        if highest >= self.stack or (owed and state.raises >= 1):
            return tuple(actions)
        minimum = highest + state.last_raise
        cap = self.stack
        pot_after_call = self.pot + sum(state.paid) + owed
        targets = (
            {cap}
            if cap < minimum
            else {
                minimum,
                min(cap, highest + max(state.last_raise, pot_after_call // 2)),
                min(cap, highest + max(state.last_raise, pot_after_call)),
                cap,
            }
        )
        kind = "raise" if owed else "bet"
        return tuple(
            actions
            + [Action(kind, target) for target in sorted(targets) if target > highest]
        )

    @lru_cache(maxsize=None)
    def advance(self, state, action):
        if action not in self.legal(state):
            raise ValueError("Illegal abstract river action")
        p = state.current
        q = 1 - p
        history = state.history + (action,)
        if action.kind == "fold":
            return RiverState(
                q, state.paid, history, state.raises, state.last_raise, True, q
            )
        if action.kind == "check":
            return RiverState(
                q,
                state.paid,
                history,
                state.raises,
                state.last_raise,
                len(history) >= 2 and history[-2].kind == "check",
            )
        paid = list(state.paid)
        if action.kind == "call":
            paid[p] = max(paid)
            return RiverState(
                q, tuple(paid), history, state.raises, state.last_raise, True
            )
        increment = action.amount - max(paid)
        paid[p] = action.amount
        return RiverState(
            q,
            tuple(paid),
            history,
            state.raises + int(action.kind == "raise"),
            increment,
            False,
        )

    def terminal_payoff(self, state, hands):
        if not state.done:
            raise ValueError("Payoff needs a terminal state")
        winner = state.winner
        if winner == -1:
            a, b = self.strength[hands[0]], self.strength[hands[1]]
            winner = 0 if a > b else 1 if b > a else -1
        if winner == -1:
            return (state.paid[1] - state.paid[0]) / 2
        return (
            self.pot / 2 + state.paid[1]
            if winner == 0
            else -self.pot / 2 - state.paid[0]
        )

    @staticmethod
    def info(state, hands):
        return (state.current, hands[state.current], state.history)

    def expected_value(self, policy0, policy1):
        return sum(
            weight * self.value_for_deal((a, b), policy0, policy1)
            for a, b, weight in self.deals
        )

    def value_for_deal(self, hands, policy0, policy1):
        def visit(state, hands):
            if state.done:
                return self.terminal_payoff(state, hands)
            policy = policy0 if state.current == 0 else policy1
            return sum(
                prob * visit(self.advance(state, action), hands)
                for action, prob in policy(state, hands).items()
            )

        return visit(RiverState(last_raise=self.big_blind), hands)


class RiverCFR:
    def __init__(self, game):
        self.game = game
        self.regrets = {}
        self.strategy_sum = {}
        self.iterations = 0

    def step(self):
        game = self.game
        policies = {}
        changes = {}

        def visit(state, hands, reach, chance):
            if state.done:
                return game.terminal_payoff(state, hands)
            key = game.info(state, hands)
            actions = game.legal(state)
            if key not in policies:
                old = self.regrets.setdefault(key, {a: 0.0 for a in actions})
                positive = {a: max(0.0, old[a]) for a in actions}
                total = sum(positive.values())
                policies[key] = {
                    a: positive[a] / total if total else 1 / len(actions)
                    for a in actions
                }
            strategy = policies[key]
            p = state.current
            sums = self.strategy_sum.setdefault(key, {a: 0.0 for a in actions})
            for action in actions:
                sums[action] += chance * reach[p] * strategy[action]
            action_values = {}
            node = 0.0
            for action in actions:
                next_reach = list(reach)
                next_reach[p] *= strategy[action]
                action_values[action] = visit(
                    game.advance(state, action), hands, next_reach, chance
                )
                node += strategy[action] * action_values[action]
            delta = changes.setdefault(key, {a: 0.0 for a in actions})
            sign = 1 if p == 0 else -1
            for action in actions:
                delta[action] += (
                    chance * reach[1 - p] * sign * (action_values[action] - node)
                )
            return node

        for a, b, chance in game.deals:
            visit(RiverState(last_raise=game.big_blind), (a, b), (1.0, 1.0), chance)
        for key, delta in changes.items():
            for action, value in delta.items():
                self.regrets[key][action] += value
        self.iterations += 1

    def train(self, iterations):
        for _ in range(iterations):
            self.step()
        return self

    def average(self, state, hands):
        actions = self.game.legal(state)
        sums = self.strategy_sum.get(self.game.info(state, hands))
        if not sums:
            return {a: 1 / len(actions) for a in actions}
        total = sum(sums.values())
        return {a: sums[a] / total for a in actions}

    def root_hand_values(self):
        """Conditional showdown EV by private hand under the average policy."""
        result = ({}, {})
        for player in (0, 1):
            for hand in self.game.ranges[player]:
                compatible = [
                    (a, b, chance)
                    for a, b, chance in self.game.deals
                    if (a, b)[player] == hand
                ]
                if not compatible:
                    continue
                weight = sum(x[2] for x in compatible)
                total = 0.0
                for a, b, chance in compatible:
                    value = self.game.value_for_deal((a, b), self.average, self.average)
                    total += chance * (value if player == 0 else -value)
                result[player][hand] = total / weight
        return result


def best_response(game, fixed_policy, player):
    """Exact information-set-constrained response within the supplied ranges."""
    grouped = defaultdict(list)

    def collect(state, hands, opponent_reach, chance):
        if state.done:
            return
        if state.current == player:
            grouped[game.info(state, hands)].append(
                (state, hands, opponent_reach * chance)
            )
            for action in game.legal(state):
                collect(game.advance(state, action), hands, opponent_reach, chance)
        else:
            for action, prob in fixed_policy(state, hands).items():
                collect(
                    game.advance(state, action), hands, opponent_reach * prob, chance
                )

    start = RiverState(last_raise=game.big_blind)
    for a, b, chance in game.deals:
        collect(start, (a, b), 1.0, chance)
    chosen = {}

    @lru_cache(maxsize=None)
    def continuation(state, hands):
        if state.done:
            payoff = game.terminal_payoff(state, hands)
            return payoff if player == 0 else -payoff
        if state.current == player:
            return continuation(
                game.advance(state, chosen[game.info(state, hands)]), hands
            )
        return sum(
            prob * continuation(game.advance(state, action), hands)
            for action, prob in fixed_policy(state, hands).items()
        )

    for key in sorted(grouped, key=lambda item: len(item[2]), reverse=True):
        nodes = grouped[key]
        actions = game.legal(nodes[0][0])
        scores = {
            action: sum(
                weight * continuation(game.advance(state, action), hands)
                for state, hands, weight in nodes
            )
            for action in actions
        }
        chosen[key] = max(actions, key=lambda action: scores[action])
        continuation.cache_clear()

    def policy(state, hands):
        action = chosen[game.info(state, hands)]
        return {
            candidate: float(candidate == action) for candidate in game.legal(state)
        }

    return (
        game.expected_value(policy, fixed_policy)
        if player == 0
        else -game.expected_value(fixed_policy, policy)
    )


def nash_conv(game, policy0, policy1):
    return best_response(game, policy1, 0) + best_response(game, policy0, 1)
