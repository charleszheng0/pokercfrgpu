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
from random import Random

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
        if not isfinite(total) or total <= 0:
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
        if (
            len(self.board) != 5
            or not all(isinstance(value, int) for value in (pot, stack, big_blind))
            or pot < 0
            or stack < 1
            or big_blind < 1
        ):
            raise ValueError(
                "Use a five-card board, nonnegative integer pot, and positive integer stack/blind"
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

    def expected_value(self, policy0, policy1, root_state=None):
        root_state = root_state or RiverState(last_raise=self.big_blind)
        return sum(
            weight * _river_profile_value(
                self, root_state, (a, b),
                lambda state, hands: policy0(state, hands)
                if state.current == 0 else policy1(state, hands),
            )
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
    """Full-tree CFR on a tiny explicit-range river abstraction.

    ``variant`` supports vanilla CFR, CFR+ regret clipping with linear
    averaging, and DCFR with the standard discount exponents. These are
    experimental update rules; they do not change the restricted game.
    """

    def __init__(self, game, variant="cfr", root_player=0, dcfr_alpha=1.5,
                 dcfr_beta=0.0, dcfr_gamma=2.0, root_state=None,
                 chance_sampling=False, sampling_seed=0):
        if variant not in {"cfr", "cfr+", "dcfr"}:
            raise ValueError("variant must be 'cfr', 'cfr+', or 'dcfr'")
        if root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        if chance_sampling and variant != "cfr":
            raise ValueError("Chance sampling currently supports vanilla CFR only")
        self.game = game
        self.variant = variant
        self.root_player = root_player
        self.root_state = root_state or RiverState(current=root_player, last_raise=game.big_blind)
        if self.root_state.done or self.root_state.current != root_player:
            raise ValueError("root_state must be a nonterminal decision for root_player")
        self.dcfr_alpha = float(dcfr_alpha)
        self.dcfr_beta = float(dcfr_beta)
        self.dcfr_gamma = float(dcfr_gamma)
        self.regrets = {}
        self.strategy_sum = {}
        self.iterations = 0
        self.chance_sampling = chance_sampling
        self.random = Random(sampling_seed)

    def step(self):
        game = self.game
        if self.variant == "dcfr" and self.iterations:
            t = float(self.iterations)
            positive_discount = t**self.dcfr_alpha / (t**self.dcfr_alpha + 1.0)
            negative_discount = t**self.dcfr_beta / (t**self.dcfr_beta + 1.0)
            for regrets in self.regrets.values():
                for action, regret in regrets.items():
                    regrets[action] *= positive_discount if regret > 0 else negative_discount
            strategy_discount = (t / (t + 1.0)) ** self.dcfr_gamma
            for sums in self.strategy_sum.values():
                for action in sums:
                    sums[action] *= strategy_discount
        policies = {}
        changes = {}

        def visit(state, hands, reach, chance):
            if state.done:
                return game.terminal_payoff(state, hands)
            if getattr(state, "chance", False):
                if getattr(state, "all_in_runout", False):
                    return game.chance_terminal_payoff(state, hands)
                if self.chance_sampling:
                    outcomes = game.chance_outcomes(state, hands)
                    draw = self.random.random()
                    cumulative = 0.0
                    for child, probability in outcomes:
                        cumulative += probability
                        if draw < cumulative:
                            # Sampling from the true chance distribution makes
                            # the sampled utility an unbiased chance estimate.
                            return visit(child, hands, reach, chance)
                    return visit(outcomes[-1][0], hands, reach, chance)
                return sum(
                    probability * visit(child, hands, reach, chance * probability)
                    for child, probability in game.chance_outcomes(state, hands)
                )
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
            averaging_weight = (
                self.iterations + 1 if self.variant == "cfr+" else 1.0
            )
            for action in actions:
                sums[action] += averaging_weight * chance * reach[p] * strategy[action]
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
            visit(self.root_state, (a, b), (1.0, 1.0), chance)
        for key, delta in changes.items():
            for action, value in delta.items():
                updated = self.regrets[key][action] + value
                self.regrets[key][action] = max(0.0, updated) if self.variant == "cfr+" else updated
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
        if total <= 0:
            # A sampled information set can be visited only with zero own
            # reach; no average policy mass has been accumulated there.
            return {a: 1 / len(actions) for a in actions}
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
                    value = _river_profile_value(
                        self.game, self.root_state, (a, b), self.average
                    )
                    total += chance * (value if player == 0 else -value)
                result[player][hand] = total / weight
        return result

    def analyze_root(self, player=0, chosen_action=None, private_hand=None):
        """Return average frequencies and conditional chip EVs at root.

        Conditional action EV means expected net chips for ``player`` given
        their private hand, conditioned on compatible opponent hands, forcing
        the named root action, then following the learned average profile.
        It is distinct from CFR's reach-weighted counterfactual value.
        """
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        game = self.game
        if player != self.root_player:
            raise ValueError("analyze_root player must match solver root_player")
        root = self.root_state
        actions = game.legal(root)
        opponent = 1 - player

        per_hand = []
        marginal_mass = defaultdict(float)
        action_mass = defaultdict(float)
        action_ev_mass = defaultdict(float)
        all_values = {}
        for hand in game.ranges[player]:
            compatible = [
                (deal[player], deal[opponent], deal[2])
                for deal in game.deals
                if deal[player] == hand
            ]
            mass = sum(row[2] for row in compatible)
            if mass <= 0:
                continue
            # Information keys use the acting hand at the root, so construct
            # any placeholder opponent holding; it is not read by that node.
            placeholder = next(iter(game.ranges[opponent]))
            hand_pair = [None, None]
            hand_pair[player], hand_pair[opponent] = hand, placeholder
            frequencies = self.average(root, tuple(hand_pair))
            evs = {}
            for action in actions:
                total = 0.0
                for own, other, joint_weight in compatible:
                    hands = [None, None]
                    hands[player], hands[opponent] = own, other
                    next_state = game.advance(root, action)
                    value0 = _river_profile_value(game, next_state, tuple(hands), self.average)
                    total += joint_weight / mass * (value0 if player == 0 else -value0)
                evs[action] = total
            all_values[hand] = evs
            hand_prob = mass
            marginal_mass[hand] = hand_prob
            for action in actions:
                action_mass[action] += hand_prob * frequencies[action]
                action_ev_mass[action] += hand_prob * evs[action]
            per_hand.append({"hand": hand, "frequency": frequencies, "action_evs": evs})

        total_mass = sum(marginal_mass.values())
        aggregate_frequencies = {
            action: action_mass[action] / total_mass for action in actions
        }
        aggregate_action_evs = {
            action: action_ev_mass[action] / total_mass for action in actions
        }
        ev_loss = None
        private_hand = tuple(private_hand) if private_hand is not None else None
        if private_hand is not None and private_hand not in all_values:
            raise ValueError("private_hand must have positive compatible range mass")
        if chosen_action is not None:
            if chosen_action not in actions:
                raise ValueError("chosen_action must be legal at the root")
            if private_hand is not None:
                hand_evs = all_values[private_hand]
                ev_loss = max(hand_evs.values()) - hand_evs[chosen_action]
        return {
            "status": getattr(
                game, "solve_scope",
                f"{game.street}_one_street_abstracted"
                if hasattr(game, "street") else "river_abstracted",
            ),
            "validation_status": "exact_best_response_available",
            "variant": self.variant,
            "iterations": self.iterations,
            "player": player,
            "actions": actions,
            "per_hand": tuple(per_hand),
            "action_frequencies": aggregate_frequencies,
            "conditional_action_evs": aggregate_action_evs,
            "chosen_action_ev_loss": ev_loss,
            "range_average_action_gap": (
                max(aggregate_action_evs.values())
                - aggregate_action_evs[chosen_action]
                if chosen_action is not None else None
            ),
            "nash_conv": nash_conv(
                game, self.average, self.average, root_state=self.root_state
            ),
        }


def _river_profile_value(game, state, hands, policy):
    """Expected seat-0 payoff under a fixed profile from a specified node."""
    if state.done:
        return game.terminal_payoff(state, hands)
    if getattr(state, "chance", False):
        if getattr(state, "all_in_runout", False):
            return game.chance_terminal_payoff(state, hands)
        return sum(
            probability * _river_profile_value(game, child, hands, policy)
            for child, probability in game.chance_outcomes(state, hands)
        )
    return sum(
        probability * _river_profile_value(
            game, game.advance(state, action), hands, policy
        )
        for action, probability in policy(state, hands).items()
    )


def best_response(game, fixed_policy, player, root_state=None):
    """Exact information-set-constrained response within the supplied ranges."""
    grouped = defaultdict(list)

    def collect(state, hands, opponent_reach, chance):
        if state.done:
            return
        if getattr(state, "chance", False):
            if getattr(state, "all_in_runout", False):
                return
            for child, probability in game.chance_outcomes(state, hands):
                collect(child, hands, opponent_reach, chance * probability)
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

    start = root_state or RiverState(last_raise=game.big_blind)
    for a, b, chance in game.deals:
        collect(start, (a, b), 1.0, chance)
    chosen = {}

    @lru_cache(maxsize=None)
    def continuation(state, hands):
        if state.done:
            payoff = game.terminal_payoff(state, hands)
            return payoff if player == 0 else -payoff
        if getattr(state, "chance", False):
            if getattr(state, "all_in_runout", False):
                payoff = game.chance_terminal_payoff(state, hands)
                return payoff if player == 0 else -payoff
            return sum(
                probability * continuation(child, hands)
                for child, probability in game.chance_outcomes(state, hands)
            )
        if state.current == player:
            return continuation(
                game.advance(state, chosen[game.info(state, hands)]), hands
            )
        return sum(
            prob * continuation(game.advance(state, action), hands)
            for action, prob in fixed_policy(state, hands).items()
        )

    for key in sorted(grouped, key=lambda item: len(item[-1]), reverse=True):
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
        game.expected_value(policy, fixed_policy, root_state=start)
        if player == 0
        else -game.expected_value(fixed_policy, policy, root_state=start)
    )


def nash_conv(game, policy0, policy1, root_state=None):
    return best_response(game, policy1, 0, root_state) + best_response(
        game, policy0, 1, root_state
    )
