"""Exact-chance, tiny-range postflop CFR game using the shared CFR engine.

The tree carries betting through all remaining streets. It is deliberately
bounded by small explicit ranges and a restricted action abstraction; a full
1,326-combo no-limit game is not tractable with this Python reference.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import isfinite

from .cards import DECK, evaluate, parse_cards
from .engine import Action
from .river_solver import WeightedRange, _river_profile_value


@lru_cache(maxsize=131072)
def _seven_strength(board, hand):
    """Reuse a private hand's rank across betting histories on one board."""
    return evaluate(hand + board)


@dataclass(frozen=True)
class PublicState:
    street: int
    board: tuple[str, ...]
    current: int
    paid: tuple[int, int] = (0, 0)
    spent: tuple[int, int] = (0, 0)
    history: tuple = ()
    checks: int = 0
    raises: int = 0
    last_raise: int = 2
    chance: bool = False
    done: bool = False
    winner: int = -1
    all_in_runout: bool = False


class MultiStreetGame:
    """Postflop public tree with exact private-deal and future-card chance.

    ``pot`` excludes all contributions on the root street. ``stack`` is each
    player's effective chip budget from that street's start. ``spent`` tracks
    those contributions across later streets, so no chip can be spent twice.
    """

    solve_scope = "postflop_multi_street_abstracted"

    def __init__(self, board, ranges, *, pot=8, stack=10, button=0,
                 big_blind=2, range_weights=None, bet_fractions=(0.5, 1.0),
                 max_raises=1):
        self.board = parse_cards(board)
        if len(self.board) not in (3, 4, 5):
            raise ValueError("Postflop solve requires a flop, turn, or river board")
        if type(pot) is not int or pot < 0 or type(stack) is not int or stack < 1:
            raise ValueError("Pot and effective stack must be integer chips")
        if type(big_blind) is not int or big_blind < 1 or button not in (0, 1):
            raise ValueError("Invalid blind or button")
        if type(max_raises) is not int or max_raises < 0:
            raise ValueError("max_raises must be nonnegative")
        fractions = tuple(float(value) for value in bet_fractions)
        if any(not isfinite(value) or value <= 0 for value in fractions):
            raise ValueError("Bet fractions must be positive finite values")
        if len(ranges) != 2 or not all(ranges):
            raise ValueError("Two nonempty private ranges are required")
        if range_weights is None:
            range_weights = tuple((1.0,) * len(hands) for hands in ranges)
        if len(range_weights) != 2:
            raise ValueError("Each seat needs range weights")
        self.range_beliefs = tuple(
            WeightedRange(tuple(hands), tuple(weights))
            for hands, weights in zip(ranges, range_weights)
        )
        self.ranges = tuple(belief.hands for belief in self.range_beliefs)
        if any(set(hand) & set(self.board) for seat in self.ranges for hand in seat):
            raise ValueError("Private ranges overlap the public board")
        joint = [
            (hand0, hand1, weight0 * weight1)
            for hand0, weight0 in zip(self.ranges[0], self.range_beliefs[0].weights)
            for hand1, weight1 in zip(self.ranges[1], self.range_beliefs[1].weights)
            if not set(hand0) & set(hand1) and weight0 * weight1 > 0
        ]
        total = sum(row[2] for row in joint)
        if total <= 0:
            raise ValueError("Ranges have no compatible joint deals")
        self.deals = tuple((a, b, mass / total) for a, b, mass in joint)
        self.pot, self.stack, self.big_blind = pot, stack, big_blind
        self.button = button
        self.bet_fractions = fractions
        self.max_raises = max_raises
        self.street = ("preflop", "flop", "turn", "river")[len(self.board) - 2]
        # Per-game caches avoid retaining every prior solve through a global
        # method cache keyed by self. Bounds also cap large chance-tree runs.
        self.legal = lru_cache(maxsize=10000)(self.legal)
        self.advance = lru_cache(maxsize=50000)(self.advance)
        self.chance_outcomes = lru_cache(maxsize=50000)(self.chance_outcomes)
        self.terminal_payoff = lru_cache(maxsize=50000)(self.terminal_payoff)
        self.chance_terminal_payoff = lru_cache(maxsize=20000)(self.chance_terminal_payoff)

    def initial_state(self, first_actor=None):
        actor = 1 - self.button if first_actor is None else first_actor
        if actor not in (0, 1):
            raise ValueError("Invalid first actor")
        return PublicState(len(self.board) - 2, self.board, actor,
                           last_raise=self.big_blind)

    def legal(self, state):
        if state.done or state.chance:
            return ()
        player, other = state.current, 1 - state.current
        owed = max(state.paid) - state.paid[player]
        actions = [Action("fold"), Action("call")] if owed else [Action("check")]
        remaining_self = self.stack - state.spent[player]
        remaining_other = self.stack - state.spent[other]
        cap = min(state.paid[player] + remaining_self,
                  state.paid[other] + remaining_other)
        highest = max(state.paid)
        if (remaining_self <= owed or remaining_other <= 0 or cap <= highest
                or owed and state.raises >= self.max_raises):
            return tuple(actions)
        minimum = highest + state.last_raise
        pot_after_call = self.pot + sum(state.spent) + owed
        targets = {cap} if cap < minimum else {minimum, cap}
        if cap >= minimum:
            for fraction in self.bet_fractions:
                target = highest + max(state.last_raise, int(pot_after_call * fraction))
                targets.add(min(cap, target))
        kind = "raise" if owed else "bet"
        return tuple(actions + [Action(kind, target) for target in sorted(targets)
                                if target > highest])

    def _round_end(self, state):
        if state.street == 3:
            return PublicState(**{**state.__dict__, "done": True, "current": -1})
        return PublicState(**{**state.__dict__, "current": -1, "chance": True,
                              "all_in_runout": max(state.spent) >= self.stack})

    def advance(self, state, action):
        if action not in self.legal(state):
            raise ValueError("Action is outside the restricted legal tree")
        player = state.current
        other = 1 - player
        history = state.history + ((state.street, player, action),)
        if action.kind == "fold":
            return PublicState(**{**state.__dict__, "history": history,
                                  "current": -1, "done": True, "winner": other})
        if action.kind == "check":
            next_state = PublicState(**{**state.__dict__, "history": history,
                                        "current": other, "checks": state.checks + 1})
            return self._round_end(next_state) if next_state.checks == 2 else next_state
        paid, spent = list(state.paid), list(state.spent)
        if action.kind == "call":
            delta = max(paid) - paid[player]
            paid[player] += delta
            spent[player] += delta
            return self._round_end(PublicState(**{**state.__dict__,
                "paid": tuple(paid), "spent": tuple(spent), "history": history,
                "current": other, "checks": 0}))
        highest = max(paid)
        delta = action.amount - paid[player]
        paid[player] = action.amount
        spent[player] += delta
        return PublicState(**{**state.__dict__, "paid": tuple(paid),
                              "spent": tuple(spent), "history": history,
                              "current": other, "checks": 0,
                              "raises": state.raises + int(action.kind == "raise"),
                              "last_raise": action.amount - highest})

    def chance_outcomes(self, state, hands):
        if not state.chance or state.done:
            raise ValueError("Expected a nonterminal chance node")
        dead = set(state.board + hands[0] + hands[1])
        remaining = tuple(card for card in DECK if card not in dead)
        probability = 1.0 / len(remaining)
        outcomes = []
        for card in remaining:
            board = state.board + (card,)
            street = state.street + 1
            complete = state.all_in_runout and street == 3
            child = PublicState(
                street=street, board=board,
                current=-1 if state.all_in_runout else 1 - self.button,
                spent=state.spent,
                history=state.history + (("board", card),),
                last_raise=self.big_blind,
                chance=state.all_in_runout and not complete,
                done=complete,
                all_in_runout=state.all_in_runout,
            )
            outcomes.append((child, probability))
        return tuple(outcomes)

    def terminal_payoff(self, state, hands):
        if not state.done or len(state.board) != 5 and state.winner == -1:
            raise ValueError("Payoff requires a fold or complete-board showdown")
        winner = state.winner
        if winner == -1:
            # Turn/river reveal order can differ while the seven-card holding
            # is identical; use one cache entry for either public-card order.
            board_key = tuple(sorted(state.board))
            score0 = _seven_strength(board_key, hands[0])
            score1 = _seven_strength(board_key, hands[1])
            winner = 0 if score0 > score1 else 1 if score1 > score0 else -1
        if winner == -1:
            return (state.spent[1] - state.spent[0]) / 2
        return (self.pot / 2 + state.spent[1] if winner == 0
                else -self.pot / 2 - state.spent[0])

    def chance_terminal_payoff(self, state, hands):
        """Exactly integrate an all-in runout with no remaining decisions."""
        if not state.chance or not state.all_in_runout or state.done:
            raise ValueError("Expected an all-in public chance node")
        dead = set(state.board + hands[0] + hands[1])
        remaining = tuple(card for card in DECK if card not in dead)
        missing = 5 - len(state.board)
        total, count = 0.0, 0
        for runout in combinations(remaining, missing):
            board_key = tuple(sorted(state.board + runout))
            score0 = _seven_strength(board_key, hands[0])
            score1 = _seven_strength(board_key, hands[1])
            if score0 > score1:
                total += self.pot / 2 + state.spent[1]
            elif score1 > score0:
                total -= self.pot / 2 + state.spent[0]
            else:
                total += (state.spent[1] - state.spent[0]) / 2
            count += 1
        return total / count

    @staticmethod
    def info(state, hands):
        return (state.current, hands[state.current], state.board, state.history)

    def expected_value(self, policy0, policy1, root_state=None):
        start = root_state or self.initial_state()
        return sum(
            mass * _river_profile_value(
                self, start, (hand0, hand1),
                lambda state, hands: policy0(state, hands)
                if state.current == 0 else policy1(state, hands),
            )
            for hand0, hand1, mass in self.deals
        )
