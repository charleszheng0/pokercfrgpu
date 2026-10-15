"""Exact-runout one-street flop/turn CFR reference.

Players can bet on the current street using the river abstraction. If nobody
folds, later public cards run out uniformly and there is no later betting.
This is a deliberately limited finite game, not a multi-street Hold'em solve.
"""

from itertools import combinations

from .cards import DECK, evaluate, parse_cards
from .river_solver import RiverGame, WeightedRange


class OneStreetGame(RiverGame):
    def __init__(self, board, ranges, pot=8, stack=10, big_blind=2,
                 range_weights=None):
        self.board = parse_cards(board)
        if len(self.board) not in (3, 4):
            raise ValueError("One-street reference requires a flop or turn board")
        if (type(pot) is not int or type(stack) is not int or type(big_blind) is not int
                or pot < 0 or stack < 1 or big_blind < 1):
            raise ValueError("Pot, stack, and big blind must be valid integer chips")
        if len(ranges) != 2 or not all(ranges):
            raise ValueError("Both players need nonempty private-hand ranges")
        self.pot, self.stack, self.big_blind = pot, stack, big_blind
        if range_weights is None:
            range_weights = tuple((1.0,) * len(player_range) for player_range in ranges)
        if len(range_weights) != 2:
            raise ValueError("Range weights need one vector per player")
        self.range_beliefs = tuple(
            WeightedRange(tuple(player_range), tuple(range_weights[seat]))
            for seat, player_range in enumerate(ranges)
        )
        self.ranges = tuple(belief.hands for belief in self.range_beliefs)
        for player_range in self.ranges:
            if any(set(hand) & set(self.board) for hand in player_range):
                raise ValueError("Private hands must be outside the public board")
        joint = [
            (hand0, hand1, weight0 * weight1)
            for hand0, weight0 in zip(self.ranges[0], self.range_beliefs[0].weights)
            for hand1, weight1 in zip(self.ranges[1], self.range_beliefs[1].weights)
            if not set(hand0) & set(hand1) and weight0 * weight1 > 0
        ]
        total = sum(row[2] for row in joint)
        if total <= 0:
            raise ValueError("Ranges have no compatible joint private deals")
        self.deals = tuple((hand0, hand1, mass / total) for hand0, hand1, mass in joint)
        self.runout_probabilities = {
            (hand0, hand1): self._showdown_probabilities(hand0, hand1)
            for hand0, hand1, _ in self.deals
        }
        self.street = "flop" if len(self.board) == 3 else "turn"
        self.continuation_assumption = "no betting after this street"

    def _showdown_probabilities(self, hand0, hand1):
        remaining = [card for card in DECK if card not in self.board
                     and card not in hand0 and card not in hand1]
        wins = ties = losses = count = 0
        for runout in combinations(remaining, 5 - len(self.board)):
            score0 = evaluate(hand0 + self.board + runout)
            score1 = evaluate(hand1 + self.board + runout)
            wins += score0 > score1
            losses += score0 < score1
            ties += score0 == score1
            count += 1
        return wins / count, losses / count, ties / count

    def terminal_payoff(self, state, hands):
        if not state.done:
            raise ValueError("Payoff needs a terminal state")
        if state.winner != -1:
            return super().terminal_payoff(state, hands)
        win, loss, tie = self.runout_probabilities[hands]
        return (
            win * (self.pot / 2 + state.paid[1])
            - loss * (self.pot / 2 + state.paid[0])
            + tie * (state.paid[1] - state.paid[0]) / 2
        )
