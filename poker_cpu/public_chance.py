"""Exact public-card probabilities for small factorized private-hand ranges.

This is chance/range plumbing for future turn and flop search, not a CFR
solver for those streets. Burn cards are unobserved and integrate out.
"""

from dataclasses import dataclass
from math import comb

from .cards import parse_cards
from .river_solver import WeightedRange


@dataclass(frozen=True)
class PublicTransition:
    board: tuple
    probability: float
    ranges: tuple | None
    compatible_deals: int


def public_transition(board, revealed, ranges):
    """Condition two ranges on an exact flop, turn, or river reveal.

    The probability is for an unordered three-card flop or one specific next
    street card, averaged over compatible joint private deals. Both players'
    factors are updated after observing the new public cards; the downstream
    game must still remove private-hand collisions when constructing deals.
    """
    board = parse_cards(board)
    revealed = parse_cards(revealed)
    if len(board) not in (0, 3, 4):
        raise ValueError("Board must be preflop, flop, or turn")
    expected = 3 if not board else 1
    if len(revealed) != expected or set(board) & set(revealed):
        raise ValueError("Reveal must be a new three-card flop or one new street card")
    if len(ranges) != 2 or not all(isinstance(item, WeightedRange) for item in ranges):
        raise ValueError("Two weighted private-hand ranges are required")
    beliefs = tuple(item.condition(dead_cards=board) for item in ranges)
    total = 0.0
    compatible = 0.0
    compatible_count = 0
    blocked = set(revealed)
    for hand0, weight0 in zip(beliefs[0].hands, beliefs[0].weights):
        for hand1, weight1 in zip(beliefs[1].hands, beliefs[1].weights):
            if set(hand0) & set(hand1):
                continue
            mass = weight0 * weight1
            total += mass
            if not blocked.intersection(hand0) and not blocked.intersection(hand1):
                compatible += mass
                compatible_count += 1
    if total <= 0:
        raise ValueError("Ranges have no compatible joint private deals")
    chance_denominator = comb(52 - len(board) - 4, len(revealed))
    probability = compatible / total / chance_denominator
    if compatible <= 0:
        return PublicTransition(board + revealed, 0.0, None, 0)
    posterior = tuple(item.condition(dead_cards=board + revealed) for item in beliefs)
    return PublicTransition(board + revealed, probability, posterior, compatible_count)
