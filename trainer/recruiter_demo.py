"""Reproducible full-hand fixtures for a future validated recruiter demo.

These are real HoldemGame decks, not solved strategies. Never attach a CFR
response to one of these states unless its exact public state, ranges, and
action abstraction have a separately validated matching cache entry.
"""

from dataclasses import dataclass

from poker_cpu.cards import DECK
from poker_cpu.engine import HoldemGame


@dataclass(frozen=True)
class ShowcaseHand:
    name: str
    lesson: str
    button: int
    hero: tuple[str, str]
    opponent: tuple[str, str]
    board: tuple[str, str, str, str, str]
    burns: tuple[str, str, str]
    opponent_seed: int


HANDS = (
    ShowcaseHand(
        "Premium pair", "Value betting with a strong made hand", 0,
        ("Ah", "Ad"), ("Kc", "Kd"),
        ("As", "7c", "2d", "9h", "3s"), ("4c", "5c", "6c"), 1201,
    ),
    ShowcaseHand(
        "Drawing hand", "Deciding whether to continue with a flush draw", 1,
        ("Jh", "Th"), ("Qc", "Qd"),
        ("9h", "8h", "2s", "3c", "Ah"), ("4c", "5c", "6c"), 1202,
    ),
    ShowcaseHand(
        "River pressure", "Facing a changing board with a marginal hand", 0,
        ("Kc", "Qs"), ("Jd", "Td"),
        ("Qh", "7d", "2s", "9c", "8d"), ("4c", "5c", "6c"), 1203,
    ),
)


def ordered_deck(hand: ShowcaseHand):
    """Put private, burn, and public cards in the engine's actual deal order."""
    first, second = (hand.hero, hand.opponent) if hand.button == 0 else (hand.opponent, hand.hero)
    prefix = (
        first[0], second[0], first[1], second[1],
        hand.burns[0], *hand.board[:3],
        hand.burns[1], hand.board[3],
        hand.burns[2], hand.board[4],
    )
    if len(set(prefix)) != len(prefix) or not set(prefix) <= set(DECK):
        raise ValueError("Showcase cards must be unique members of the standard deck")
    return prefix + tuple(card for card in DECK if card not in set(prefix))


def new_game(index: int):
    if type(index) is not int or not 0 <= index < len(HANDS):
        raise ValueError("Unknown showcase hand")
    hand = HANDS[index]
    return HoldemGame(stacks=(20, 20), blinds=(1, 2), button=hand.button,
                      deck=ordered_deck(hand))
