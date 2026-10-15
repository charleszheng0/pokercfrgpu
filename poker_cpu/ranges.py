"""Editable combo ranges and illustrative preflop priors.

Preset frequencies are design assumptions, not solved or measured strategies.
They are separate from the river CFR solver and from betting-history likelihoods.
"""

from itertools import combinations

from .cards import DECK, RANKS, SUITS, parse_cards
from .river_solver import WeightedRange


def canonical_hand(cards):
    cards = parse_cards(cards)
    if len(cards) != 2:
        raise ValueError("A private hand needs exactly two cards")
    return tuple(sorted(cards, key=lambda card: (-RANKS.index(card[0]), SUITS.index(card[1]))))


def hand_class(cards):
    a, b = canonical_hand(cards)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def _illustrative_frequency(name, hand):
    a, b = hand
    high = RANKS.index(a[0]) + 2
    low = RANKS.index(b[0]) + 2
    pair = high == low
    suited = a[1] == b[1]
    gap = high - low
    if name == "button_open":
        if pair or high == 14 or (high >= 10 and low >= 10):
            return 1.0
        if suited and (high >= 11 or gap <= 2 and high >= 7):
            return 0.75
        if (high >= 11 and low >= 8) or (suited and low >= 5):
            return 0.5
        if gap <= 2 and high >= 8:
            return 0.25
        return 0.0
    if name == "big_blind_defend":
        if pair or (high == 14 and suited) or (high >= 11 and low >= 10):
            return 1.0
        if high == 14 or (suited and (high >= 10 or gap <= 2 and high >= 6)):
            return 0.75
        if (high >= 10 and low >= 7) or (suited and low >= 4):
            return 0.5
        if suited or (gap <= 2 and high >= 7):
            return 0.25
        return 0.0
    raise ValueError("Unknown illustrative range")


def preset_catalog():
    """Return UI definitions; class weights are per-combo inclusion frequencies."""
    representative = {}
    for combo in combinations(DECK, 2):
        hand = canonical_hand(combo)
        representative.setdefault(hand_class(hand), hand)
    broad = {
        name: {label: _illustrative_frequency(name, hand)
               for label, hand in representative.items()}
        for name in ("button_open", "big_blind_defend")
    }
    return [
        {
            "id": "river_demo_hero",
            "label": "River example · hero",
            "description": "Three illustrative combos for trying the restricted river solver; no betting history was used.",
            "combos": {"Jc Tc": 1.0, "9c 8c": 1.0, "4c 4d": 1.0},
        },
        {
            "id": "river_demo_opponent",
            "label": "River example · opponent",
            "description": "Three illustrative opponent combos with relative frequencies; no betting history was used.",
            "combos": {"Ac Ad": 0.5, "Jd Jh": 1.0, "8h 8s": 0.5},
        },
        {
            "id": "turn_mechanics_hero",
            "label": "Turn mechanics · hero",
            "description": "One illustrative combo for a fast multi-street mechanics check. A one-combo range is not realistic opponent modeling.",
            "combos": {"Jc Tc": 1.0},
        },
        {
            "id": "turn_mechanics_opponent",
            "label": "Turn mechanics · opponent",
            "description": "One illustrative combo for testing turn-to-river search. This is an assumed known hand, not an inferred range.",
            "combos": {"9h 9s": 1.0},
        },
        {
            "id": "button_open",
            "label": "Button opening range · assumption",
            "description": "Illustrative preflop open frequencies at 100 BB, not a measured human range or solved chart. Narrow it before river solving.",
            "classes": broad["button_open"],
        },
        {
            "id": "big_blind_defend",
            "label": "Big blind defending range · assumption",
            "description": "Illustrative preflop defense frequencies at 100 BB, not a measured human range or solved chart. Narrow it before river solving.",
            "classes": broad["big_blind_defend"],
        },
    ]


def preset_range(preset_id, dead_cards=()):
    preset = next((item for item in preset_catalog() if item["id"] == preset_id), None)
    if preset is None:
        raise ValueError("Unknown range preset")
    dead = set(parse_cards(dead_cards))
    hands, weights = [], []
    for raw in combinations(DECK, 2):
        hand = canonical_hand(raw)
        if dead.intersection(hand):
            continue
        key = " ".join(hand)
        weight = preset.get("combos", {}).get(key, preset.get("classes", {}).get(hand_class(hand), 0.0))
        if weight > 0:
            hands.append(hand)
            weights.append(weight)
    return WeightedRange(tuple(hands), tuple(weights))
