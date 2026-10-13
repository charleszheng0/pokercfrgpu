"""Small, transparent seven-card evaluator; correctness before speed."""

from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)


def parse_cards(cards):
    result = tuple(cards.split()) if isinstance(cards, str) else tuple(cards)
    if len(result) != len(set(result)) or any(c not in DECK for c in result):
        raise ValueError("Cards must be distinct standard 52-card codes")
    return result


def _five(cards):
    ranks = [RANKS.index(c[0]) + 2 for c in cards]
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    straight = max((top for top in range(5, 15) if all(r in unique for r in range(top - 4, top + 1))), default=0)
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    if groups[0][0] == 2 and groups[1][0] == 2:
        high, low = sorted((groups[0][1], groups[1][1]), reverse=True)
        return (2, high, low, next(r for r in ranks if r != high and r != low))
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    return (0, *sorted(ranks, reverse=True))


def evaluate(cards):
    """Comparable tuple: category 0..8 followed by tie breakers."""
    cards = parse_cards(cards)
    if len(cards) not in (5, 6, 7):
        raise ValueError("Hand evaluation requires five to seven cards")
    return max(_five(hand) for hand in combinations(cards, 5))
