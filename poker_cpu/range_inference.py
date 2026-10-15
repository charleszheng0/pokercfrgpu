"""Bayesian range updates from public events and supplied action likelihoods.

This module keeps beliefs coherent; it does not estimate action likelihoods.
The caller must supply a fitted and separately validated likelihood provider
before the result can be used for automatic advice.
"""

from dataclasses import dataclass

from .cards import parse_cards
from .ranges import canonical_hand
from .river_solver import WeightedRange


@dataclass(frozen=True)
class RangeEstimate:
    ranges: tuple[WeightedRange, WeightedRange]
    public_board: tuple[str, ...]
    action_count: int
    source: str
    validated_for_live_analysis: bool = False


def infer_ranges(events, initial_ranges, likelihood_provider, *,
                 perspective_player=None, known_hand=None, source="supplied_prior"):
    """Replay public board/action events in order and apply Bayes updates.

    likelihood_provider(event, hand, board, preceding_actions) must return
    P(observed action | hand, public context) in [0,1]. It is called for every
    remaining hand of the acting player. Missing probabilities are an error.
    Known private cards are used only for the perspective player's belief and
    to block impossible opponent hands; they are never read from game state.
    """
    if len(initial_ranges) != 2 or not all(isinstance(item, WeightedRange) for item in initial_ranges):
        raise ValueError("Two WeightedRange priors are required")
    if not callable(likelihood_provider):
        raise ValueError("An action-likelihood provider is required")
    beliefs = [WeightedRange(
        tuple(canonical_hand(hand) for hand in prior.hands), prior.weights
    ) for prior in initial_ranges]
    if perspective_player is None:
        if known_hand is not None:
            raise ValueError("known_hand requires perspective_player")
    elif perspective_player not in (0, 1):
        raise ValueError("perspective_player must be seat 0 or 1")
    else:
        hand = canonical_hand(known_hand)
        beliefs[perspective_player] = beliefs[perspective_player].condition(
            action_likelihoods={hand: 1.0}
        )
        beliefs[1 - perspective_player] = beliefs[1 - perspective_player].condition(
            dead_cards=hand
        )
    board = ()
    actions = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("History events must be dictionaries")
        kind = event.get("type")
        if kind == "board":
            new_board = parse_cards(event.get("cards", ()))
            expected_length = 3 if not board else len(board) + 1
            if len(new_board) != expected_length or new_board[:len(board)] != board:
                raise ValueError("Board events must reveal flop, turn, and river in order")
            board = new_board
            beliefs = [belief.condition(dead_cards=board) for belief in beliefs]
        elif kind == "action":
            seat = event.get("player")
            if seat not in (0, 1):
                raise ValueError("Action event needs a valid acting seat")
            likelihoods = {}
            for hand in beliefs[seat].hands:
                probability = likelihood_provider(event, hand, board, tuple(actions))
                if probability is None:
                    raise ValueError("Action likelihood missing for a possible hand")
                likelihoods[hand] = probability
            beliefs[seat] = beliefs[seat].condition(action_likelihoods=likelihoods)
            actions.append(event.copy())
        elif kind in ("deal", "blind", "uncalled_return", "award"):
            continue
        else:
            raise ValueError(f"Unsupported public event type: {kind}")
    return RangeEstimate(tuple(beliefs), board, len(actions), source)
