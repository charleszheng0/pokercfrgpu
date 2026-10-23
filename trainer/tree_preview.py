"""Cheap public betting-tree preview for Scenario Lab.

No CFR or private-range inference happens here. The same MultiStreetGame.legal
and advance methods used by the solver generate every displayed wager target.
"""

from poker_cpu.cards import DECK, parse_cards
from poker_cpu.engine import Action
from poker_cpu.multi_street import MultiStreetGame
from poker_cpu.solve import ActionAbstraction


def _action_json(action):
    return {"kind": action.kind, "amount": action.amount}


def _current_street_size(game, first_actor):
    """Count public decisions on the current street, stopping at chance nodes."""
    seen = set()
    edges = 0

    def visit(state):
        nonlocal edges
        if state.done or state.chance or state in seen:
            return
        seen.add(state)
        actions = game.legal(state)
        edges += len(actions)
        for action in actions:
            visit(game.advance(state, action))

    visit(game.initial_state(first_actor))
    return len(seen), edges


def preview_betting_tree(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be an object")
    board = parse_cards(payload.get("board", ""))
    if len(board) not in (3, 4, 5):
        raise ValueError("Enter a complete flop, turn, or river board")
    pot = payload.get("pot")
    stacks = payload.get("stacks")
    if type(pot) is not int or pot < 0:
        raise ValueError("Pot must be a nonnegative integer")
    if (not isinstance(stacks, list) or len(stacks) != 2
            or any(type(stack) is not int or stack < 1 for stack in stacks)):
        raise ValueError("Enter two positive integer stacks")
    tree = payload.get("betting_tree", "default")
    if tree not in ("default", "shark_2_0_inspired"):
        raise ValueError("Unknown betting tree")
    spot = payload.get("spot", "opening")
    if spot not in ("opening", "checked_to_hero", "facing_bet",
                    "facing_bet_first", "facing_raise"):
        raise ValueError("Unknown decision sequence")
    abstraction = (ActionAbstraction.shark_style_postflop()
                   if tree == "shark_2_0_inspired" else ActionAbstraction())
    # Public betting legality does not depend on which unblocked private hands
    # are held. These dummy hands only satisfy MultiStreetGame's constructor.
    available = [card for card in DECK if card not in board]
    game = MultiStreetGame(
        board, ((tuple(available[:2]),), (tuple(available[2:4]),)),
        pot=pot, stack=min(stacks), button=0, big_blind=2,
        bet_fractions=abstraction.bet_fractions,
        raise_fractions=abstraction.raise_fractions,
        bet_fractions_by_street=dict(abstraction.bet_fractions_by_street),
        raise_fractions_by_street=dict(abstraction.raise_fractions_by_street),
        max_raises=abstraction.max_raises,
        include_min_size=abstraction.include_min_size,
    )
    first_actor = 1 if spot in ("checked_to_hero", "facing_bet_first") else 0
    state = game.initial_state(first_actor)
    history = []

    def advance(action):
        nonlocal state
        if action not in game.legal(state):
            raise ValueError("Decision sequence is outside this betting tree")
        state = game.advance(state, action)
        history.append(_action_json(action))

    if spot in ("checked_to_hero", "facing_bet"):
        advance(Action("check"))
    bet_targets = [action.amount for action in game.legal(state)
                   if action.kind == "bet"]
    if spot == "facing_raise":
        bet_targets = [target for target in bet_targets
                       if any(action.kind == "raise" for action in game.legal(
                           game.advance(state, Action("bet", target))))]
    selected_bet = payload.get("prior_bet")
    if spot in ("facing_bet", "facing_bet_first", "facing_raise"):
        if not bet_targets:
            raise ValueError("No bet is available at this stack size")
        if selected_bet not in bet_targets:
            selected_bet = bet_targets[0]
        advance(Action("bet", selected_bet))
    raise_targets = [action.amount for action in game.legal(state)
                     if action.kind == "raise"]
    selected_raise = payload.get("prior_raise")
    if spot == "facing_raise":
        if not raise_targets:
            raise ValueError("No raise is available after this bet")
        if selected_raise not in raise_targets:
            selected_raise = raise_targets[0]
        advance(Action("raise", selected_raise))
    if state.done or state.chance or state.current != 0:
        raise ValueError("This sequence does not end at your decision")
    public_decisions, public_edges = _current_street_size(game, first_actor)
    return {
        "street": game.street,
        "betting_tree": tree,
        "spot": spot,
        "first_actor": first_actor,
        "prior_bet_targets": bet_targets,
        "prior_raise_targets": raise_targets,
        "selected_prior_bet": selected_bet,
        "selected_prior_raise": selected_raise,
        "history": history,
        "paid": state.paid,
        "pot_at_decision": pot + sum(state.spent),
        "legal_actions": [_action_json(action) for action in game.legal(state)],
        "max_raises": abstraction.max_raises,
        "current_street_public_decisions": public_decisions,
        "current_street_public_action_edges": public_edges,
    }
