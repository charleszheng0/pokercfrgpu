"""Focused correctness tests for the custom Leduc/NFSP implementation."""

import random
import unittest
from collections import defaultdict

import torch
import torch.nn.functional as F

from small_games import compare_cfr_nfsp as reference
from small_games.dqn import ACTIONS
from small_games.encoder import INPUT_DIM, encode_state
from small_games.leduc import LeducPoker
from small_games.train_nfsp import NFSPAgent


def _reachable_states():
    states = set()

    def visit(state):
        if state in states:
            return
        states.add(state)
        if not state.finished:
            for child in reference.children(state).values():
                visit(child)

    for deal in reference.DEALS:
        visit(reference.initial(deal))
    return states


def _game_from_reference(state):
    """Construct an environment at the same public state as a reference state."""
    game = LeducPoker()
    game.players = list(state.cards[:2])
    game.public_card = state.cards[2] if state.round == 1 else None
    game.deck = [state.cards[2]]
    game.pot = state.pot
    game.contributions = list(state.paid)
    game.round_bets = list(state.round_paid)
    game.round = state.round
    game.current_player = state.current
    game.raises_this_round = state.raises
    game.done = state.finished
    game.history = list(state.history)
    game.actions_this_round = [
        action for round_id, _, action in state.history if round_id == state.round
    ]
    if state.folded != -1:
        game.winner = 1 - state.folded
    elif state.finished:
        game.winner = game.showdown()
    else:
        game.winner = None
    return game


def _state_signature(state):
    return (
        state.round,
        state.current,
        state.pot,
        state.paid,
        state.round_paid,
        state.raises,
        state.history,
        state.finished,
    )


def _game_signature(game):
    return (
        game.round,
        game.current_player,
        game.pot,
        tuple(game.contributions),
        tuple(game.round_bets),
        game.raises_this_round,
        tuple(game.history),
        game.done,
    )


def test_reference_rules_match_environment_exhaustively():
    states = _reachable_states()
    assert len(states) == 3864

    for state in states:
        game = _game_from_reference(state)
        if state.finished:
            assert game.rewards() == [
                reference.terminal_payoff(state, player) for player in range(2)
            ]
            continue

        assert tuple(game.legal_actions()) == reference.legal(state)
        for action, expected in reference.children(state).items():
            successor = _game_from_reference(state)
            successor.step(action)
            assert _game_signature(successor) == _state_signature(expected)


def test_encoder_is_an_information_set_encoding():
    by_info_set = {}
    by_encoding = defaultdict(set)

    for state in _reachable_states():
        if state.finished:
            continue
        info_set = reference.information_set(state, state.current)
        encoded = tuple(encode_state(reference.observation(state)).tolist())
        if info_set in by_info_set:
            assert by_info_set[info_set] == encoded
        by_info_set[info_set] = encoded
        by_encoding[encoded].add(info_set)

    assert len(by_info_set) == 528
    assert all(len(info_sets) == 1 for info_sets in by_encoding.values())


def test_dqn_loss_can_fit_a_terminal_target():
    random.seed(7)
    torch.manual_seed(7)
    agent = NFSPAgent(lr=0.01)
    state = torch.randn(INPUT_DIM)
    terminal = torch.zeros(INPUT_DIM)
    no_legal_actions = torch.zeros(len(ACTIONS), dtype=torch.bool)
    target = 3.0

    for _ in range(64):
        agent.rl_buffer.push(
            (
                state.clone(),
                1,
                target,
                terminal.clone(),
                True,
                no_legal_actions.clone(),
            )
        )

    for _ in range(100):
        agent.train_dqn(64)

    with torch.no_grad():
        fitted = agent.q_network(state)[1]
    assert F.smooth_l1_loss(fitted, torch.tensor(target)).item() < 1e-4


def test_average_policy_loss_can_fit_a_legal_action():
    random.seed(7)
    torch.manual_seed(7)
    agent = NFSPAgent(lr=0.01)
    state = torch.randn(INPUT_DIM)
    legal = torch.tensor([True, True, False, False, False])

    for _ in range(64):
        agent.sl_buffer.push((state.clone(), 1, legal.clone()))

    for _ in range(100):
        agent.train_average_policy(64)

    with torch.no_grad():
        logits = agent.policy_network(state).masked_fill(~legal, -1e9)
    assert torch.softmax(logits, dim=0)[1].item() > 0.999


def load_tests(loader, tests, pattern):
    """Expose the historical pytest-style checks to stdlib discovery too."""
    for name in (
        "test_reference_rules_match_environment_exhaustively",
        "test_encoder_is_an_information_set_encoding",
        "test_dqn_loss_can_fit_a_terminal_target",
        "test_average_policy_loss_can_fit_a_legal_action",
    ):
        tests.addTest(unittest.FunctionTestCase(globals()[name]))
    return tests
