import unittest

from poker_cpu.engine import Action
from poker_cpu.river_solver import RiverGame, RiverState, WeightedRange
from poker_cpu.solver_interface import (
    SolverRequest,
    TinyRiverCFRBackend,
    UnsupportedHoldemSolver,
)

BOARD = "As Kh Qd 7c 2s"
RANGES = (
    WeightedRange((("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")), (1, 1, 1)),
    WeightedRange((("Jd", "Td"), ("8h", "8s"), ("4c", "4d")), (1, 1, 1)),
)
HANDS = tuple(belief.hands for belief in RANGES)
WEIGHTS = tuple(belief.weights for belief in RANGES)


class SolverInterfaceTests(unittest.TestCase):
    def test_unsupported_solver_never_fabricates_advice(self):
        request = SolverRequest(
            public_state={"street": 1},
            acting_player=0,
            ranges=None,
            legal_action_abstraction=(),
            stacks=(100, 100),
            pot=4,
            betting_history=(),
            computation_budget_ms=100,
        )
        result = UnsupportedHoldemSolver().analyze(request)
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.validation_status, "not_validated")
        self.assertIsNone(result.action_frequencies)
        self.assertIsNone(result.action_evs)
        self.assertIsNone(result.ev_loss)
        self.assertIsNone(result.iterations)
        self.assertIsNone(result.latency_ms)
        self.assertIsNone(result.convergence)
        self.assertIsNone(result.confidence)

    def test_river_adapter_returns_only_validated_restricted_game_results(self):
        game = RiverGame(BOARD, HANDS, pot=8, stack=10, range_weights=WEIGHTS)
        request = SolverRequest(
            public_state={"street": "river", "board": BOARD},
            acting_player=0,
            ranges=RANGES,
            legal_action_abstraction=game.legal(RiverState()),
            stacks=(10, 12),
            pot=8,
            betting_history=(),
            computation_budget_ms=2,
            chosen_action=Action("check"),
            acting_hand=("Jc", "Tc"),
        )
        result = TinyRiverCFRBackend().analyze(request)
        self.assertEqual(result.status, "river_abstracted")
        self.assertEqual(result.validation_status, "tested_restricted_implementation")
        self.assertAlmostEqual(sum(result.action_frequencies), 1.0)
        self.assertEqual(len(result.action_evs), len(request.legal_action_abstraction))
        self.assertGreaterEqual(result.ev_loss, 0.0)
        self.assertEqual(result.ev_conditioning, "conditional_on_acting_hand")
        self.assertGreaterEqual(result.iterations, 1)
        self.assertGreater(result.latency_ms, 0.0)
        self.assertGreater(result.search_latency_ms, 0.0)
        self.assertLessEqual(result.search_latency_ms, result.latency_ms)
        self.assertIn("nash_conv_chips", result.convergence)

    def test_river_adapter_refuses_implicit_ranges_and_prior_river_actions(self):
        game = RiverGame(BOARD, HANDS, range_weights=WEIGHTS)
        legal = game.legal(RiverState())
        base = dict(
            public_state={"board": BOARD}, acting_player=0, ranges=RANGES,
            legal_action_abstraction=legal, stacks=(10, 10), pot=8,
            betting_history=(), computation_budget_ms=1,
        )
        backend = TinyRiverCFRBackend()
        no_ranges = backend.analyze(SolverRequest(**{**base, "ranges": None}))
        prior_action = backend.analyze(SolverRequest(**{
            **base, "betting_history": ((3, 1, "check", 0),)
        }))
        prior_streets = backend.analyze(SolverRequest(**{
            **base, "betting_history": ((0, 0, "call", 1), (1, 1, "check", 0))
        }))
        unknown_hand = backend.analyze(SolverRequest(**{
            **base, "acting_hand": ("As", "Ac")
        }))
        self.assertEqual(no_ranges.status, "unsupported")
        self.assertIsNone(no_ranges.action_frequencies)
        self.assertEqual(prior_action.status, "unsupported")
        self.assertEqual(prior_streets.status, "river_abstracted")
        self.assertEqual(unknown_hand.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
