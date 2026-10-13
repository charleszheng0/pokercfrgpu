import unittest

from poker_cpu.engine import Action, HoldemGame
from poker_cpu.river_solver import RiverCFR, RiverGame, RiverState, WeightedRange, nash_conv


BOARD = "As Kh Qd 7c 2s"
RANGES = (
    (("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")),
    (("Jd", "Td"), ("8h", "8s"), ("4c", "4d")),
)


class RiverSolverTests(unittest.TestCase):
    def test_abstract_bets_match_main_engine_on_river(self):
        abstract = RiverGame(BOARD, RANGES, pot=8, stack=10)
        engine = HoldemGame(seed=1)
        engine.street = 3
        engine.board = list(abstract.board)
        engine.hole = [RANGES[0][0], RANGES[1][0]]
        engine.stacks = [10, 10]
        engine.pot = 8
        engine.paid = [4, 4]
        engine.street_paid = [0, 0]
        engine.current_player = 0
        engine.last_full_raise = 2
        self.assertEqual(tuple(engine.legal_actions()), abstract.legal(RiverState()))
        engine.step(Action("bet", 4))
        abstract_state = abstract.advance(RiverState(), Action("bet", 4))
        # The local solver intentionally removes a second raise from this tree.
        self.assertEqual(tuple(engine.legal_actions()), abstract.legal(abstract_state))

    def test_action_tree_and_payoff(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        start = RiverState()
        self.assertIn(Action("bet", 4), game.legal(start))
        after_bet = game.advance(start, Action("bet", 4))
        self.assertIn(Action("raise", 8), game.legal(after_bet))
        after_raise = game.advance(after_bet, Action("raise", 8))
        self.assertEqual(game.legal(after_raise), (Action("fold"), Action("call")))
        folded = game.advance(after_bet, Action("fold"))
        self.assertEqual(game.terminal_payoff(folded, (RANGES[0][0], RANGES[1][0])), 4)
        called = game.advance(after_bet, Action("call"))
        self.assertTrue(called.done)
        self.assertEqual(called.paid, (4, 4))

    def test_showdown_values_are_zero_sum_and_include_matched_bets(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        state = RiverState(paid=(4, 4), done=True, winner=-1)
        strong = ("Jc", "Tc")
        weak = ("4c", "4d")
        self.assertEqual(game.terminal_payoff(state, (strong, weak)), 8)
        self.assertEqual(game.terminal_payoff(state, (weak, strong)), -8)
        self.assertEqual(game.terminal_payoff(state, (strong, strong)), 0)

    def test_fold_payoff_is_zero_sum_with_unequal_contributions(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        state = RiverState(paid=(2, 6), done=True, winner=1)
        payoff = game.terminal_payoff(state, (RANGES[0][0], RANGES[1][0]))
        self.assertEqual(payoff, -6)

    def test_weighted_range_conditions_on_blockers_and_observed_actions(self):
        belief = WeightedRange(
            (("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")),
            (2, 1, 1),
        )
        posterior = belief.condition(
            dead_cards=("Jc",),
            action_likelihoods={("9h", "9s"): 0.25, ("3c", "3d"): 1.0},
        )
        self.assertEqual(posterior.hands, (("9h", "9s"), ("3c", "3d")))
        self.assertAlmostEqual(posterior.weights[0], 0.2)
        self.assertAlmostEqual(posterior.weights[1], 0.8)
        with self.assertRaises(ValueError):
            belief.condition(dead_cards=("Jc", "9h", "3c"))

    def test_ranges_hide_collisions_and_use_weights(self):
        game = RiverGame(BOARD, RANGES, range_weights=((2, 1, 1), (1, 1, 1)))
        self.assertAlmostEqual(sum(weight for _, _, weight in game.deals), 1)
        self.assertGreater(game.deals[0][2], game.deals[-1][2])
        with self.assertRaises(ValueError):
            RiverGame(BOARD, ((("As", "Td"),), RANGES[1]), pot=8)

    def test_cfr_decreases_exact_vulnerability(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        solver = RiverCFR(game).train(20)
        early = nash_conv(game, solver.average, solver.average)
        solver.train(80)
        late = nash_conv(game, solver.average, solver.average)
        self.assertLess(late, early)
        self.assertLess(late, 1.0)
        values = solver.root_hand_values()
        self.assertEqual(set(values[0]), set(RANGES[0]))
        self.assertGreater(values[0][RANGES[0][0]], values[0][RANGES[0][2]])


if __name__ == "__main__":
    unittest.main()
