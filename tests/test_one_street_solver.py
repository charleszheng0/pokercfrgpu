import math
import unittest

from poker_cpu.cards import DECK
from poker_cpu.engine import Action
from poker_cpu.one_street_solver import OneStreetGame
from poker_cpu.river_solver import RiverCFR, RiverGame, RiverState


class OneStreetSolverTests(unittest.TestCase):
    def test_turn_exact_runout_payoff_and_betting_tree(self):
        board = "As Kh Qd 7c"
        ranges = ((("Jc", "Tc"),), (("9h", "9s"),))
        game = OneStreetGame(board, ranges, pot=8, stack=10)
        river = RiverGame("As Kh Qd 7c 2s", ranges, pot=8, stack=10)
        self.assertEqual(game.legal(RiverState()), river.legal(RiverState()))
        probabilities = game.runout_probabilities[(ranges[0][0], ranges[1][0])]
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertEqual(probabilities, (1.0, 0.0, 0.0))
        checked = game.advance(game.advance(RiverState(), Action("check")), Action("check"))
        self.assertEqual(game.terminal_payoff(checked, (ranges[0][0], ranges[1][0])), 4)
        bet = game.advance(RiverState(), Action("bet", 4))
        called = game.advance(bet, Action("call"))
        self.assertEqual(game.terminal_payoff(called, (ranges[0][0], ranges[1][0])), 8)
        solver = RiverCFR(game, variant="cfr+").train(30)
        report = solver.analyze_root(private_hand=ranges[0][0])
        self.assertEqual(report["status"], "turn_one_street_abstracted")
        self.assertTrue(math.isfinite(report["nash_conv"]))
        self.assertAlmostEqual(sum(report["per_hand"][0]["frequency"].values()), 1)

    def test_flop_runout_probabilities_and_unsupported_preflop(self):
        ranges = ((("Jc", "Tc"),), (("9h", "9s"),))
        game = OneStreetGame("As Kh Qd", ranges, pot=8, stack=10)
        win, loss, tie = game.runout_probabilities[(ranges[0][0], ranges[1][0])]
        self.assertAlmostEqual(win + loss + tie, 1)
        self.assertGreater(loss, 0)
        self.assertEqual(game.continuation_assumption, "no betting after this street")
        with self.assertRaisesRegex(ValueError, "flop or turn"):
            OneStreetGame((), ranges)

    def test_turn_terminal_value_matches_average_of_river_showdowns(self):
        board = ("As", "Kh", "Qd", "7c")
        hero, opponent = ("8h", "8s"), ("9h", "9s")
        game = OneStreetGame(board, ((hero,), (opponent,)), pot=8, stack=10)
        called = game.advance(game.advance(RiverState(), Action("bet", 4)), Action("call"))
        actual = game.terminal_payoff(called, (hero, opponent))
        remaining = [card for card in DECK if card not in board + hero + opponent]
        expected = sum(
            RiverGame(board + (card,), ((hero,), (opponent,)), pot=16, stack=6)
            .terminal_payoff(RiverState(done=True), (hero, opponent))
            for card in remaining
        ) / len(remaining)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_turn_solver_can_start_after_public_bet(self):
        hero, opponent = ("Jc", "Tc"), ("9h", "9s")
        game = OneStreetGame("As Kh Qd 7c", ((hero,), (opponent,)), pot=8, stack=10)
        root = game.advance(RiverState(current=1), Action("bet", 4))
        solver = RiverCFR(game, variant="cfr+", root_player=0, root_state=root).train(30)
        report = solver.analyze_root(player=0, private_hand=hero)
        self.assertEqual({action.kind for action in report["actions"]}, {"fold", "call", "raise"})
        self.assertAlmostEqual(sum(report["per_hand"][0]["frequency"].values()), 1)
        self.assertTrue(math.isfinite(report["nash_conv"]))


if __name__ == "__main__":
    unittest.main()
