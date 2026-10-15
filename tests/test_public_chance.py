import unittest
from math import comb

from poker_cpu.cards import DECK
from poker_cpu.public_chance import public_transition
from poker_cpu.river_solver import WeightedRange


class PublicChanceTests(unittest.TestCase):
    def setUp(self):
        self.ranges = (
            WeightedRange((("Jc", "Tc"),), (1,)),
            WeightedRange((("Ac", "Ad"),), (1,)),
        )

    def test_flop_turn_and_river_chance_account_for_four_private_cards(self):
        flop = public_transition((), ("As", "Kh", "Qd"), self.ranges)
        self.assertAlmostEqual(flop.probability, 1 / comb(48, 3))
        self.assertEqual(flop.compatible_deals, 1)
        turn = public_transition(flop.board, ("7c",), flop.ranges)
        self.assertAlmostEqual(turn.probability, 1 / 45)
        river = public_transition(turn.board, ("2s",), turn.ranges)
        self.assertAlmostEqual(river.probability, 1 / 44)
        blocked = public_transition(flop.board, ("Jc",), flop.ranges)
        self.assertEqual(blocked.probability, 0)
        self.assertIsNone(blocked.ranges)

    def test_turn_distribution_sums_to_one_with_weighted_blockers(self):
        ranges = (
            WeightedRange((("Jc", "Tc"), ("9c", "8c")), (1, 3)),
            WeightedRange((("Ac", "Ad"), ("Jh", "Jd")), (2, 1)),
        )
        board = ("As", "Kh", "Qd")
        outcomes = [public_transition(board, (card,), ranges) for card in DECK if card not in board]
        self.assertAlmostEqual(sum(outcome.probability for outcome in outcomes), 1)
        self.assertEqual(public_transition(board, ("Ac",), ranges).compatible_deals, 2)
        with self.assertRaisesRegex(ValueError, "new three-card flop"):
            public_transition(board, ("As",), ranges)


if __name__ == "__main__":
    unittest.main()
