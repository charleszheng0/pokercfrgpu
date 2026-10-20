import unittest

from poker_cpu.cards import DECK
from poker_cpu.engine import Action, HoldemGame
from poker_cpu.rollout_feedback import estimate_action_evs


class RolloutFeedbackTests(unittest.TestCase):
    def test_preflop_estimate_uses_public_state_and_reports_net_ev(self):
        game = HoldemGame(button=0, seed=13)
        original_hidden = game.hole[1]
        original_deck = tuple(game.deck)
        first = estimate_action_evs(game, samples=3, seed=7, equity_samples=1)
        self.assertEqual(first["street"], "preflop")
        self.assertEqual(first["samples_per_action"], 3)
        fold = next(row for row in first["actions"] if row["action"]["kind"] == "fold")
        self.assertEqual(fold["ev_chips"], -1)
        self.assertEqual(game.hole[1], original_hidden)
        self.assertEqual(tuple(game.deck), original_deck)

        # Hidden cards in the real game cannot affect a public estimate.
        other = HoldemGame(button=0, seed=13)
        unseen = [card for card in DECK if card not in other.hole[0]]
        other.hole[1] = tuple(unseen[-2:])
        other.deck = list(reversed(unseen[:-2]))
        second = estimate_action_evs(other, samples=3, seed=7, equity_samples=1)
        self.assertEqual(first["actions"], second["actions"])

    def test_requires_a_hero_decision_and_bounded_sample_count(self):
        game = HoldemGame(button=1, seed=13)
        with self.assertRaisesRegex(ValueError, "hero decision"):
            estimate_action_evs(game, samples=2)
        game = HoldemGame(button=0, seed=13)
        with self.assertRaisesRegex(ValueError, "Samples"):
            estimate_action_evs(game, samples=0)

    def test_postflop_estimate_respects_public_board(self):
        game = HoldemGame(button=0, seed=21)
        game.step(Action("call"))
        game.step(Action("check"))
        game.step(Action("check"))  # Opponent acts first on the flop.
        self.assertEqual(game.street, 1)
        self.assertEqual(game.current_player, 0)
        report = estimate_action_evs(game, samples=2, seed=9, equity_samples=1)
        self.assertEqual(report["street"], "flop")
        self.assertEqual(len(game.board), 3)
        self.assertEqual(len(report["actions"]), len(game.legal_actions()))


if __name__ == "__main__":
    unittest.main()
