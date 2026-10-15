import unittest

from poker_cpu.range_inference import infer_ranges
from poker_cpu.ranges import canonical_hand, hand_class, preset_catalog, preset_range
from poker_cpu.river_solver import WeightedRange
from poker_cpu.engine import HoldemGame


class RangeModelTests(unittest.TestCase):
    def test_presets_expand_to_distinct_weighted_combos_and_respect_blockers(self):
        catalog = {item["id"]: item for item in preset_catalog()}
        self.assertEqual(len(catalog["button_open"]["classes"]), 169)
        self.assertIn("assumption", catalog["button_open"]["label"].lower())
        self.assertIn("assumption", catalog["big_blind_defend"]["label"].lower())
        self.assertGreater(len(preset_range("button_open").hands), 8)
        sample = preset_range("river_demo_opponent", dead_cards=("Ac",))
        self.assertEqual(len(sample.hands), 2)
        self.assertTrue(all("Ac" not in hand for hand in sample.hands))
        self.assertAlmostEqual(sum(sample.weights), 1)
        self.assertEqual(hand_class(("Tc", "Jc")), "JTs")
        self.assertEqual(canonical_hand(("Jh", "Jd")), ("Jd", "Jh"))

    def test_public_history_updates_action_likelihoods_and_card_removal(self):
        priors = (
            WeightedRange((("Jc", "Tc"), ("9c", "8c")), (1, 1)),
            WeightedRange((("Ac", "Ad"), ("Jh", "Jd"), ("As", "8h")), (1, 1, 1)),
        )
        events = [
            {"type": "deal", "button": 0},
            {"type": "action", "street": 0, "player": 0, "kind": "raise", "amount": 6},
            {"type": "action", "street": 0, "player": 1, "kind": "call", "amount": 0},
            {"type": "board", "street": 1, "cards": ("As", "Kh", "Qd")},
            {"type": "board", "street": 2, "cards": ("As", "Kh", "Qd", "7c")},
            {"type": "board", "street": 3, "cards": ("As", "Kh", "Qd", "7c", "2s")},
        ]

        def likelihood(event, hand, board, preceding):
            self.assertEqual(len(board), 0)
            if event["player"] == 0:
                return 0.6
            self.assertEqual(len(preceding), 1)
            return 0.2 if hand == ("Ac", "Ad") else 0.8

        estimate = infer_ranges(
            events, priors, likelihood, perspective_player=0,
            known_hand=("Tc", "Jc"), source="test_assumption",
        )
        self.assertEqual(estimate.ranges[0].hands, (("Jc", "Tc"),))
        self.assertEqual(estimate.ranges[1].hands, (("Ac", "Ad"), ("Jd", "Jh")))
        self.assertAlmostEqual(estimate.ranges[1].weights[0], 0.2)
        self.assertAlmostEqual(estimate.ranges[1].weights[1], 0.8)
        self.assertEqual(len(estimate.public_board), 5)
        self.assertEqual(estimate.action_count, 2)
        self.assertFalse(estimate.validated_for_live_analysis)

    def test_missing_or_impossible_likelihood_does_not_invent_posterior(self):
        priors = (
            WeightedRange((("Jc", "Tc"),), (1,)),
            WeightedRange((("Ac", "Ad"),), (1,)),
        )
        events = ({"type": "action", "street": 0, "player": 1, "kind": "call", "amount": 0},)
        with self.assertRaisesRegex(ValueError, "missing"):
            infer_ranges(events, priors, lambda *args: None)
        with self.assertRaisesRegex(ValueError, "no possible private hands"):
            infer_ranges(events, priors, lambda *args: 0)
        with self.assertRaisesRegex(ValueError, "required"):
            infer_ranges(events, priors, None)

    def test_public_event_pipeline_accepts_real_engine_history(self):
        game = HoldemGame(seed=29)
        while not game.done:
            legal = game.legal_actions()
            action = next((candidate for candidate in legal if candidate.kind == "call"), None)
            if action is None:
                action = next(candidate for candidate in legal if candidate.kind == "check")
            game.step(action)
        hero = canonical_hand(game.hole[0])
        opponent = canonical_hand(game.hole[1])
        priors = (WeightedRange((hero,), (1,)), WeightedRange((opponent,), (1,)))
        estimate = infer_ranges(game.events, priors, lambda *_: 0.5,
                                perspective_player=0, known_hand=hero)
        self.assertEqual(estimate.public_board, tuple(game.board))
        self.assertEqual(estimate.action_count, len(game.history))
        self.assertEqual(estimate.ranges[0].hands, (hero,))
        self.assertEqual(estimate.ranges[1].hands, (opponent,))
        self.assertFalse(estimate.validated_for_live_analysis)


if __name__ == "__main__":
    unittest.main()
