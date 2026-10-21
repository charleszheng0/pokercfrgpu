import unittest

from experiments.postflop_validation_batch import generate_requests
from poker_cpu.solve import ActionAbstraction


class PostflopValidationBatchTests(unittest.TestCase):
    def test_batch_has_30_deterministic_supported_profile_inputs(self):
        first = generate_requests()
        again = generate_requests()
        self.assertEqual(len(first), 30)
        self.assertEqual([row["scenario_id"] for row in first],
                         [row["scenario_id"] for row in again])
        self.assertEqual([row["request"] for row in first],
                         [row["request"] for row in again])
        self.assertEqual(len({row["scenario_id"] for row in first}), 30)
        self.assertEqual(len({row["profile"] for row in first}), 5)
        for case in first:
            request = case["request"]
            dead = set(request.board)
            for belief in request.ranges:
                for hand in belief.hands:
                    self.assertFalse(dead.intersection(hand))
                    self.assertFalse(set(hand).intersection(dead))
                    dead.update(hand)
            if case["profile"] == "river_facing_bet":
                self.assertEqual(request.betting.current_player, 1)
                self.assertEqual(request.betting.street_actions[0].kind, "bet")

    def test_shark_menu_applies_to_full_tree_and_cutoffs_keep_fixed_tree(self):
        default = generate_requests()
        shark = generate_requests(shark_style=True)
        by_id_default = {row["scenario_id"]: row["request"] for row in default}
        by_id_shark = {row["scenario_id"]: row["request"] for row in shark}
        for scenario_id, request in by_id_shark.items():
            if request.config.continuation_mode == "full_tree":
                self.assertEqual(request.abstraction,
                                 ActionAbstraction.shark_style_postflop())
            else:
                self.assertEqual(request.abstraction, ActionAbstraction())
            if request.street == "river" and request.betting.street_actions:
                bet = request.betting.street_actions[0]
                self.assertIn(bet.amount, (2, 4, 5, 6))
        self.assertNotEqual(by_id_default["river_root_00"].abstraction,
                            by_id_shark["river_root_00"].abstraction)


if __name__ == "__main__":
    unittest.main()
