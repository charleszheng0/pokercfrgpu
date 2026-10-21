import unittest

from experiments.profile_sizing_trees import build_request, tree_counts
from poker_cpu.solve import ActionAbstraction


class SizingTreeProfileTests(unittest.TestCase):
    def test_more_sizes_or_raise_depth_increase_the_finite_tree(self):
        baseline = tree_counts(build_request(ActionAbstraction()))
        no_minimum = tree_counts(build_request(ActionAbstraction(include_min_size=False)))
        dense = tree_counts(build_request(ActionAbstraction(
            bet_fractions=(0.25, 0.5, 0.75, 1.0),
            raise_fractions=(0.5, 1.0), include_min_size=True)))
        two_raises = tree_counts(build_request(ActionAbstraction(max_raises=2)))

        self.assertLess(no_minimum["public_decision_states"], baseline["public_decision_states"])
        self.assertGreater(dense["public_decision_states"], baseline["public_decision_states"])
        self.assertGreater(two_raises["public_decision_states"], baseline["public_decision_states"])
        for counts in (baseline, no_minimum, dense, two_raises):
            self.assertGreater(counts["terminal_histories"], 0)
            self.assertGreaterEqual(counts["public_action_edges"], counts["public_decision_states"])
            self.assertGreater(counts["information_sets"], 0)
            self.assertGreater(counts["information_set_action_slots"], 0)
            self.assertEqual(counts["compatible_joint_deals"], 4)

    def test_profile_reports_legal_root_menu_instead_of_a_synthetic_count(self):
        abstraction = ActionAbstraction(bet_fractions=(0.75,),
                                        raise_fractions=(0.75,),
                                        include_min_size=False)
        counts = tree_counts(build_request(abstraction))
        self.assertEqual(counts["root_actions"][0]["kind"], "check")
        self.assertTrue(all(row["kind"] == "bet" for row in counts["root_actions"][1:]))
        self.assertIn({"kind": "bet", "amount": 40}, counts["root_actions"])


if __name__ == "__main__":
    unittest.main()
