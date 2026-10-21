import math
import unittest

from poker_cpu.solve import solve
from poker_cpu.tree_profiles import supported_tree_profiles


class RestrictedTreeProfilesTests(unittest.TestCase):
    def test_declared_tree_profiles_return_finite_normalized_strategy_and_ev(self):
        profiles = supported_tree_profiles()
        self.assertEqual(len(profiles), 7)
        self.assertEqual(len({row.profile_id for row in profiles}), len(profiles))
        for profile in profiles:
            with self.subTest(profile=profile.profile_id):
                result = solve(profile.request)
                self.assertEqual(result.status, "supported_restricted_game", result.reason)
                self.assertGreater(result.iterations, 0)
                self.assertEqual(result.joint_deals, 1)
                self.assertTrue(math.isfinite(result.nash_conv_chips))
                self.assertGreaterEqual(result.nash_conv_chips, 0.0)
                if profile.profile_id not in ("flop_full_tree_experimental",
                                               "flop_full_tree_sampled"):
                    self.assertLessEqual(result.nash_conv_chips, 0.1)
                self.assertGreaterEqual(result.latency_ms, 0.0)
                self.assertTrue(result.actions)
                self.assertTrue(result.per_hand)
                for hand in result.per_hand:
                    self.assertAlmostEqual(sum(item.frequency for item in hand.actions), 1.0)
                    self.assertEqual({item.action for item in hand.actions}, set(result.actions))
                    for item in hand.actions:
                        self.assertTrue(math.isfinite(item.ev_chips))
                        self.assertGreaterEqual(item.ev_loss_chips, -1e-10)
                self.assertAlmostEqual(
                    sum(item.frequency for item in result.aggregate_actions), 1.0)

    def test_profile_catalog_labels_experimental_and_cutoff_boundaries(self):
        profiles = {profile.profile_id: profile for profile in supported_tree_profiles()}
        self.assertTrue(profiles["flop_full_tree_experimental"].request.config.allow_experimental_flop)
        self.assertEqual(profiles["flop_showdown_cutoff"].request.config.continuation_mode,
                         "showdown_cutoff")
        self.assertIn("no later betting", profiles["flop_showdown_cutoff"].note)
        self.assertIn("fixed half-pot/pot/all-in", profiles["turn_showdown_cutoff"].note)
        result = solve(profiles["flop_full_tree_experimental"].request)
        self.assertEqual(result.validation_status, "experimental_full_flop_residual_measured")
        self.assertIn("not validated for product advice", result.reason)
        sampled = profiles["flop_full_tree_sampled"].request.config
        self.assertEqual(sampled.variant, "cfr")
        self.assertTrue(sampled.chance_sampling)
        self.assertEqual(sampled.sampling_seed, 7)


if __name__ == "__main__":
    unittest.main()
