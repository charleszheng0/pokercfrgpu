import math
import unittest

from experiments.river_range_scaling import (
    compatible_joint_deals,
    concrete_ranges,
    run_condition,
)


class RiverRangeScalingHarnessTests(unittest.TestCase):
    def test_combo_generation_is_seeded_and_blocker_count_matches(self):
        ranges_a = concrete_ranges(4, seed=17)
        ranges_b = concrete_ranges(4, seed=17)
        self.assertEqual(ranges_a, ranges_b)
        self.assertEqual(len(ranges_a[0]), 4)
        self.assertEqual(len(ranges_a[1]), 4)
        self.assertTrue(all(len(set(hand)) == 2 for seat in ranges_a for hand in seat))
        self.assertEqual(compatible_joint_deals(ranges_a),
                         compatible_joint_deals(ranges_b))

    def test_one_combo_smoke_uses_shared_api_and_records_real_solver_outputs(self):
        row = run_condition(1, seed=17, iterations=1)
        self.assertEqual(row["mode"], "shared_api")
        self.assertIsNone(row["error"])
        self.assertEqual(row["status"], "supported_restricted_game")
        self.assertEqual(row["compatible_joint_deals"], 1)
        self.assertEqual(row["iterations"], 1)
        self.assertTrue(math.isfinite(row["nash_conv_chips"]))
        self.assertGreaterEqual(row["latency_ms"], row["search_latency_ms"])
        self.assertAlmostEqual(
            sum(item["frequency"] for item in row["aggregate_action_frequencies"]),
            1.0,
        )
        self.assertEqual(
            [item["kind"] for item in row["aggregate_action_frequencies"]],
            [item["kind"] for item in row["aggregate_conditional_action_evs_chips"]],
        )


if __name__ == "__main__":
    unittest.main()
