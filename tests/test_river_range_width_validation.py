import unittest

from experiments.river_range_width_validation import generate_requests


class RiverRangeWidthValidationTests(unittest.TestCase):
    def test_width_sweep_has_deterministic_blocker_valid_weighted_requests(self):
        first = generate_requests()
        second = generate_requests()
        self.assertEqual(len(first), 16)
        self.assertEqual([row["scenario_id"] for row in first],
                         [row["scenario_id"] for row in second])
        self.assertEqual([row["request"] for row in first],
                         [row["request"] for row in second])
        self.assertEqual({width: sum(row["range_width"] == width for row in first)
                          for width in (2, 3, 4, 6, 8)},
                         {2: 4, 3: 4, 4: 4, 6: 2, 8: 2})
        for row in first:
            request = row["request"]
            self.assertEqual(len(request.ranges[0].hands), row["range_width"])
            self.assertEqual(len(request.ranges[1].hands), row["range_width"])
            self.assertTrue(all(weight > 0 for range_ in request.ranges
                                for weight in range_.weights))
            dead = set(request.board)
            for range_ in request.ranges:
                for hand in range_.hands:
                    self.assertFalse(dead.intersection(hand))
                    dead.update(hand)
            actor = request.betting.current_player
            self.assertIn(request.acting_hand, request.ranges[actor].hands)


if __name__ == "__main__":
    unittest.main()
