import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainer import scenario_db
from trainer.server import analyze_river_position


class ScenarioDatabaseTests(unittest.TestCase):
    def test_exact_versioned_quality_gate_and_candidate_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "solves.sqlite3"
            request = {"street": "river", "board": "As Kh Qd 7c 2s",
                       "ranges": [{"hands": [["Jc", "Tc"]], "weights": [1.0]}],
                       "iterations": 500}
            result = {"status": "restricted_river_analysis", "nash_conv": 0.05,
                      "action_evs": [{"ev": 1.25}]}
            self.assertTrue(scenario_db.record(request, result, path=path))
            self.assertEqual(scenario_db.lookup(request, path=path), result)
            self.assertIsNone(scenario_db.lookup({**request, "iterations": 1000}, path=path))
            self.assertIsNone(scenario_db.lookup({**request, "board": "As Kh Qd 7c 3s"}, path=path))
            self.assertIsNone(scenario_db.lookup(request, quality_ceiling_chips=0.01, path=path))
            with patch("trainer.scenario_db.solver_fingerprint", return_value="changed-source"):
                self.assertIsNone(scenario_db.lookup(request, path=path))
            candidate = {**request, "iterations": 20}
            self.assertFalse(scenario_db.record(candidate, {**result, "nash_conv": 0.2}, path=path))
            self.assertIsNone(scenario_db.lookup(candidate, path=path))
            self.assertEqual(scenario_db.stats(path), {"river": {"recorded": 2, "qualified": 1}})

    def test_scenario_lab_returns_calculated_result_then_database_hit(self):
        with tempfile.TemporaryDirectory() as temp:
            request = {
                "street": "river", "board": "As Kh Qd 7c 2s", "hero_hand": "Jc Tc",
                "pot": 8, "stacks": [10, 10], "iterations": 100,
                "ranges": [
                    {"hands": [["Jc", "Tc"]], "weights": [1.0]},
                    {"hands": [["9h", "9s"]], "weights": [1.0]},
                ],
            }
            path = Path(temp) / "solves.sqlite3"
            with patch.dict("os.environ", {"POKER_SCENARIO_DB": str(path)}), \
                    patch("trainer.cfr_scenario_worker.get_qualified_scenario", return_value=None):
                first = analyze_river_position(request, use_database=True)
                self.assertEqual(first["cache_status"], "fresh_qualified_database_record")
                with patch("trainer.server.solve", side_effect=AssertionError("must use exact cache")):
                    second = analyze_river_position(request, use_database=True)
                self.assertEqual(second["cache_status"], "exact_database_hit")
                self.assertEqual(second["frequencies"], first["frequencies"])
                self.assertEqual(second["action_evs"], first["action_evs"])
                self.assertEqual(second["nash_conv"], first["nash_conv"])


if __name__ == "__main__":
    unittest.main()
