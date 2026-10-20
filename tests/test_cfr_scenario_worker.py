import json
import tempfile
import unittest
from pathlib import Path

from poker_cpu.cards import parse_cards
from poker_cpu.ranges import canonical_hand
from trainer.cfr_scenario_worker import make_request, run_loop, solve_one


class CFRScenarioWorkerTests(unittest.TestCase):
    def test_scenario_sampling_is_reproducible_and_card_legal(self):
        first = make_request(701, iterations=20)
        second = make_request(701, iterations=20)
        self.assertEqual(first, second)
        board = parse_cards(first["board"])
        hands = [canonical_hand(hand) for seat in first["ranges"] for hand in seat["hands"]]
        self.assertEqual(len(set(board + tuple(card for hand in hands for card in hand))), 13)
        self.assertEqual(len(first["ranges"][0]["hands"]), 2)
        self.assertEqual(len(first["ranges"][1]["hands"]), 2)
        self.assertEqual(first["range_source"], "offline_sampled_range")
        self.assertEqual(first["first_actor"], 0)

    def test_worker_persists_real_solver_output_and_only_qualified_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = solve_one(702, iterations=20, threshold_chips=1e9,
                            store=root / "candidates", cache=root / "cache")
            stored = json.loads((root / "candidates" / f"{row['scenario_id']}.json").read_text())
            cached = json.loads((root / "cache" / f"{row['scenario_id']}.json").read_text())
        self.assertEqual(row["status"], "cached_qualified")
        self.assertEqual(stored["result"]["nash_conv"], cached["result"]["nash_conv"])
        self.assertEqual(row["joint_deals"], 4)
        self.assertEqual(row["iterations"], 20)
        self.assertGreaterEqual(row["latency_seconds"], 0)
        self.assertEqual(row["result"]["range_source"], "offline_sampled_range")

    def test_bounded_worker_finishes_and_records_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status = run_loop(duration_seconds=60, pause_seconds=0, max_solves=1,
                              iterations=20, threshold_chips=1e9, seed=703,
                              store=root / "store", cache=root / "cache")
            history = json.loads((root / "store" / "history.json").read_text())
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(status["completed"], 1)
        self.assertEqual(status["qualified"], 1)
        self.assertEqual(history[0]["status"], "cached_qualified")


if __name__ == "__main__":
    unittest.main()
