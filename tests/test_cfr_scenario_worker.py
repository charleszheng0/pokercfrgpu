import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from poker_cpu.cards import parse_cards
from poker_cpu.ranges import canonical_hand
from trainer.cfr_scenario_worker import (
    get_qualified_scenario, make_request, request_key, run_loop,
    scenario_key, solve_one, solver_fingerprint,
)


class CFRScenarioWorkerTests(unittest.TestCase):
    def test_scenario_sampling_is_reproducible_and_card_legal(self):
        first = make_request(701, iterations=20)
        second = make_request(701, iterations=20)
        self.assertEqual(first, second)
        board = parse_cards(first["board"])
        hands = [canonical_hand(hand) for seat in first["ranges"] for hand in seat["hands"]]
        self.assertEqual(len(set(board + tuple(card for hand in hands for card in hand))), 21)
        self.assertEqual(len(first["ranges"][0]["hands"]), 4)
        self.assertEqual(len(first["ranges"][1]["hands"]), 4)
        self.assertEqual(first["range_source"], "offline_sampled_range")
        self.assertEqual(first["first_actor"], 0)

    def test_worker_persists_real_solver_output_and_only_qualified_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = solve_one(702, iterations=20, combos_per_range=2, threshold_chips=1e9,
                            store=root / "candidates", cache=root / "cache")
            stored = json.loads((root / "candidates" / f"{row['solve_id']}.json").read_text())
            cached = json.loads((root / "cache" / f"{row['solve_id']}.json").read_text())
            cache_hit = get_qualified_scenario(
                row["request"], cache=root / "cache", max_nash_conv_chips=1e9)
        self.assertEqual(row["status"], "cached_qualified")
        self.assertEqual(stored["result"]["nash_conv"], cached["result"]["nash_conv"])
        self.assertEqual(row["joint_deals"], 4)
        self.assertEqual(row["iterations"], 20)
        self.assertGreaterEqual(row["latency_seconds"], 0)
        self.assertEqual(row["result"]["range_source"], "offline_sampled_range")
        self.assertEqual(row["solver_fingerprint"], solver_fingerprint())
        self.assertIsNotNone(cache_hit)
        budget_changed = {**row["request"], "iterations": 40}
        self.assertEqual(scenario_key(row["request"]), scenario_key(budget_changed))
        self.assertNotEqual(request_key(row["request"]), request_key(budget_changed))

    def test_cache_lookup_rejects_stale_or_nonqualified_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            request = make_request(704, iterations=20, combos_per_range=2)
            fingerprint = solver_fingerprint()
            solve_id = request_key(request, fingerprint)
            good = {
                "status": "cached_qualified", "solve_id": solve_id,
                "solver_fingerprint": fingerprint, "request": request,
                "nash_conv_chips": 0.05, "result": {},
            }
            path = cache / f"{solve_id}.json"
            path.write_text(json.dumps(good))
            self.assertEqual(get_qualified_scenario(request, cache=cache), good)
            for field, value in (("status", "candidate_only"),
                                 ("solver_fingerprint", "stale"),
                                 ("nash_conv_chips", 0.2),
                                 ("nash_conv_chips", float("nan")),
                                 ("nash_conv_chips", True),
                                 ("result", None)):
                invalid = {**good, field: value}
                path.write_text(json.dumps(invalid))
                self.assertIsNone(get_qualified_scenario(request, cache=cache))
            changed = {**request, "pot": request["pot"] + 1}
            self.assertIsNone(get_qualified_scenario(changed, cache=cache))

    def test_qualified_worker_solve_is_consumed_by_scenario_lab(self):
        from trainer.server import analyze_river_position

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = solve_one(702, iterations=1000, combos_per_range=2,
                            threshold_chips=0.1, store=root / "candidates",
                            cache=root / "cache")
            self.assertEqual(row["status"], "cached_qualified")
            with patch("trainer.cfr_scenario_worker.CACHE", root / "cache"):
                report = analyze_river_position(row["request"])
        self.assertEqual(report["cache_status"], "exact_scenario_cache_hit")
        self.assertEqual(report["nash_conv"], row["result"]["nash_conv"])

    def test_bounded_worker_finishes_and_records_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status = run_loop(duration_seconds=60, pause_seconds=0, max_solves=1,
                              iterations=20, combos_per_range=4,
                              threshold_chips=1e9, seed=703,
                              store=root / "store", cache=root / "cache")
            history = json.loads((root / "store" / "history.json").read_text())
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(status["completed"], 1)
        self.assertEqual(status["qualified"], 1)
        self.assertEqual(history[0]["status"], "cached_qualified")


if __name__ == "__main__":
    unittest.main()
