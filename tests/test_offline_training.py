import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from trainer.offline_training import DEFAULT, _promotion_gate, paired_policy_difference, run_once, validate_config
from trainer.solution_cache import get, put, sample_demo_action
from trainer.demo import DEMO_RIVER_REQUEST
from trainer.server import analyze_demo_position
from experiments.holdem_nfsp import evaluate, evaluate_head_to_head


class OfflineTrainingTests(unittest.TestCase):
    def test_paired_difference_uses_per_hand_deltas(self):
        row = paired_policy_difference([2, 1, -1, 3], [1, 0, -1, 2])
        self.assertEqual(row["hands"], 4)
        self.assertAlmostEqual(row["mean_chips_per_hand"], 0.75)
        self.assertGreater(row["ci95_normal"][0], 0)
        with self.assertRaisesRegex(ValueError, "equal lists"):
            paired_policy_difference([1, 2], [1])

    def test_promotion_requires_both_paired_and_direct_lower_bounds(self):
        row = {"paired_heuristic_difference": {"ci95_normal": [0.1, 1.0]},
               "head_to_head_incumbent": {"ci95_normal": [0.01, 0.5]}}
        self.assertTrue(_promotion_gate(row, 100))
        self.assertFalse(_promotion_gate(row, 99))
        self.assertFalse(_promotion_gate({**row,
            "head_to_head_incumbent": {"ci95_normal": [-0.01, 0.5]}}, 100))
        self.assertFalse(_promotion_gate({"paired_heuristic_difference": row[
            "paired_heuristic_difference"]}, 100))

    def test_direct_match_reports_seat_balanced_chip_return(self):
        class Passive:
            def act(self, obs, record=False):
                action = next(action for action in obs["legal_actions"]
                              if action.kind in ("check", "call"))
                return 0, action

        row = evaluate_head_to_head((Passive(), Passive()),
                                    (Passive(), Passive()), 4, 73)
        self.assertEqual(row["hands"], 4)
        self.assertEqual(row["policy_scope"], "seat_specific_checkpoint_pair")
        self.assertTrue(math.isfinite(row["mean_chips_per_hand"]))
        self.assertAlmostEqual(row["ci95_normal"][0] + row["ci95_normal"][1],
                               2 * row["mean_chips_per_hand"])

    def test_deployed_policy_evaluation_keeps_its_training_seat(self):
        class SeatRecorder:
            def __init__(self):
                self.seats = set()

            def act(self, obs, record=False):
                self.seats.add(obs["player"])
                action = next(action for action in obs["legal_actions"]
                              if action.kind in ("check", "call"))
                return 0, action

        fixed = SeatRecorder()
        report = evaluate(fixed, "random", 4, 113, agent_seat=1)
        self.assertEqual(fixed.seats, {1})
        self.assertEqual(report["agent_seat"], 1)
        alternating = SeatRecorder()
        evaluate(alternating, "random", 4, 113)
        self.assertEqual(alternating.seats, {0, 1})

    def test_rejects_invalid_hyperparameters(self):
        for setting in ({"learning_rate": float("nan")}, {"batch_size": 0},
                        {"hidden_layers": [16, 0]}, {"epsilon": 1.2},
                        {"unexpected": 1}):
            with self.assertRaises(ValueError):
                validate_config(setting)

    def test_short_run_writes_real_candidate_and_checkpoints(self):
        with TemporaryDirectory() as directory:
            row = run_once({"hands_per_run": 24, "checkpoint_every": 8,
                            "batch_size": 4, "evaluation_hands": 4,
                            "hidden_layers": [16, 16], "rl_capacity": 32,
                            "sl_capacity": 32}, output=Path(directory), allow_promotion=False)
            self.assertEqual(row["promotion"], "candidate_only")
            self.assertEqual([point["hands"] for point in row["curves"]], [8, 16, 24])
            self.assertEqual(len(row["evaluation"]), 2)
            self.assertTrue((Path(row["model_dir"]) / "policy_player_1.pt").is_file())
            self.assertEqual(json.loads((Path(directory) / "history.json").read_text())[0]["run_id"], row["run_id"])
            self.assertFalse((Path(directory) / "active.json").exists())

    def test_demo_cache_samples_only_from_solved_strategy(self):
        with TemporaryDirectory() as directory:
            with patch("trainer.solution_cache.CACHE", Path(directory)):
                with self.assertRaisesRegex(ValueError, "No verified cached"):
                    sample_demo_action()
                solved = analyze_demo_position()
                self.assertTrue(put(DEMO_RIVER_REQUEST, solved))
                self.assertEqual(get(DEMO_RIVER_REQUEST)["analysis"]["nash_conv"], solved["analysis"]["nash_conv"])
                legal = [row["action"] for row in solved["analysis"]["frequencies"]]
                for _ in range(20):
                    self.assertIn(sample_demo_action()["action"], legal)

    def test_checkpoint_resume_matches_uninterrupted_weights(self):
        from trainer.offline_training import play_self_play_hand
        settings = {"hands_per_run": 16, "checkpoint_every": 4, "batch_size": 4,
                    "evaluation_hands": 2, "hidden_layers": [16, 16],
                    "rl_capacity": 32, "sl_capacity": 32}
        with TemporaryDirectory() as continuous, TemporaryDirectory() as resumed:
            full = run_once(settings, output=Path(continuous), allow_promotion=False)
            calls = 0
            def stop_after_eight(*args, **kwargs):
                nonlocal calls
                result = play_self_play_hand(*args, **kwargs)
                calls += 1
                if calls == 8:
                    (Path(resumed) / "stop").touch()
                return result
            with patch("trainer.offline_training.play_self_play_hand", side_effect=stop_after_eight):
                partial = run_once(settings, output=Path(resumed), allow_promotion=False)
            self.assertEqual(partial["hands"], 8)
            (Path(resumed) / "stop").unlink()
            restarted = run_once(settings, output=Path(resumed), allow_promotion=False)
            a = torch.load(Path(full["model_dir"]) / "policy_player_1.pt", weights_only=True)
            b = torch.load(Path(restarted["model_dir"]) / "policy_player_1.pt", weights_only=True)
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))

    def test_completed_scheduled_runs_continue_same_nfsp_learner(self):
        settings = {"hands_per_run": 8, "checkpoint_every": 4, "batch_size": 4,
                    "evaluation_hands": 2, "hidden_layers": [16, 16],
                    "rl_capacity": 32, "sl_capacity": 32}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            first = run_once(settings, output=output, allow_promotion=False)
            saved_first = torch.load(output / "latest_checkpoint.pt", weights_only=False)
            second = run_once(settings, output=output, allow_promotion=False)
            saved_second = torch.load(output / "latest_checkpoint.pt", weights_only=False)
            self.assertEqual(second["continued_from_run_id"], first["run_id"])
            self.assertEqual(first["cumulative_training_hands"], 8)
            self.assertEqual(second["cumulative_training_hands"], 16)
            self.assertEqual(saved_second["experience_hands"], 16)
            self.assertGreater(len(saved_second["agents"][0]["rl"]),
                               len(saved_first["agents"][0]["rl"]))

    def test_candidate_comparison_records_paired_result_without_hand_records(self):
        settings = {"hands_per_run": 8, "checkpoint_every": 4, "batch_size": 4,
                    "evaluation_hands": 4, "hidden_layers": [16, 16],
                    "rl_capacity": 32, "sl_capacity": 32}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            first = run_once(settings, output=output, allow_promotion=False)
            (output / "active.json").write_text(json.dumps({
                "run_id": first["run_id"], "model_dir": first["model_dir"],
                "hidden_layers": [16, 16]}))
            second = run_once(settings, output=output, allow_promotion=False)
            pair = second["paired_heuristic_difference"]
            self.assertEqual(pair["hands"], 4)
            self.assertAlmostEqual(pair["mean_chips_per_hand"],
                                   second["heuristic_delta_chips"])
            self.assertNotIn("per_hand_gains", second["evaluation"][1])
            self.assertNotIn("per_hand_gains", second["incumbent_heuristic"])
            self.assertEqual(second["evaluation"][1]["agent_seat"], 1)
            self.assertEqual(second["incumbent_heuristic"]["agent_seat"], 1)
            self.assertEqual(second["head_to_head_incumbent"]["policy_scope"],
                             "seat_specific_checkpoint_pair")
            self.assertEqual(second["promotion"], "candidate_only")


if __name__ == "__main__":
    unittest.main()
