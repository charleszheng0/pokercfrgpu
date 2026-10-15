import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from trainer.offline_training import DEFAULT, run_once, validate_config
from trainer.solution_cache import get, put, sample_demo_action
from trainer.demo import DEMO_RIVER_REQUEST
from trainer.server import analyze_demo_position


class OfflineTrainingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
