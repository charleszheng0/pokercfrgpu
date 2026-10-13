import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from trainer.server import EXPERIMENTS, TableSession, experiment_results
from poker_cpu.nfsp import HoldemNFSPAgent


class TrainerTests(unittest.TestCase):
    def test_hidden_cards_and_complete_history(self):
        table = TableSession()
        state = table.new_hand("random")
        class PassiveAgent:
            def act(self, observation):
                call = next(
                    (a for a in observation["legal_actions"] if a.kind == "call"),
                    None,
                )
                return call or next(
                    a for a in observation["legal_actions"] if a.kind == "check"
                )

        table.agent = PassiveAgent()
        self.assertIsNone(state["opponent_hole"])
        self.assertNotIn("deck", state)
        for _ in range(40):
            if state["done"]:
                break
            action = next(
                (
                    i
                    for i, a in enumerate(state["legal_actions"])
                    if a["kind"] == "call"
                ),
                None,
            )
            if action is None:
                action = next(
                    (
                        i
                        for i, a in enumerate(state["legal_actions"])
                        if a["kind"] == "check"
                    ),
                    0,
                )
            state = table.act(action)
        self.assertTrue(state["done"])
        self.assertEqual(len(state["opponent_hole"]), 2)
        self.assertEqual(len(state["hands"]), 1)
        self.assertEqual(sum(state["rewards"]), 0)

    def test_invalid_action_and_opponent(self):
        table = TableSession()
        with self.assertRaises(ValueError):
            table.new_hand("unknown")
        table.new_hand("random")
        with self.assertRaises(ValueError):
            table.act(999)

    def test_fold_does_not_reveal_opponent_cards_in_current_or_saved_history(self):
        table = TableSession()
        state = table.new_hand("random")
        fold_index = next(
            index
            for index, action in enumerate(state["legal_actions"])
            if action["kind"] == "fold"
        )
        state = table.act(fold_index)
        self.assertTrue(state["done"])
        self.assertIsNone(state["opponent_hole"])
        self.assertIsNone(state["hands"][0]["opponent_hole"])
        self.assertEqual(len(state["hands"][0]["hole"]), 2)

    def test_visible_street_bets_and_raise_amounts(self):
        table = TableSession()
        state = table.new_hand("random")
        self.assertEqual(state["street_bets"], tuple(table.game.street_paid))
        raise_index = next(
            index
            for index, action in enumerate(state["legal_actions"])
            if action["kind"] == "raise"
        )
        raise_target = state["legal_actions"][raise_index]["amount"]
        starting_bet = state["street_bets"][0]
        state = table.act(raise_index)
        first_action = state["actions"][0]
        self.assertEqual(first_action["kind"], "raise")
        self.assertEqual(first_action["target"], raise_target)
        self.assertEqual(first_action["amount_paid"], raise_target - starting_bet)

    def test_custom_legal_wager_target_is_playable(self):
        table = TableSession()
        state = table.new_hand("random")
        self.assertEqual(state["button"], 0)
        self.assertEqual(state["betting_limits"]["kind"], "raise")
        self.assertNotIn(
            {"kind": "raise", "amount": 7}, state["legal_actions"]
        )
        state = table.act(action={"kind": "raise", "amount": 7})
        self.assertEqual(state["history"][0], (0, 0, "raise", 7))
        self.assertEqual(state["actions"][0]["target"], 7)
        with self.assertRaises(ValueError):
            table.act(action={"kind": "raise", "amount": 3})

    def test_next_hand_alternates_button_and_preserves_history(self):
        table = TableSession()
        state = table.new_hand("random")
        while not state["done"]:
            call = next(
                (i for i, action in enumerate(state["legal_actions"]) if action["kind"] == "call"),
                None,
            )
            index = call if call is not None else next(
                i for i, action in enumerate(state["legal_actions"]) if action["kind"] == "check"
            )
            state = table.act(index)
        self.assertEqual(len(state["hands"]), 1)
        self.assertIn("events", state)
        self.assertEqual(state["hands"][0]["button"], 0)
        next_state = table.new_hand("random")
        self.assertEqual(next_state["hand"], 2)
        self.assertEqual(next_state["button"], 1)

    def test_dashboard_rows_have_saved_sources(self):
        rows = experiment_results()
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue((EXPERIMENTS / row["source"]).is_file())

    def test_nfsp_opponent_loads_a_saved_policy(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "policy.pt"
            torch.save(HoldemNFSPAgent().policy.state_dict(), model_path)
            with patch("trainer.server.NFSP_MODEL", model_path), patch(
                "trainer.server.NFSP_HERO_MODEL", model_path
            ):
                table = TableSession()
                state = table.new_hand("nfsp")
                self.assertEqual(state["starting_stack"], 20)
                self.assertIsNone(state["opponent_hole"])
                if state["analysis"] is not None:
                    self.assertAlmostEqual(
                        sum(state["analysis"]["action_probabilities"]), 1, delta=1e-6
                    )
                    self.assertIsNone(state["analysis"]["action_ev"])
                state = table.act(action={"kind": "raise", "amount": 7})
                self.assertTrue(state["decision_reviews"])
                review = state["decision_reviews"][0]
                self.assertFalse(review["chosen_in_policy_abstraction"])
                self.assertIsNone(review["chosen_probability"])
                self.assertAlmostEqual(
                    sum(review["action_probabilities"]), 1, delta=1e-6
                )
                self.assertIn("hand", state)


if __name__ == "__main__":
    unittest.main()
