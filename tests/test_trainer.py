import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from trainer.server import EXPERIMENTS, TableSession, analyze_demo_position, analyze_river_position, experiment_results, live_analysis_availability
from poker_cpu.nfsp import HoldemNFSPAgent


class TrainerTests(unittest.TestCase):
    def test_one_click_demo_uses_real_restricted_river_analysis(self):
        result = analyze_demo_position()
        position, report = result["position"], result["analysis"]
        self.assertEqual(position["board"], ("As", "Kh", "Qd", "7c", "2s"))
        self.assertEqual(position["hero_hand"], ("8h", "8s"))
        self.assertEqual(report["iterations"], 2000)
        self.assertEqual(report["joint_deals"], 4)
        self.assertLess(report["nash_conv"], 0.1)
        self.assertEqual({row["action"]["kind"] for row in report["action_evs"]}, {"check", "bet"})
        self.assertAlmostEqual(sum(row["frequency"] for row in report["frequencies"]), 1)
        self.assertEqual(len(report["per_hand"]), 2)
        self.assertIn("not inferred", position["description"])
        self.assertNotIn("opponent_hole", position)

    def test_live_analysis_status_is_explicit_for_each_street(self):
        for street in range(4):
            report = live_analysis_availability(street, ())
            self.assertEqual(report["status"], "unavailable")
            self.assertIn("unavailable", report["reason"])
        after_bet = live_analysis_availability(3, ((3, 1, "bet", 10),))
        self.assertIn("earlier river actions", after_bet["reason"])

    def test_play_table_exposes_a_public_study_position_without_opponent_cards(self):
        table = TableSession()
        state = table.new_hand("random")
        study = state["study_position"]
        self.assertEqual(study["street"], "preflop")
        self.assertEqual(study["hero_hand"], state["hole"])
        self.assertEqual(study["pot_before_street"], 0)
        self.assertEqual(study["stacks_at_street_start"], (100, 100))
        self.assertEqual(study["street_actions"], ())
        self.assertIsNone(state["opponent_hole"])
        self.assertNotIn("opponent_hand", study)
        self.assertEqual(state["solver_feedback"]["solver_api_status"], "unsupported")

    @staticmethod
    def river_payload(iterations=20):
        return {
            "board": "As Kh Qd 7c 2s", "hero_hand": "Jc Tc", "pot": 20,
            "stacks": [100, 100], "iterations": iterations,
            "ranges": [
                {"hands": [["Jc", "Tc"], ["9c", "8c"], ["4c", "4d"]], "weights": [1, 1, 1]},
                {"hands": [["Ac", "Ad"], ["Jh", "Jd"], ["8h", "8s"]], "weights": [1, 2, 1]},
            ],
        }

    def test_explicit_river_analysis_returns_hand_frequencies_evs_and_exact_nashconv(self):
        report = analyze_river_position(self.river_payload())
        self.assertEqual(report["status"], "restricted_river_analysis")
        self.assertEqual(report["iterations"], 20)
        self.assertGreaterEqual(report["latency_ms"], 0)
        self.assertEqual(len(report["frequencies"]), len(report["action_evs"]))
        self.assertAlmostEqual(sum(row["frequency"] for row in report["frequencies"]), 1)
        self.assertTrue(all(isinstance(row["ev"], float) for row in report["action_evs"]))
        best_ev = max(row["ev"] for row in report["action_evs"])
        self.assertTrue(all(abs(row["ev_loss"] - (best_ev - row["ev"])) < 1e-10 for row in report["action_evs"]))
        self.assertGreaterEqual(report["nash_conv"], 0)
        self.assertIn("exact NashConv", report["validation_status"])
        self.assertIn("not inferred from betting history", report["range_note"])
        self.assertIn("supplied abstract action history", report["supported_state"])
        self.assertIsNone(report["target_nash_conv_chips"])
        self.assertTrue(report["convergence_trace"])

    def test_explicit_nashconv_target_is_returned_with_measured_status(self):
        payload = self.river_payload(iterations=40)
        payload["target_nash_conv_chips"] = 1_000_000
        report = analyze_river_position(payload)
        self.assertEqual(report["solver_quality_status"], "restricted_target_met")
        self.assertEqual(report["target_nash_conv_chips"], 1_000_000)
        self.assertLess(report["iterations"], 40)
        self.assertLessEqual(report["nash_conv"], report["target_nash_conv_chips"])
        self.assertGreaterEqual(len(report["convergence_trace"]), 1)

    def test_river_input_accepts_reversed_private_card_order(self):
        payload = self.river_payload()
        payload["hero_hand"] = "Tc Jc"
        payload["ranges"][0]["hands"][0] = ["Tc", "Jc"]
        self.assertEqual(analyze_river_position(payload)["hero_hand"], ("Jc", "Tc"))

    def test_river_analysis_uses_the_actual_post_action_root(self):
        payload = self.river_payload()
        payload["river_actions"] = [
            {"kind": "check", "amount": 0},
            {"kind": "bet", "amount": 10},
        ]
        report = analyze_river_position(payload)
        self.assertEqual(report["river_paid"], (0, 10))
        self.assertEqual({row["action"]["kind"] for row in report["frequencies"]} & {"fold", "call"}, {"fold", "call"})
        self.assertAlmostEqual(sum(row["frequency"] for row in report["frequencies"]), 1)
        payload["river_actions"][1]["amount"] = 9
        with self.assertRaisesRegex(ValueError, "outside the restricted tree"):
            analyze_river_position(payload)
        payload["river_actions"] = [{"kind": "check", "amount": 0}]
        with self.assertRaisesRegex(ValueError, "hero must be facing"):
            analyze_river_position(payload)

    def test_river_analysis_when_opponent_acts_first(self):
        payload = self.river_payload()
        payload["first_actor"] = 1
        payload["river_actions"] = [{"kind": "check", "amount": 0}]
        checked_to_hero = analyze_river_position(payload)
        self.assertEqual(checked_to_hero["first_actor"], 1)
        self.assertIn("check", {row["action"]["kind"] for row in checked_to_hero["frequencies"]})
        payload["river_actions"] = [{"kind": "bet", "amount": 10}]
        facing_bet = analyze_river_position(payload)
        self.assertEqual(facing_bet["river_paid"], (0, 10))
        self.assertIn("call", {row["action"]["kind"] for row in facing_bet["frequencies"]})

    def test_river_analysis_after_bet_and_raise(self):
        payload = self.river_payload()
        payload["river_actions"] = [
            {"kind": "bet", "amount": 2},
            {"kind": "raise", "amount": 14},
        ]
        report = analyze_river_position(payload)
        self.assertEqual(report["river_paid"], (2, 14))
        self.assertEqual({row["action"]["kind"] for row in report["frequencies"]}, {"fold", "call"})

    def test_shared_lab_api_can_solve_a_tiny_turn_position(self):
        payload = self.river_payload(iterations=2)
        payload["street"] = "turn"
        payload["board"] = "As Kh Qd 7c"
        payload["stacks"] = [4, 4]
        payload["ranges"] = [
            {"hands": [["Jc", "Tc"]], "weights": [1]},
            {"hands": [["9h", "9s"]], "weights": [1]},
        ]
        report = analyze_river_position(payload)
        self.assertEqual(report["status"], "restricted_turn_analysis")
        self.assertEqual(report["iterations"], 2)
        self.assertEqual(len(report["per_hand"]), 1)
        self.assertAlmostEqual(sum(row["frequency"] for row in report["frequencies"]), 1)
        self.assertGreaterEqual(report["nash_conv"], 0)

    def test_river_analysis_rejects_unsupported_or_inconsistent_inputs(self):
        payload = self.river_payload()
        payload["board"] = "As Kh Qd 7c"
        with self.assertRaisesRegex(ValueError, "Unsupported state"):
            analyze_river_position(payload)
        payload = self.river_payload()
        payload["hero_hand"] = "Ac Ad"
        with self.assertRaisesRegex(ValueError, "must appear"):
            analyze_river_position(payload)
        payload = self.river_payload()
        payload["ranges"][0]["weights"][0] = 0
        with self.assertRaisesRegex(ValueError, "no compatible opponent range mass"):
            analyze_river_position(payload)
        payload = self.river_payload()
        payload["iterations"] = 10
        with self.assertRaisesRegex(ValueError, "20 to 5000"):
            analyze_river_position(payload)

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

    def test_experimental_river_review_uses_saved_public_decision(self):
        table = TableSession()
        state = table.new_hand("random")
        class PassiveAgent:
            def act(self, observation):
                call = next((a for a in observation["legal_actions"] if a.kind == "call"), None)
                return call or next(a for a in observation["legal_actions"] if a.kind == "check")
        table.agent = PassiveAgent()
        for _ in range(20):
            if state["done"]:
                break
            action = next((i for i, a in enumerate(state["legal_actions"])
                           if a["kind"] == "call"), None)
            if action is None:
                action = next(i for i, a in enumerate(state["legal_actions"])
                              if a["kind"] == "check")
            state = table.act(action)
        self.assertTrue(state["done"])
        reviews = state["solver_reviews"]
        self.assertEqual(len(reviews), 4)
        self.assertNotIn("opponent_hole", reviews[-1])
        self.assertEqual(reviews[-1]["street"], "river")
        early = table.review_decision(state["hand"], 0)
        self.assertEqual(early["status"], "unsupported")
        for index, street in ((1, "flop"), (2, "turn")):
            cutoff = table.review_decision(state["hand"], index)
            self.assertEqual(cutoff["status"], "experimental_assumed_range")
            self.assertEqual(cutoff["iterations"], 500)
            self.assertIn("no later betting", cutoff["label"])
            self.assertIn("no later betting", cutoff["solver_scope_note"])
            self.assertEqual(cutoff["validation_status"],
                             "experimental_showdown_cutoff_residual_measured")
            self.assertAlmostEqual(sum(row["frequency"] for row in cutoff["actions"]), 1)
            self.assertTrue(all(math.isfinite(row["ev_chips"]) for row in cutoff["actions"]))
        river = table.review_decision(state["hand"], 3)
        self.assertEqual(river["status"], "experimental_assumed_range")
        self.assertEqual(river["iterations"], 1000)
        self.assertEqual(len(river["opponent_range"]), 3)
        self.assertAlmostEqual(sum(row["frequency"] for row in river["actions"]), 1)
        self.assertGreaterEqual(river["selected_action_ev_loss_chips"], 0)
        self.assertIs(state["hands"][-1]["solver_reviews"][-1]["analysis"], river)

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
