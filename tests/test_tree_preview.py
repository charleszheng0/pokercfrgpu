"""The Lab must show the exact public actions accepted by its solver tree."""

import unittest

from poker_cpu.cards import DECK
from poker_cpu.engine import Action
from poker_cpu.multi_street import MultiStreetGame
from poker_cpu.solve import ActionAbstraction
from trainer.tree_preview import preview_betting_tree


class TreePreviewTests(unittest.TestCase):
    @staticmethod
    def payload(tree="default", spot="opening", **changes):
        row = {"board": "As Kh Qd 7c 2s", "pot": 20,
               "stacks": [100, 100], "betting_tree": tree, "spot": spot}
        row.update(changes)
        return row

    def test_each_preview_matches_solver_game_legal_actions(self):
        for tree in ("default", "shark_2_0_inspired"):
            abstraction = (ActionAbstraction.shark_style_postflop()
                           if tree != "default" else ActionAbstraction())
            board = ("As", "Kh", "Qd", "7c", "2s")
            available = [card for card in DECK if card not in board]
            game = MultiStreetGame(
                board, ((tuple(available[:2]),), (tuple(available[2:4]),)),
                pot=20, stack=100, button=0,
                bet_fractions=abstraction.bet_fractions,
                raise_fractions=abstraction.raise_fractions,
                bet_fractions_by_street=dict(abstraction.bet_fractions_by_street),
                raise_fractions_by_street=dict(abstraction.raise_fractions_by_street),
                max_raises=abstraction.max_raises,
                include_min_size=abstraction.include_min_size,
            )
            for spot in ("opening", "checked_to_hero", "facing_bet",
                         "facing_bet_first", "facing_raise"):
                with self.subTest(tree=tree, spot=spot):
                    report = preview_betting_tree(self.payload(tree, spot))
                    state = game.initial_state(report["first_actor"])
                    for row in report["history"]:
                        state = game.advance(state, Action(row["kind"], row["amount"]))
                    self.assertEqual(state.current, 0)
                    self.assertEqual(report["paid"], state.paid)
                    self.assertEqual(report["legal_actions"], [
                        {"kind": action.kind, "amount": action.amount}
                        for action in game.legal(state)
                    ])

    def test_tree_menu_changes_prior_bets_and_never_keeps_illegal_target(self):
        default = preview_betting_tree(self.payload(spot="facing_bet", prior_bet=10))
        shark = preview_betting_tree(self.payload(
            "shark_2_0_inspired", "facing_bet", prior_bet=10))
        self.assertEqual(default["selected_prior_bet"], 10)
        self.assertEqual(shark["prior_bet_targets"], [6, 13, 20, 100])
        self.assertEqual(shark["selected_prior_bet"], 6)
        self.assertNotEqual(default["legal_actions"], shark["legal_actions"])
        self.assertGreater(default["current_street_public_decisions"], 0)
        self.assertGreater(shark["current_street_public_action_edges"],
                           shark["current_street_public_decisions"])

    def test_invalid_position_is_rejected_before_a_solve(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            preview_betting_tree(self.payload(board="As As Qd 7c 2s"))
        with self.assertRaisesRegex(ValueError, "Unknown betting tree"):
            preview_betting_tree(self.payload(tree="fictional"))

    def test_facing_raise_omits_bets_that_cannot_be_raised(self):
        row = preview_betting_tree(self.payload(
            "shark_2_0_inspired", "facing_raise", prior_bet=100))
        self.assertNotIn(100, row["prior_bet_targets"])
        self.assertEqual(row["selected_prior_bet"], 6)
        self.assertTrue(row["prior_raise_targets"])


if __name__ == "__main__":
    unittest.main()
