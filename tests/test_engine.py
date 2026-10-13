import random
import unittest

from poker_cpu.cards import evaluate, parse_cards
from poker_cpu.engine import Action, HoldemGame
from poker_cpu.agents import RandomAgent, HeuristicAgent


class EngineTests(unittest.TestCase):
    def test_hand_order_and_ties(self):
        hands = [
            "As Kd Qc Jh 9s", "2s 2d Ac Kh Qc", "2s 2d 3c 3h As",
            "2s 2d 2c Ah Ks", "As 2d 3c 4h 5s", "As Js 8s 4s 2s",
            "2s 2d 2c 3h 3s", "2s 2d 2c 2h As", "9s Ts Js Qs Ks",
        ]
        scores = [evaluate(parse_cards(h)) for h in hands]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(evaluate(parse_cards("As 2d 3c 4h 5s"))[1], 5)
        self.assertEqual(evaluate(parse_cards("As Kd Qc Jh 9s 8d 7c")), scores[0])
        self.assertRaises(ValueError, parse_cards, "As As")

    def test_blind_option_and_streets(self):
        g = HoldemGame(seed=1)
        self.assertEqual((g.pot, g.current_player), (3, 0))
        g.step(Action("call"))
        self.assertEqual((g.street, g.current_player), (0, 1))
        g.step(Action("check"))
        self.assertEqual((g.street, len(g.board), g.current_player), (1, 3, 1))
        for expected in (2, 3, 3):
            g.step(Action("check"))
            g.step(Action("check"))
            self.assertEqual(g.street, expected)
        self.assertTrue(g.done)
        self.assertEqual(sum(g.rewards()), 0)

    def test_heads_up_action_order_for_both_button_positions(self):
        for button in (0, 1):
            g = HoldemGame(button=button, seed=10 + button)
            self.assertEqual(g.current_player, button)
            self.assertEqual(g.street_paid[button], 1)
            self.assertEqual(g.street_paid[1 - button], 2)
            g.step(Action("call"))
            self.assertEqual(g.current_player, 1 - button)
            g.step(Action("check"))
            self.assertEqual(g.street, 1)
            self.assertEqual(g.current_player, 1 - button)

    def test_arbitrary_integer_chip_raise_and_minimum_enforcement(self):
        g = HoldemGame(seed=6)
        limits = g.betting_limits()
        self.assertEqual(limits["kind"], "raise")
        self.assertEqual((limits["min_target"], limits["max_target"]), (4, 100))
        self.assertNotIn(Action("raise", 7), g.legal_actions())
        self.assertTrue(g.is_legal_action(Action("raise", 7)))
        with self.assertRaises(ValueError):
            g.step(Action("raise", 3))
        g.step(Action("raise", 7))
        self.assertEqual(g.street_paid[0], 7)

    def test_short_all_in_raise_does_not_reopen_action(self):
        g = HoldemGame(stacks=(20, 5), seed=7)
        g.step(Action("raise", 4))
        self.assertTrue(g.is_legal_action(Action("raise", 5)))
        g.step(Action("raise", 5))
        self.assertFalse(g.raises_open)
        self.assertEqual(g.legal_actions(), [Action("fold"), Action("call")])
        g.step(Action("call"))
        self.assertTrue(g.done)
        self.assertTrue(g.showdown)
        self.assertEqual(g.paid, [5, 5])
        self.assertEqual(len(g.board), 5)

    def test_short_blind_is_posted_and_unmatched_blind_returned(self):
        g = HoldemGame(stacks=(1, 10), button=0, seed=8)
        self.assertTrue(g.done)
        self.assertEqual(g.paid, [1, 1])
        self.assertEqual(g.pot, 2)
        self.assertEqual(sum(g.stacks), 11)
        self.assertEqual(len(g.board), 5)

    def test_odd_chip_on_tie_goes_left_of_button(self):
        g = HoldemGame(button=0, seed=9)
        g.hole = [("2c", "3d"), ("4c", "5d")]
        g.board = ["As", "Ks", "Qs", "Js", "Ts"]
        g.initial_stacks = (10, 10)
        g.stacks = [9, 8]
        g.paid = [1, 2]
        g.pot = 3
        g._showdown()
        self.assertIsNone(g.winner)
        self.assertEqual(g.stacks, [10, 10])
        self.assertEqual(g.rewards(), (0, 0))

    def test_deck_hole_board_and_burn_cards_remain_unique(self):
        g = HoldemGame(seed=19)
        while not g.done:
            action = next(
                (a for a in g.legal_actions() if a.kind in ("call", "check")),
                g.legal_actions()[0],
            )
            g.step(action)
        all_cards = tuple(g.hole[0] + g.hole[1] + tuple(g.board) + tuple(g.burned) + tuple(g.deck))
        self.assertEqual(len(all_cards), 52)
        self.assertEqual(len(set(all_cards)), 52)

    def test_public_event_log_tracks_deal_streets_and_award_without_hole_leak(self):
        g = HoldemGame(seed=21)
        self.assertEqual(g.events[0]["type"], "deal")
        self.assertEqual([event["type"] for event in g.events[1:3]], ["blind", "blind"])
        self.assertNotIn("hole", g.observe(0)["events"][-1])
        while not g.done:
            action = next(
                (a for a in g.legal_actions() if a.kind in ("call", "check")),
                g.legal_actions()[0],
            )
            g.step(action)
        self.assertEqual(
            [event["street"] for event in g.events if event["type"] == "board"],
            [1, 2, 3],
        )
        self.assertEqual(g.events[-1]["type"], "award")
        self.assertTrue(all("hole" not in event for event in g.events))

    def test_randomized_arbitrary_wagers_preserve_chip_conservation(self):
        for seed in range(75):
            rng = random.Random(seed + 8000)
            stacks = (rng.randint(2, 35), rng.randint(2, 35))
            game = HoldemGame(stacks=stacks, button=seed % 2, seed=seed)
            for _ in range(80):
                if game.done:
                    break
                self.assertEqual(sum(game.stacks) + game.pot, sum(stacks))
                limits = game.betting_limits()
                if limits and rng.random() < 0.5:
                    action = Action(
                        limits["kind"],
                        rng.randint(limits["min_target"], limits["max_target"]),
                    )
                else:
                    action = rng.choice(game.legal_actions())
                self.assertTrue(game.is_legal_action(action))
                game.step(action)
            self.assertTrue(game.done)
            self.assertEqual(sum(game.stacks), sum(stacks))
            self.assertEqual(sum(game.rewards()), 0)

    def test_bet_call_fold_and_all_in(self):
        g = HoldemGame(stacks=(20, 20), seed=2)
        self.assertIn(Action("raise", 4), g.legal_actions())
        g.step(Action("raise", 20))
        self.assertEqual(g.legal_actions(), [Action("fold"), Action("call")])
        g.step(Action("call"))
        self.assertTrue(g.done)
        self.assertEqual(len(g.board), 5)
        self.assertEqual(sum(g.stacks), 40)
        f = HoldemGame(seed=2)
        f.step(Action("fold"))
        self.assertFalse(f.showdown)
        self.assertEqual(f.rewards(), (-1, 1))

    def test_short_stack_refund(self):
        g = HoldemGame(stacks=(3, 20), seed=4)
        g.step(Action("call"))
        g.step(Action("check"))
        self.assertEqual(g.street, 1)
        self.assertIn(Action("bet", 1), g.legal_actions())
        g.step(Action("bet", 1))
        g.step(Action("call"))
        self.assertTrue(g.done)
        self.assertEqual(sum(g.stacks), 23)
        self.assertEqual(g.paid, [3, 3])

    def test_observation_hides_other_hand_and_deck(self):
        g = HoldemGame(seed=8)
        obs = g.observe(0)
        self.assertNotIn("deck", obs)
        self.assertNotIn("opponent", obs)
        self.assertNotIn(g.hole[1][0], obs["hole"] + obs["board"])
        # Swap hidden cards while preserving public state: observation is identical.
        g.hole[1] = (g.deck[0], g.deck[1])
        self.assertEqual(obs, g.observe(0))

    def test_randomized_conservation_legality_and_determinism(self):
        for seed in range(150):
            stacks = (random.Random(seed).randint(2, 30), random.Random(seed + 1234).randint(2, 30))
            g = HoldemGame(stacks=stacks, seed=seed, button=seed % 2)
            twin = HoldemGame(stacks=stacks, seed=seed, button=seed % 2)
            bot = RandomAgent(seed)
            for _ in range(40):
                self.assertEqual(sum(g.stacks) + (0 if g.done else g.pot), sum(stacks))
                self.assertEqual(g.pot, sum(g.paid))
                self.assertEqual(g.observe(g.current_player), twin.observe(twin.current_player))
                if g.done:
                    break
                action = bot.act(g.observe(g.current_player))
                g.step(action)
                twin.step(action)
            self.assertTrue(g.done)
            self.assertEqual(g.rewards(), twin.rewards())
            self.assertEqual(sum(g.rewards()), 0)

    def test_baseline_agents_choose_legal_actions(self):
        g = HoldemGame(seed=3)
        for bot in (RandomAgent(3), HeuristicAgent(3, samples=8)):
            self.assertIn(bot.act(g.observe(0)), g.legal_actions())


if __name__ == "__main__":
    unittest.main()
