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
