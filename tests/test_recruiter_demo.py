import unittest

from poker_cpu.agents import HeuristicAgent
from trainer.recruiter_demo import HANDS, new_game, ordered_deck
from trainer.server import TableSession


class RecruiterFixtureTests(unittest.TestCase):
    def test_all_three_decks_are_unique_reproducible_and_reach_showdown(self):
        self.assertEqual(len(HANDS), 3)
        for index, hand in enumerate(HANDS):
            with self.subTest(hand=hand.name):
                self.assertEqual(len(set(ordered_deck(hand))), 52)
                game = new_game(index)
                self.assertEqual(game.hole[0], hand.hero)
                self.assertEqual(game.hole[1], hand.opponent)
                self.assertEqual(game.button, hand.button)
                opponent = HeuristicAgent(hand.opponent_seed, samples=8)
                while not game.done:
                    observation = game.observe(game.current_player)
                    if game.current_player == 0:
                        action = next(a for a in observation["legal_actions"]
                                      if a.kind in ("check", "call"))
                    else:
                        action = opponent.act(observation)
                    game.step(action)
                self.assertTrue(game.showdown)
                self.assertEqual(tuple(game.board), hand.board)
                self.assertEqual(sum(game.rewards()), 0)
                self.assertEqual(tuple(game.stacks),
                                 tuple(start + reward for start, reward
                                       in zip(game.initial_stacks, game.rewards())))

    def test_invalid_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown showcase"):
            new_game(3)

    def test_session_advances_three_playable_hands_then_stops(self):
        session = TableSession()
        state = session.start_showcase()
        for index, hand in enumerate(HANDS):
            self.assertEqual(state["showcase"]["index"], index + 1)
            self.assertEqual(state["showcase"]["name"], hand.name)
            if index == 0:
                with self.assertRaisesRegex(ValueError, "Finish"):
                    session.next_showcase_hand()
            while not state["done"]:
                action = next(a for a in state["legal_actions"]
                              if a["kind"] in ("check", "call"))
                state = session.act(action=action)
            self.assertEqual(tuple(state["board"]), hand.board)
            self.assertIsNotNone(state["opponent_hole"])
            if index + 1 < len(HANDS):
                state = session.next_showcase_hand()
        with self.assertRaisesRegex(ValueError, "complete"):
            session.next_showcase_hand()
        self.assertIsNone(session.new_hand()["showcase"])

    def test_same_user_choices_replay_identical_opponent_decisions(self):
        histories = []
        for _ in range(2):
            session = TableSession()
            state = session.start_showcase()
            while not state["done"]:
                action = next(a for a in state["legal_actions"]
                              if a["kind"] in ("check", "call"))
                state = session.act(action=action)
            histories.append((state["history"], state["board"], state["rewards"]))
        self.assertEqual(histories[0], histories[1])


if __name__ == "__main__":
    unittest.main()
