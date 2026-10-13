import unittest

from poker_cpu.engine import HoldemGame
from poker_cpu.nfsp import (
    ACTION_COUNT,
    INPUT_DIM,
    HoldemNFSPAgent,
    encode,
    indexed_actions,
    play_self_play_hand,
)


class HoldemNFSPTests(unittest.TestCase):
    def test_information_state_and_legal_action_slots(self):
        game = HoldemGame(stacks=(20, 20), seed=7)
        obs = game.observe(game.current_player)
        state = encode(obs)
        self.assertEqual(len(state), INPUT_DIM)
        slots = indexed_actions(obs["legal_actions"])
        self.assertEqual(len(slots), len(obs["legal_actions"]))
        self.assertTrue(all(0 <= slot < ACTION_COUNT for slot in slots))
        self.assertEqual(set(slots.values()), set(obs["legal_actions"]))
        before = state.clone()
        game.hole[1 - game.current_player] = (game.deck[0], game.deck[1])
        self.assertTrue(before.equal(encode(game.observe(game.current_player))))

    def test_self_play_transitions_and_masked_actions(self):
        agents = [HoldemNFSPAgent(), HoldemNFSPAgent()]
        for i in range(30):
            rewards = play_self_play_hand(agents, seed=i, button=i % 2)
            self.assertEqual(sum(rewards), 0)
        self.assertTrue(all(len(agent.rl) > 0 for agent in agents))
        for agent in agents:
            for state, action, reward, next_state, done, mask in agent.rl:
                self.assertEqual(len(state), INPUT_DIM)
                self.assertEqual(len(mask), ACTION_COUNT)
                if done:
                    self.assertFalse(mask.any().item())
            for state, action, mask in agent.sl.data:
                self.assertTrue(mask[action].item())


if __name__ == "__main__":
    unittest.main()
