import math
import gc
import unittest
import weakref

from poker_cpu.engine import Action
from poker_cpu.multi_street import MultiStreetGame
from poker_cpu.one_street_solver import OneStreetGame
from poker_cpu.river_solver import RiverCFR, RiverGame, RiverState


class MultiStreetTests(unittest.TestCase):
    def test_per_game_caches_release_finished_solve(self):
        game = MultiStreetGame("As Kh Qd 7c", ((("8h", "8s"),),
                                              (("9h", "9s"),)),
                               pot=8, stack=1)
        root = game.initial_state()
        game.advance(root, game.legal(root)[0])
        reference = weakref.ref(game)
        del game, root
        gc.collect()
        self.assertIsNone(reference())

    def test_river_tree_matches_preserved_river_reference(self):
        board = "As Kh Qd 7c 2s"
        ranges = ((("Jc", "Tc"), ("8h", "8s")),
                  (("9h", "9s"), ("Ac", "Ad")))
        old = RiverGame(board, ranges, pot=8, stack=10)
        new = MultiStreetGame(board, ranges, pot=8, stack=10, button=1)
        root = new.initial_state()
        self.assertEqual(root.current, 0)
        self.assertEqual(new.legal(root), old.legal(RiverState()))
        for hand0, hand1, _ in old.deals:
            for first in old.legal(RiverState()):
                if first.kind != "bet":
                    continue
                old_bet = old.advance(RiverState(), first)
                new_bet = new.advance(root, first)
                self.assertEqual(new.legal(new_bet), old.legal(old_bet))
                for response in old.legal(old_bet):
                    if response.kind not in ("fold", "call"):
                        continue
                    self.assertAlmostEqual(
                        new.terminal_payoff(new.advance(new_bet, response), (hand0, hand1)),
                        old.terminal_payoff(old.advance(old_bet, response), (hand0, hand1)),
                    )
        old_solver = RiverCFR(old, variant="cfr+").train(20)
        new_solver = RiverCFR(new, variant="cfr+", root_state=root).train(20)
        old_report = old_solver.analyze_root(private_hand=ranges[0][0])
        new_report = new_solver.analyze_root(private_hand=ranges[0][0])
        self.assertAlmostEqual(new_report["nash_conv"], old_report["nash_conv"], places=9)
        for action in old_report["actions"]:
            self.assertAlmostEqual(new_report["action_frequencies"][action],
                                   old_report["action_frequencies"][action], places=9)
            self.assertAlmostEqual(new_report["conditional_action_evs"][action],
                                   old_report["conditional_action_evs"][action], places=9)

    def test_turn_all_in_runout_matches_one_street_exact_equity(self):
        hero, opponent = ("8h", "8s"), ("9h", "9s")
        ranges = ((hero,), (opponent,))
        turn = MultiStreetGame("As Kh Qd 7c", ranges, pot=8, stack=2,
                               button=1, bet_fractions=(), max_raises=0)
        state = turn.initial_state()
        state = turn.advance(state, Action("bet", 2))
        state = turn.advance(state, Action("call"))
        self.assertTrue(state.chance)
        outcomes = tuple(turn.chance_outcomes(state, (hero, opponent)))
        self.assertEqual(len(outcomes), 44)
        self.assertAlmostEqual(sum(probability for _, probability in outcomes), 1)
        self.assertTrue(all(child.done for child, _ in outcomes))
        actual = sum(probability * turn.terminal_payoff(child, (hero, opponent))
                     for child, probability in outcomes)
        baseline = OneStreetGame("As Kh Qd 7c", ranges, pot=8, stack=2)
        old = baseline.advance(baseline.advance(RiverState(), Action("bet", 2)),
                               Action("call"))
        self.assertAlmostEqual(actual, baseline.terminal_payoff(old, (hero, opponent)))
        self.assertAlmostEqual(turn.chance_terminal_payoff(state, (hero, opponent)), actual)

    def test_flop_all_in_collapse_matches_explicit_ordered_chance(self):
        hero, opponent = ("8h", "8s"), ("9h", "9s")
        game = MultiStreetGame("As Kh Qd", ((hero,), (opponent,)),
                               pot=8, stack=1, button=1,
                               bet_fractions=(), max_raises=0)
        state = game.advance(game.advance(game.initial_state(), Action("bet", 1)),
                             Action("call"))
        self.assertTrue(state.all_in_runout)
        explicit = sum(p_turn * p_river * game.terminal_payoff(river, (hero, opponent))
                       for turn, p_turn in game.chance_outcomes(state, (hero, opponent))
                       for river, p_river in game.chance_outcomes(turn, (hero, opponent)))
        self.assertAlmostEqual(game.chance_terminal_payoff(state, (hero, opponent)),
                               explicit, places=10)

    def test_sampled_chance_is_seeded_and_root_regret_matches_exact_in_mean(self):
        hero, opponent = ("8h", "8s"), ("9h", "9s")
        game = MultiStreetGame("As Kh Qd 7c", ((hero,), (opponent,)),
                               pot=8, stack=1, button=1,
                               bet_fractions=(), max_raises=0)
        root = game.initial_state()
        key = game.info(root, (hero, opponent))
        exact = RiverCFR(game, variant="cfr", root_state=root).train(1).regrets[key]
        sampled = [RiverCFR(game, variant="cfr", root_state=root,
                            chance_sampling=True, sampling_seed=seed).train(1)
                   for seed in range(500)]
        self.assertEqual(sampled[0].regrets, RiverCFR(
            game, variant="cfr", root_state=root,
            chance_sampling=True, sampling_seed=0).train(1).regrets)
        for action in exact:
            mean = sum(solver.regrets[key][action] for solver in sampled) / len(sampled)
            self.assertLess(abs(mean - exact[action]), 0.05)

    def test_turn_includes_river_betting_and_after_bet_root(self):
        hero, opponent = ("Jc", "Tc"), ("9h", "9s")
        game = MultiStreetGame("As Kh Qd 7c", ((hero,), (opponent,)),
                               pot=8, stack=4, button=0,
                               bet_fractions=(), max_raises=0)
        root = game.advance(game.initial_state(), Action("bet", 2))
        self.assertEqual(root.current, 0)
        solver = RiverCFR(game, variant="cfr+", root_state=root).train(3)
        report = solver.analyze_root(private_hand=hero)
        self.assertEqual(report["status"], "postflop_multi_street_abstracted")
        self.assertTrue(math.isfinite(report["nash_conv"]))
        self.assertAlmostEqual(sum(report["per_hand"][0]["frequency"].values()), 1)
        self.assertIn(Action("call"), report["actions"])

    def test_flop_public_chance_reaches_turn_and_river_betting(self):
        hero, opponent = ("Jc", "Tc"), ("9h", "9s")
        game = MultiStreetGame("As Kh Qd", ((hero,), (opponent,)),
                               pot=8, stack=4, button=0,
                               bet_fractions=(), max_raises=0)
        state = game.initial_state()
        self.assertEqual(state.current, 1)
        state = game.advance(game.advance(state, Action("check")), Action("check"))
        self.assertTrue(state.chance)
        turns = game.chance_outcomes(state, (hero, opponent))
        self.assertEqual(len(turns), 45)
        self.assertAlmostEqual(sum(probability for _, probability in turns), 1)
        turn = turns[0][0]
        self.assertEqual((turn.street, turn.current, len(turn.board)), (2, 1, 4))
        turn = game.advance(game.advance(turn, Action("check")), Action("check"))
        rivers = game.chance_outcomes(turn, (hero, opponent))
        self.assertEqual(len(rivers), 44)
        river = rivers[0][0]
        self.assertEqual((river.street, river.current, len(river.board)), (3, 1, 5))
        showdown = game.advance(game.advance(river, Action("check")), Action("check"))
        self.assertTrue(showdown.done)
        self.assertTrue(math.isfinite(game.terminal_payoff(showdown, (hero, opponent))))

    def test_flop_checkdown_value_matches_independent_runout_reference(self):
        hero, opponent = ("8h", "8s"), ("9h", "9s")
        ranges = ((hero,), (opponent,))
        game = MultiStreetGame("As Kh Qd", ranges, pot=8, stack=4,
                               button=0, bet_fractions=(), max_raises=0)
        check = lambda _state, _hands: {Action("check"): 1.0}
        actual = game.expected_value(check, check)
        reference = OneStreetGame("As Kh Qd", ranges, pot=8, stack=4)
        terminal = reference.advance(
            reference.advance(RiverState(), Action("check")), Action("check"))
        expected = reference.terminal_payoff(terminal, (hero, opponent))
        self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    unittest.main()
