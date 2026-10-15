import unittest
import math

from poker_cpu.engine import Action, HoldemGame
from poker_cpu.river_solver import RiverCFR, RiverGame, RiverState, WeightedRange, nash_conv


BOARD = "As Kh Qd 7c 2s"
RANGES = (
    (("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")),
    (("Jd", "Td"), ("8h", "8s"), ("4c", "4d")),
)


class RiverSolverTests(unittest.TestCase):
    def test_abstract_bets_match_main_engine_on_river(self):
        abstract = RiverGame(BOARD, RANGES, pot=8, stack=10)
        engine = HoldemGame(seed=1)
        engine.street = 3
        engine.board = list(abstract.board)
        engine.hole = [RANGES[0][0], RANGES[1][0]]
        engine.stacks = [10, 10]
        engine.pot = 8
        engine.paid = [4, 4]
        engine.street_paid = [0, 0]
        engine.current_player = 0
        engine.last_full_raise = 2
        self.assertEqual(tuple(engine.legal_actions()), abstract.legal(RiverState()))
        engine.step(Action("bet", 4))
        abstract_state = abstract.advance(RiverState(), Action("bet", 4))
        # The local solver intentionally removes a second raise from this tree.
        self.assertEqual(tuple(engine.legal_actions()), abstract.legal(abstract_state))

    def test_action_tree_and_payoff(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        start = RiverState()
        self.assertIn(Action("bet", 4), game.legal(start))
        after_bet = game.advance(start, Action("bet", 4))
        self.assertIn(Action("raise", 8), game.legal(after_bet))
        after_raise = game.advance(after_bet, Action("raise", 8))
        self.assertEqual(game.legal(after_raise), (Action("fold"), Action("call")))
        folded = game.advance(after_bet, Action("fold"))
        self.assertEqual(game.terminal_payoff(folded, (RANGES[0][0], RANGES[1][0])), 4)
        called = game.advance(after_bet, Action("call"))
        self.assertTrue(called.done)
        self.assertEqual(called.paid, (4, 4))

    def test_showdown_values_are_zero_sum_and_include_matched_bets(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        state = RiverState(paid=(4, 4), done=True, winner=-1)
        strong = ("Jc", "Tc")
        weak = ("4c", "4d")
        self.assertEqual(game.terminal_payoff(state, (strong, weak)), 8)
        self.assertEqual(game.terminal_payoff(state, (weak, strong)), -8)
        self.assertEqual(game.terminal_payoff(state, (strong, strong)), 0)

    def test_fold_payoff_is_zero_sum_with_unequal_contributions(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        state = RiverState(paid=(2, 6), done=True, winner=1)
        payoff = game.terminal_payoff(state, (RANGES[0][0], RANGES[1][0]))
        self.assertEqual(payoff, -6)

    def test_weighted_range_conditions_on_blockers_and_observed_actions(self):
        belief = WeightedRange(
            (("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")),
            (2, 1, 1),
        )
        posterior = belief.condition(
            dead_cards=("Jc",),
            action_likelihoods={("9h", "9s"): 0.25, ("3c", "3d"): 1.0},
        )
        self.assertEqual(posterior.hands, (("9h", "9s"), ("3c", "3d")))
        self.assertAlmostEqual(posterior.weights[0], 0.2)
        self.assertAlmostEqual(posterior.weights[1], 0.8)
        with self.assertRaises(ValueError):
            belief.condition(dead_cards=("Jc", "9h", "3c"))

    def test_weighted_range_rejects_overflowing_normalization_mass(self):
        with self.assertRaisesRegex(ValueError, "positive total weight"):
            WeightedRange((("Jc", "Tc"), ("9h", "9s")), (1e308, 1e308))

    def test_ranges_hide_collisions_and_use_weights(self):
        game = RiverGame(BOARD, RANGES, range_weights=((2, 1, 1), (1, 1, 1)))
        self.assertAlmostEqual(sum(weight for _, _, weight in game.deals), 1)
        self.assertGreater(game.deals[0][2], game.deals[-1][2])
        with self.assertRaises(ValueError):
            RiverGame(BOARD, ((("As", "Td"),), RANGES[1]), pot=8)

    def test_cfr_decreases_exact_vulnerability(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        solver = RiverCFR(game).train(20)
        early = nash_conv(game, solver.average, solver.average)
        solver.train(80)
        late = nash_conv(game, solver.average, solver.average)
        self.assertLess(late, early)
        self.assertLess(late, 1.0)
        values = solver.root_hand_values()
        self.assertEqual(set(values[0]), set(RANGES[0]))
        self.assertGreater(values[0][RANGES[0][0]], values[0][RANGES[0][2]])

    def test_analysis_reports_mixed_strategy_and_conditional_action_ev(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        solver = RiverCFR(game).train(80)
        report = solver.analyze_root(
            chosen_action=Action("check"), private_hand=RANGES[0][0]
        )
        self.assertEqual(report["status"], "river_abstracted")
        self.assertEqual(report["iterations"], 80)
        self.assertAlmostEqual(sum(report["action_frequencies"].values()), 1.0)
        self.assertEqual(set(report["conditional_action_evs"]), set(game.legal(RiverState())))
        self.assertGreaterEqual(report["chosen_action_ev_loss"], 0.0)
        self.assertGreaterEqual(report["range_average_action_gap"], 0.0)
        unconditioned = solver.analyze_root(chosen_action=Action("check"))
        self.assertIsNone(unconditioned["chosen_action_ev_loss"])
        self.assertIsNotNone(unconditioned["range_average_action_gap"])
        for row in report["per_hand"]:
            self.assertAlmostEqual(sum(row["frequency"].values()), 1.0)
            self.assertEqual(set(row["action_evs"]), set(game.legal(RiverState())))
            self.assertTrue(all(math.isfinite(value) for value in row["action_evs"].values()))

    def test_root_seat_one_is_trained_and_analyzed_separately(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        solver = RiverCFR(game, root_player=1).train(60)
        report = solver.analyze_root(player=1)
        self.assertAlmostEqual(sum(report["action_frequencies"].values()), 1.0)
        self.assertEqual(report["player"], 1)
        with self.assertRaises(ValueError):
            solver.analyze_root(player=0)

    def test_post_action_river_root_has_conditional_action_values(self):
        game = RiverGame(BOARD, RANGES, pot=8, stack=10)
        root = RiverState(last_raise=game.big_blind)
        root = game.advance(root, Action("check"))
        root = game.advance(root, Action("bet", 4))
        solver = RiverCFR(game, variant="cfr+", root_player=0, root_state=root).train(30)
        report = solver.analyze_root(player=0, private_hand=RANGES[0][0])
        self.assertEqual(solver.root_state.paid, (0, 4))
        self.assertIn(Action("fold"), report["actions"])
        self.assertIn(Action("call"), report["actions"])
        self.assertAlmostEqual(sum(report["per_hand"][0]["frequency"].values()), 1)
        self.assertTrue(all(math.isfinite(value) for value in report["per_hand"][0]["action_evs"].values()))
        hero = RANGES[0][0]
        hand_values = next(row["action_evs"] for row in report["per_hand"] if row["hand"] == hero)
        compatible = [(opponent, mass) for own, opponent, mass in game.deals if own == hero]
        total = sum(mass for _, mass in compatible)
        call_state = game.advance(root, Action("call"))
        expected_call = sum(mass / total * game.terminal_payoff(call_state, (hero, opponent)) for opponent, mass in compatible)
        self.assertAlmostEqual(hand_values[Action("call")], expected_call)
        self.assertEqual(hand_values[Action("fold")], -game.pot / 2)
        with self.assertRaisesRegex(ValueError, "nonterminal decision"):
            RiverCFR(game, root_player=1, root_state=root)

    def test_conditional_action_evs_match_independent_profile_enumeration(self):
        for player in (0, 1):
            game = RiverGame(BOARD, RANGES, pot=8, stack=10)
            solver = RiverCFR(game, root_player=player).train(50)
            report = solver.analyze_root(player=player)
            root = solver.root_state
            for action, expected in report["conditional_action_evs"].items():
                def brute_force(state, hands, forced_action):
                    if state.done:
                        return game.terminal_payoff(state, hands)
                    if state == root:
                        choices = ((forced_action, 1.0),)
                    else:
                        choices = tuple(solver.average(state, hands).items())
                    total = 0.0
                    for candidate, probability in choices:
                        successor = game.advance(state, candidate)
                        total += probability * brute_force(successor, hands, forced_action)
                    return total

                enumerated = sum(
                    chance * (1 if player == 0 else -1)
                    * brute_force(root, (hand0, hand1), action)
                    for hand0, hand1, chance in game.deals
                )
                self.assertAlmostEqual(expected, enumerated, places=10)

    def test_river_solver_accepts_effective_stack_below_big_blind(self):
        game = RiverGame(BOARD, RANGES, pot=3, stack=1, big_blind=2)
        actions = game.legal(RiverState())
        self.assertIn(Action("bet", 1), actions)
        solver = RiverCFR(game).train(5)
        self.assertTrue(solver.analyze_root()["conditional_action_evs"])

    def test_cfr_update_variants_are_finite_and_repeatable(self):
        for variant in ("cfr", "cfr+", "dcfr"):
            first = RiverCFR(RiverGame(BOARD, RANGES), variant=variant).train(40)
            second = RiverCFR(RiverGame(BOARD, RANGES), variant=variant).train(40)
            a = first.analyze_root()
            b = second.analyze_root()
            self.assertEqual(a["action_frequencies"], b["action_frequencies"])
            self.assertTrue(all(math.isfinite(x) for x in a["conditional_action_evs"].values()))
            if variant == "cfr+":
                self.assertTrue(all(regret >= 0 for row in first.regrets.values() for regret in row.values()))


if __name__ == "__main__":
    unittest.main()
