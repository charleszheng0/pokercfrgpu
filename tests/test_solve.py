import unittest

from poker_cpu.engine import Action, HoldemGame
from poker_cpu.river_solver import WeightedRange
from poker_cpu.one_street_solver import OneStreetGame
from poker_cpu.river_solver import RiverCFR, RiverState
from poker_cpu.solve import (ActionAbstraction, BettingContext, SolveRequest,
                             SolverConfig, request_from_game, solve)


class SharedSolverTests(unittest.TestCase):
    @staticmethod
    def ranges():
        return (
            WeightedRange((("Jc", "Tc"), ("8h", "8s")), (1, 1)),
            WeightedRange((("9h", "9s"), ("Ac", "Ad")), (1, 1)),
        )

    def test_showdown_cutoff_matches_preserved_one_street_reference(self):
        hero, opponent = ("Jc", "Tc"), ("9h", "9s")
        ranges = (WeightedRange((hero,), (1,)),
                  WeightedRange((opponent,), (1,)))
        for street, board in (("flop", ("As", "Kh", "Qd")),
                              ("turn", ("As", "Kh", "Qd", "7c"))):
            with self.subTest(street=street):
                request = SolveRequest(
                    board=board, street=street, pot_before_street=8,
                    stacks_at_street_start=(10, 10),
                    betting=BettingContext(button=1, first_actor=0, current_player=0),
                    ranges=ranges, acting_hand=hero,
                    config=SolverConfig(iterations=10,
                                        continuation_mode="showdown_cutoff"),
                )
                actual = solve(request)
                game = OneStreetGame(board, ((hero,), (opponent,)), pot=8, stack=10)
                expected = RiverCFR(game, variant="cfr+", root_state=RiverState()).train(10).analyze_root(private_hand=hero)
                self.assertEqual(actual.status, "supported_restricted_game")
                self.assertAlmostEqual(actual.nash_conv_chips, expected["nash_conv"], places=9)
                for row in actual.selected_hand.actions:
                    self.assertAlmostEqual(row.frequency,
                                           expected["per_hand"][0]["frequency"][row.action], places=9)
                    self.assertAlmostEqual(row.ev_chips,
                                           expected["per_hand"][0]["action_evs"][row.action], places=9)

    def test_river_returns_every_hand_and_conditional_ev_units(self):
        request = SolveRequest(
            board=("As", "Kh", "Qd", "7c", "2s"), street="river",
            pot_before_street=8, stacks_at_street_start=(10, 10),
            betting=BettingContext(button=1, first_actor=0, current_player=0),
            ranges=self.ranges(), config=SolverConfig(iterations=20),
            acting_hand=("Tc", "Jc"), selected_action=Action("check"),
        )
        result = solve(request)
        self.assertEqual(result.status, "supported_restricted_game")
        self.assertEqual(len(result.per_hand), 2)
        self.assertEqual(result.selected_hand.hand, ("Jc", "Tc"))
        self.assertAlmostEqual(sum(row.marginal_probability for row in result.per_hand), 1)
        for strategy in result.per_hand:
            self.assertAlmostEqual(sum(row.frequency for row in strategy.actions), 1)
            for row in strategy.actions:
                self.assertAlmostEqual(row.ev_bb * 2, row.ev_chips)
        self.assertAlmostEqual(sum(row.frequency for row in result.aggregate_actions), 1)
        selected_check = next(row for row in result.selected_hand.actions
                              if row.action == Action("check"))
        self.assertAlmostEqual(result.selected_action_ev_loss_chips,
                               selected_check.ev_loss_chips)
        self.assertGreaterEqual(result.nash_conv_chips, 0)

    def test_turn_reaches_river_betting_through_the_same_api(self):
        ranges = (
            WeightedRange((("Jc", "Tc"),), (1,)),
            WeightedRange((("9h", "9s"),), (1,)),
        )
        result = solve(SolveRequest(
            board=("As", "Kh", "Qd", "7c"), street="turn",
            pot_before_street=8, stacks_at_street_start=(4, 4),
            betting=BettingContext(button=1, first_actor=0, current_player=0),
            ranges=ranges, abstraction=ActionAbstraction((), 0),
            config=SolverConfig(iterations=2), acting_hand=("Jc", "Tc"),
        ))
        self.assertEqual(result.status, "supported_restricted_game")
        self.assertEqual(result.street, "turn")
        self.assertEqual(result.iterations, 2)
        self.assertGreaterEqual(result.nash_conv_chips, 0)

    def test_sampled_turn_result_is_explicitly_experimental(self):
        ranges = (WeightedRange((("8h", "8s"),), (1,)),
                  WeightedRange((("9h", "9s"),), (1,)))
        base = dict(board=("As", "Kh", "Qd", "7c"), street="turn",
                    pot_before_street=8, stacks_at_street_start=(1, 1),
                    betting=BettingContext(button=1, first_actor=0, current_player=0),
                    ranges=ranges, abstraction=ActionAbstraction((), 0))
        result = solve(SolveRequest(**base, config=SolverConfig(
            variant="cfr", iterations=5, chance_sampling=True, sampling_seed=17)))
        self.assertEqual(result.status, "supported_restricted_game")
        self.assertEqual(result.validation_status, "research_sampled_residual_measured")
        self.assertEqual(result.sampling_seed, 17)
        self.assertTrue(result.chance_sampling)
        invalid = solve(SolveRequest(**base, config=SolverConfig(
            variant="cfr+", iterations=5, chance_sampling=True)))
        self.assertEqual(invalid.status, "unsupported")

    def test_live_and_unsupported_states_do_not_expose_advice(self):
        base = dict(board=("As", "Kh", "Qd", "7c", "2s"), street="river",
                    pot_before_street=8, stacks_at_street_start=(10, 10),
                    betting=BettingContext(button=1, first_actor=0, current_player=0))
        live = solve(SolveRequest(**base, ranges=self.ranges(), use_context="live"))
        self.assertEqual(live.status, "unsupported")
        self.assertEqual(live.per_hand, ())
        preflop = solve(SolveRequest(**{**base, "street": "preflop", "board": ()},
                                     ranges=self.ranges()))
        self.assertEqual(preflop.status, "unsupported")
        flop = solve(SolveRequest(**{**base, "street": "flop", "board": ("As", "Kh", "Qd")},
                                  ranges=self.ranges()))
        self.assertEqual(flop.status, "unsupported")

    def test_nashconv_target_is_measured_and_stops_within_the_finite_game(self):
        request = SolveRequest(
            board=("As", "Kh", "Qd", "7c", "2s"), street="river",
            pot_before_street=8, stacks_at_street_start=(10, 10),
            betting=BettingContext(button=1, first_actor=0, current_player=0),
            ranges=self.ranges(), config=SolverConfig(
                iterations=40, target_nash_conv_chips=1_000_000),
        )
        result = solve(request)
        self.assertEqual(result.validation_status, "restricted_target_met")
        self.assertEqual(result.target_nash_conv_chips, 1_000_000)
        self.assertLess(result.iterations, 40)
        self.assertLessEqual(result.nash_conv_chips, result.target_nash_conv_chips)
        self.assertTrue(result.convergence_trace)

    def test_invalid_nashconv_target_is_unsupported(self):
        base = dict(board=("As", "Kh", "Qd", "7c", "2s"), street="river",
                    pot_before_street=8, stacks_at_street_start=(10, 10),
                    betting=BettingContext(button=1, first_actor=0, current_player=0),
                    ranges=self.ranges())
        for target in (-0.1, float("inf"), float("nan")):
            result = solve(SolveRequest(**base, config=SolverConfig(
                iterations=20, target_nash_conv_chips=target)))
            self.assertEqual(result.status, "unsupported")
            self.assertIn("finite nonnegative", result.reason)

    def test_replayed_betting_state_must_match_public_contributions(self):
        report = solve(SolveRequest(
            board=("As", "Kh", "Qd", "7c", "2s"), street="river",
            pot_before_street=8, stacks_at_street_start=(10, 10),
            betting=BettingContext(button=1, first_actor=0, current_player=0,
                                   street_actions=(Action("check"), Action("bet", 4)),
                                   street_paid=(0, 3)),
            ranges=self.ranges(), config=SolverConfig(iterations=1),
        ))
        self.assertEqual(report.status, "unsupported")
        self.assertIn("contributions disagree", report.reason)

    def test_live_request_builder_uses_public_engine_state_and_hero_hand(self):
        game = HoldemGame(seed=41, button=0)
        game.step(Action("call"))
        game.step(Action("check"))
        self.assertEqual(game.street, 1)
        request = request_from_game(game)
        self.assertEqual(request.street, "flop")
        self.assertEqual(request.board, tuple(game.board))
        self.assertEqual(request.betting.first_actor, 1)
        self.assertEqual(request.betting.street_paid, (0, 0))
        self.assertEqual(request.pot_before_street, game.pot)
        self.assertEqual(request.range_source, "unvalidated_inference")
        self.assertIsNone(request.ranges)
        self.assertEqual(solve(request).status, "unsupported")


if __name__ == "__main__":
    unittest.main()
