import unittest

from small_games.kuhn_reference import CFR, INFOSETS, nash_conv, payoff, value


class KuhnReferenceTests(unittest.TestCase):
    def test_terminal_payoffs(self):
        self.assertEqual(payoff("bf", ("J", "K")), 1)
        self.assertEqual(payoff("cbf", ("J", "K")), -1)
        self.assertEqual(payoff("cc", ("J", "K")), -1)
        self.assertEqual(payoff("bk", ("K", "J")), 2)

    def test_exact_best_response_has_no_hidden_card_access(self):
        uniform = {info: (0.5, 0.5) for p in (0, 1) for info in INFOSETS[p]}
        nc, baseline, br0, min0 = nash_conv(uniform, uniform)
        self.assertAlmostEqual(nc, br0 - min0)
        self.assertLessEqual(min0, baseline)
        self.assertGreaterEqual(br0, baseline)
        self.assertAlmostEqual(value(uniform, uniform), baseline)

    def test_cfr_converges_toward_known_kuhn_value(self):
        solver = CFR().train(100)
        early = nash_conv(solver.average(), solver.average())[0]
        solver.train(9900)
        policy = solver.average()
        late, game_value, _, _ = nash_conv(policy, policy)
        self.assertLess(late, 0.01)
        self.assertLess(late, early)
        self.assertAlmostEqual(game_value, -1 / 18, delta=0.005)
        for probs in policy.values():
            self.assertAlmostEqual(sum(probs), 1)
            self.assertTrue(all(0 <= x <= 1 for x in probs))


if __name__ == "__main__":
    unittest.main()
