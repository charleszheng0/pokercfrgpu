import unittest

from small_games.compare_cfr_nfsp import (
    DEALS,
    SynchronousCFR,
    advance,
    initial,
    legal,
    nash_conv,
    terminal_payoff,
)


class LeducCFRTests(unittest.TestCase):
    def test_sync_cfr_reduces_exact_nash_conv(self):
        solver = SynchronousCFR()
        for _ in range(20):
            solver.step()
        nc, _, _, _ = nash_conv(solver.average, solver.average)
        self.assertLess(nc, 2.0)
        self.assertEqual(solver.iterations, 20)
        for sums in solver.sum_policy.values():
            self.assertGreater(sum(sums.values()), 0)

    def test_terminal_rewards_are_zero_sum(self):
        # A simple terminal continuation on every physical deal.
        for deal in DEALS:
            state = initial(deal)
            state = advance(state, "bet")
            self.assertIn("fold", legal(state))
            state = advance(state, "fold")
            self.assertEqual(terminal_payoff(state, 0) + terminal_payoff(state, 1), 0)


if __name__ == "__main__":
    unittest.main()
