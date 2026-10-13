import unittest

from poker_cpu.solver_interface import SolverRequest, UnsupportedHoldemSolver


class SolverInterfaceTests(unittest.TestCase):
    def test_unsupported_solver_never_fabricates_advice(self):
        request = SolverRequest(
            public_state={"street": 1},
            acting_player=0,
            ranges=None,
            legal_action_abstraction=(),
            stacks=(100, 100),
            pot=4,
            betting_history=(),
            computation_budget_ms=100,
        )
        result = UnsupportedHoldemSolver().analyze(request)
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.validation_status, "not_validated")
        self.assertIsNone(result.action_frequencies)
        self.assertIsNone(result.action_evs)
        self.assertIsNone(result.ev_loss)
        self.assertIsNone(result.iterations)
        self.assertIsNone(result.latency_ms)
        self.assertIsNone(result.convergence)
        self.assertIsNone(result.confidence)


if __name__ == "__main__":
    unittest.main()
