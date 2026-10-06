#include "cfr.hpp"
#include "game.hpp"

#include <cmath>
#include <iostream>

int main() {
  int failures = 0;
  const poker::KuhnState showdown({2, 0}, "bb");
  if (showdown.utility(0) != 2.0 || showdown.utility(1) != -2.0) {
    std::cerr << "showdown utility test failed\n";
    ++failures;
  }
  const poker::KuhnState fold({0, 2}, "bp");
  if (fold.utility(0) != 1.0) {
    std::cerr << "fold utility test failed\n";
    ++failures;
  }

  poker::CFRSolverCPU solver;
  const auto result = solver.solve(100000);
  // Kuhn's equilibrium value for the first player is -1/18.
  if (std::abs(result.game_value_player0 + 1.0 / 18.0) > 0.015) {
    std::cerr << "value did not converge: " << result.game_value_player0 << "\n";
    ++failures;
  }
  if (result.nash_conv > 0.03) {
    std::cerr << "NashConv too high: " << result.nash_conv << "\n";
    ++failures;
  }
  if (result.strategy.size() != 12) {
    std::cerr << "expected 12 information sets, got " << result.strategy.size() << "\n";
    ++failures;
  }
  return failures == 0 ? 0 : 1;
}
