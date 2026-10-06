#include "cfr.hpp"
#include "cuda_solver.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  int iterations = 100000;
  std::string backend = "cpu";
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
    else if (argument == "--backend" && i + 1 < argc) backend = argv[++i];
    else {
      std::cerr << "Usage: poker_solver [--iterations N] [--backend cpu|cuda]\n";
      return 2;
    }
  }

  try {
    std::unique_ptr<poker::Solver> solver;
    if (backend == "cpu") solver = std::make_unique<poker::CFRSolverCPU>();
    else if (backend == "cuda") solver = std::make_unique<poker::CFRSolverCUDA>();
    else throw std::invalid_argument("backend must be cpu or cuda");
    const auto result = solver->solve(iterations);
    std::cout << std::fixed << std::setprecision(4)
              << "Backend: " << result.backend << "\n"
              << "Iterations: " << result.iterations << "\n"
              << "Time: " << result.elapsed_ms << " ms\n"
              << "States visited: " << result.states_visited << "\n"
              << "States/sec: " << result.states_per_second << "\n"
              << "Game value (P0): " << result.game_value_player0 << "\n"
              << "NashConv: " << result.nash_conv << "\n"
              << "Exploitability: " << result.exploitability << "\n\n"
              << "Average strategy (pass, bet):\n";
    for (const auto& [infoset, strategy] : result.strategy) {
      std::cout << std::setw(9) << infoset << "  " << strategy.pass << "  " << strategy.bet << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
