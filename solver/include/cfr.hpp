#pragma once

#include "game.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace poker {

struct StrategyEntry {
  double pass = 0.5;
  double bet = 0.5;
};

struct SolverResult {
  std::map<std::string, StrategyEntry> strategy;
  int iterations = 0;
  double elapsed_ms = 0.0;
  std::uint64_t states_visited = 0;
  double states_per_second = 0.0;
  double iterations_per_second = 0.0;
  double game_value_player0 = 0.0;
  double nash_conv = 0.0;
  double exploitability = 0.0;
  std::string backend = "cpu";
};

class Solver {
 public:
  virtual ~Solver() = default;
  virtual SolverResult solve(int iterations) = 0;
};

class CFRSolverCPU final : public Solver {
 public:
  // Reserved for depth-limited solving; unset means traverse to terminal states.
  using LeafEvaluator = std::function<double(const KuhnState&, int)>;

  explicit CFRSolverCPU(LeafEvaluator leaf_evaluator = {});
  SolverResult solve(int iterations) override;

 private:
  struct Node {
    std::array<double, 2> regret_sum{0.0, 0.0};
    std::array<double, 2> strategy_sum{0.0, 0.0};

    std::array<double, 2> current_strategy() const;
    StrategyEntry average_strategy() const;
  };

  double traverse(const KuhnState& state, double reach0, double reach1);
  std::map<std::string, StrategyEntry> average_strategy() const;

  std::map<std::string, Node> nodes_;
  std::uint64_t states_visited_ = 0;
  LeafEvaluator leaf_evaluator_;
};

double expected_value(const std::map<std::string, StrategyEntry>& strategy);
double nash_conv(const std::map<std::string, StrategyEntry>& strategy);

}  // namespace poker
