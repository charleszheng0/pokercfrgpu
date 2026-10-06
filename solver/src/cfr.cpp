#include "cfr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace poker {

namespace {

double probability(const StrategyEntry& entry, Action action) {
  return action == Action::Pass ? entry.pass : entry.bet;
}

StrategyEntry strategy_for(const std::map<std::string, StrategyEntry>& strategy,
                           const KuhnState& state) {
  const auto it = strategy.find(state.information_set());
  return it == strategy.end() ? StrategyEntry{} : it->second;
}

double evaluate_profile(const KuhnState& state,
                        const std::map<std::string, StrategyEntry>& strategy) {
  if (state.is_terminal()) return state.utility(0);
  const auto entry = strategy_for(strategy, state);
  double value = 0.0;
  for (const auto action : state.legal_actions()) {
    value += probability(entry, action) * evaluate_profile(state.child(action), strategy);
  }
  return value;
}

double evaluate_with_policy(const KuhnState& state,
                            const std::map<std::string, StrategyEntry>& opponent,
                            int best_responder,
                            const std::map<std::string, Action>& policy) {
  if (state.is_terminal()) return state.utility(best_responder);
  if (state.current_player() == best_responder) {
    return evaluate_with_policy(state.child(policy.at(state.information_set())), opponent,
                                best_responder, policy);
  }
  const auto entry = strategy_for(opponent, state);
  double value = 0.0;
  for (const auto action : state.legal_actions()) {
    value += probability(entry, action) *
             evaluate_with_policy(state.child(action), opponent, best_responder, policy);
  }
  return value;
}

double best_response_value(const std::map<std::string, StrategyEntry>& strategy, int player) {
  std::vector<std::string> infosets;
  const std::vector<std::string> histories = player == 0
      ? std::vector<std::string>{"", "pb"}
      : std::vector<std::string>{"p", "b"};
  for (int card = 0; card < 3; ++card) {
    for (const auto& history : histories) {
      infosets.push_back("P" + std::to_string(player) + ":" + std::to_string(card) + ":" + history);
    }
  }

  double best = -std::numeric_limits<double>::infinity();
  const std::uint64_t policy_count = std::uint64_t{1} << infosets.size();
  for (std::uint64_t mask = 0; mask < policy_count; ++mask) {
    std::map<std::string, Action> policy;
    for (std::size_t i = 0; i < infosets.size(); ++i) {
      policy[infosets[i]] = (mask & (std::uint64_t{1} << i)) ? Action::Bet : Action::Pass;
    }
    double value = 0.0;
    for (const auto& cards : kuhn_deals()) {
      value += evaluate_with_policy(KuhnState(cards), strategy, player, policy) / 6.0;
    }
    best = std::max(best, value);
  }
  return best;
}

}  // namespace

std::array<double, 2> CFRSolverCPU::Node::current_strategy() const {
  const double pass = std::max(0.0, regret_sum[0]);
  const double bet = std::max(0.0, regret_sum[1]);
  const double normalizer = pass + bet;
  if (normalizer <= 0.0) return {0.5, 0.5};
  return {pass / normalizer, bet / normalizer};
}

StrategyEntry CFRSolverCPU::Node::average_strategy() const {
  const double normalizer = strategy_sum[0] + strategy_sum[1];
  if (normalizer <= 0.0) return {};
  return {strategy_sum[0] / normalizer, strategy_sum[1] / normalizer};
}

CFRSolverCPU::CFRSolverCPU(LeafEvaluator leaf_evaluator)
    : leaf_evaluator_(std::move(leaf_evaluator)) {}

double CFRSolverCPU::traverse(const KuhnState& state, double reach0, double reach1) {
  ++states_visited_;
  if (state.is_terminal()) return state.utility(state.current_player());

  auto& node = nodes_[state.information_set()];
  const auto strategy = node.current_strategy();
  const int player = state.current_player();
  const double own_reach = player == 0 ? reach0 : reach1;
  for (int action = 0; action < 2; ++action) {
    node.strategy_sum[action] += own_reach * strategy[action];
  }

  std::array<double, 2> action_utility{};
  double node_utility = 0.0;
  for (int action = 0; action < 2; ++action) {
    const auto child = state.child(action == 0 ? Action::Pass : Action::Bet);
    action_utility[action] = player == 0
        ? -traverse(child, reach0 * strategy[action], reach1)
        : -traverse(child, reach0, reach1 * strategy[action]);
    node_utility += strategy[action] * action_utility[action];
  }

  const double opponent_reach = player == 0 ? reach1 : reach0;
  for (int action = 0; action < 2; ++action) {
    node.regret_sum[action] += opponent_reach * (action_utility[action] - node_utility);
  }
  return node_utility;
}

std::map<std::string, StrategyEntry> CFRSolverCPU::average_strategy() const {
  std::map<std::string, StrategyEntry> result;
  for (const auto& [key, node] : nodes_) result[key] = node.average_strategy();
  return result;
}

SolverResult CFRSolverCPU::solve(int iterations) {
  if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
  nodes_.clear();
  states_visited_ = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (const auto& cards : kuhn_deals()) traverse(KuhnState(cards), 1.0, 1.0);
  }
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count();

  SolverResult result;
  result.strategy = average_strategy();
  result.iterations = iterations;
  result.elapsed_ms = seconds * 1000.0;
  result.states_visited = states_visited_;
  result.states_per_second = seconds > 0.0 ? states_visited_ / seconds : 0.0;
  result.iterations_per_second = seconds > 0.0 ? iterations / seconds : 0.0;
  result.game_value_player0 = expected_value(result.strategy);
  result.nash_conv = poker::nash_conv(result.strategy);
  result.exploitability = result.nash_conv / 2.0;
  return result;
}

double expected_value(const std::map<std::string, StrategyEntry>& strategy) {
  double value = 0.0;
  for (const auto& cards : kuhn_deals()) {
    value += evaluate_profile(KuhnState(cards), strategy) / 6.0;
  }
  return value;
}

double nash_conv(const std::map<std::string, StrategyEntry>& strategy) {
  return best_response_value(strategy, 0) + best_response_value(strategy, 1);
}

}  // namespace poker
