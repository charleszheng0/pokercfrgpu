#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace poker {

enum class Action : std::uint8_t { Pass = 0, Bet = 1 };

// Kuhn poker: three cards, one private card per player, and one betting round.
class KuhnState {
 public:
  KuhnState(std::array<int, 2> cards, std::string history = {});

  int current_player() const;
  bool is_terminal() const;
  std::vector<Action> legal_actions() const;
  KuhnState child(Action action) const;
  std::string information_set() const;
  double utility(int player) const;

  const std::array<int, 2>& cards() const { return cards_; }
  const std::string& history() const { return history_; }
  static char action_symbol(Action action);

 private:
  std::array<int, 2> cards_;
  std::string history_;
};

std::vector<std::array<int, 2>> kuhn_deals();

}  // namespace poker
