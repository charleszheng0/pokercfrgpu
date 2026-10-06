#include "game.hpp"

#include <stdexcept>

namespace poker {

KuhnState::KuhnState(std::array<int, 2> cards, std::string history)
    : cards_(cards), history_(std::move(history)) {
  if (cards_[0] == cards_[1] || cards_[0] < 0 || cards_[0] > 2 ||
      cards_[1] < 0 || cards_[1] > 2) {
    throw std::invalid_argument("Kuhn cards must be distinct values in [0, 2]");
  }
}

int KuhnState::current_player() const { return history_.size() % 2; }

bool KuhnState::is_terminal() const {
  return history_ == "pp" || history_ == "bp" || history_ == "bb" ||
         history_ == "pbp" || history_ == "pbb";
}

std::vector<Action> KuhnState::legal_actions() const {
  if (is_terminal()) return {};
  if (history_ == "" || history_ == "p" || history_ == "b" || history_ == "pb") {
    return {Action::Pass, Action::Bet};
  }
  throw std::logic_error("invalid Kuhn betting history: " + history_);
}

KuhnState KuhnState::child(Action action) const {
  if (is_terminal()) throw std::logic_error("terminal state has no children");
  return KuhnState(cards_, history_ + action_symbol(action));
}

std::string KuhnState::information_set() const {
  if (is_terminal()) throw std::logic_error("terminal state has no information set");
  return "P" + std::to_string(current_player()) + ":" +
         std::to_string(cards_[current_player()]) + ":" + history_;
}

double KuhnState::utility(int player) const {
  if (!is_terminal()) throw std::logic_error("utility requested for non-terminal state");
  double player0_value = 0.0;
  if (history_ == "bp") {
    player0_value = 1.0;
  } else if (history_ == "pbp") {
    player0_value = -1.0;
  } else {
    const double stake = history_ == "pp" ? 1.0 : 2.0;
    player0_value = cards_[0] > cards_[1] ? stake : -stake;
  }
  return player == 0 ? player0_value : -player0_value;
}

char KuhnState::action_symbol(Action action) {
  return action == Action::Pass ? 'p' : 'b';
}

std::vector<std::array<int, 2>> kuhn_deals() {
  return {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}};
}

}  // namespace poker
