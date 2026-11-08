// Hand ranges: a weight in [0, 1] for each of the 1,326 two-card combos.
#pragma once
#include <array>
#include <string>
#include <utility>

#include "cards.hpp"

namespace poker {

constexpr int kNumCombos = 1326;

struct Range {
  std::array<float, kNumCombos> weight{};
  int num_combos() const;  // combos with weight > 0
};

int combo_index(Card a, Card b);              // order does not matter
std::pair<Card, Card> combo_cards(int index);  // first card has the higher rank
std::string combo_str(int index);              // e.g. "AhKd"

// 13x13 grid cell of a combo. Row/col 0 = ace. Upper right = suited, lower left = offsuit.
int grid_cell(int index);
std::string grid_label(int cell);  // "AA", "AKs", "AKo"

// Standard notation: "AA, KK:0.5, AKs, AQo+, 22+, A5s-A2s, T9s, AhKh".
// Later tokens overwrite earlier ones. Throws std::invalid_argument with a clear message.
Range parse_range(const std::string& text);

}  // namespace poker
