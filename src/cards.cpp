#include "cards.hpp"

#include <stdexcept>

namespace poker {

static const char kRanks[] = "23456789TJQKA";
static const char kSuits[] = "cdhs";

int parse_rank(char ch) {
  if (ch >= 'a' && ch <= 'z') ch = static_cast<char>(ch - 'a' + 'A');
  for (int r = 0; r < 13; ++r)
    if (kRanks[r] == ch) return r;
  return -1;
}

int parse_card(const std::string& s) {
  if (s.size() != 2) return -1;
  int r = parse_rank(s[0]);
  int su = -1;
  char c = static_cast<char>(s[1] | 0x20);  // lower case
  for (int i = 0; i < 4; ++i)
    if (kSuits[i] == c) su = i;
  if (r < 0 || su < 0) return -1;
  return make_card(r, su);
}

std::string card_str(Card c) { return {kRanks[rank_of(c)], kSuits[suit_of(c)]}; }

std::vector<Card> parse_cards(const std::string& text) {
  std::string s;
  for (char ch : text)
    if (ch != ' ' && ch != ',') s += ch;
  if (s.size() % 2 != 0) throw std::invalid_argument("cards must be pairs like 'Ah Kd': " + text);
  std::vector<Card> out;
  uint64_t seen = 0;
  for (size_t i = 0; i < s.size(); i += 2) {
    int c = parse_card(s.substr(i, 2));
    if (c < 0) throw std::invalid_argument("bad card '" + s.substr(i, 2) + "'");
    if (seen >> c & 1) throw std::invalid_argument("duplicate card " + s.substr(i, 2));
    seen |= 1ULL << c;
    out.push_back(static_cast<Card>(c));
  }
  return out;
}

std::string cards_str(const std::vector<Card>& cards) {
  std::string s;
  for (Card c : cards) s += card_str(c);
  return s;
}

namespace {

inline uint32_t make_value(int category, uint32_t v) { return (static_cast<uint32_t>(category) << 20) | v; }

// Highest card of a 5-in-a-row inside a 13-bit rank mask, or -1. Wheel (A2345) gives 3.
int straight_high(uint32_t m) {
  for (int hi = 12; hi >= 4; --hi)
    if (((m >> (hi - 4)) & 0x1F) == 0x1F) return hi;
  if ((m & 0x100F) == 0x100F) return 3;
  return -1;
}

// The k highest ranks of a mask, packed 4 bits each.
uint32_t top_ranks(uint32_t m, int k) {
  uint32_t v = 0;
  for (int r = 12; r >= 0 && k > 0; --r)
    if (m >> r & 1) {
      v = (v << 4) | static_cast<uint32_t>(r);
      --k;
    }
  return v;
}

}  // namespace

uint32_t eval_hand(const Card* cards, int n) {
  uint32_t suit_mask[4] = {0, 0, 0, 0};
  int count[13] = {0};
  uint32_t all = 0;
  for (int i = 0; i < n; ++i) {
    int r = rank_of(cards[i]);
    suit_mask[suit_of(cards[i])] |= 1u << r;
    ++count[r];
    all |= 1u << r;
  }
  for (int s = 0; s < 4; ++s) {
    if (__builtin_popcount(suit_mask[s]) >= 5) {
      int sh = straight_high(suit_mask[s]);
      if (sh >= 0) return make_value(8, static_cast<uint32_t>(sh));
      return make_value(5, top_ranks(suit_mask[s], 5));
    }
  }
  int quad = -1, trips[2] = {-1, -1}, pairs[3] = {-1, -1, -1};
  int nt = 0, np = 0;
  for (int r = 12; r >= 0; --r) {
    if (count[r] == 4) quad = r;
    else if (count[r] == 3 && nt < 2) trips[nt++] = r;
    else if (count[r] == 2 && np < 3) pairs[np++] = r;
  }
  if (quad >= 0)
    return make_value(7, static_cast<uint32_t>(quad) << 4 | top_ranks(all & ~(1u << quad), 1));
  if (nt >= 1 && (nt >= 2 || np >= 1)) {
    int pair = nt >= 2 ? trips[1] : -1;
    if (np >= 1 && pairs[0] > pair) pair = pairs[0];
    return make_value(6, static_cast<uint32_t>(trips[0]) << 4 | static_cast<uint32_t>(pair));
  }
  int sh = straight_high(all);
  if (sh >= 0) return make_value(4, static_cast<uint32_t>(sh));
  if (nt == 1)
    return make_value(3, static_cast<uint32_t>(trips[0]) << 8 | top_ranks(all & ~(1u << trips[0]), 2));
  if (np >= 2) {
    uint32_t rest = all & ~(1u << pairs[0]) & ~(1u << pairs[1]);
    return make_value(2, static_cast<uint32_t>(pairs[0]) << 8 | static_cast<uint32_t>(pairs[1]) << 4 |
                             top_ranks(rest, 1));
  }
  if (np == 1)
    return make_value(1, static_cast<uint32_t>(pairs[0]) << 12 | top_ranks(all & ~(1u << pairs[0]), 3));
  return make_value(0, top_ranks(all, 5));
}

const char* category_name(int category) {
  static const char* names[] = {"high card", "pair",  "two pair",   "trips",         "straight",
                                "flush",     "full house", "quads", "straight flush"};
  return (category >= 0 && category <= 8) ? names[category] : "?";
}

}  // namespace poker
