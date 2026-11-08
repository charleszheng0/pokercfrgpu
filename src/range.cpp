#include "range.hpp"

#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace poker {

namespace {

struct ComboTable {
  int index[52][52];
  std::pair<Card, Card> cards[kNumCombos];
  ComboTable() {
    int k = 0;
    for (int a = 0; a < 52; ++a)
      for (int b = a + 1; b < 52; ++b) {
        index[a][b] = index[b][a] = k;
        // put the higher card first for display
        cards[k] = a > b ? std::make_pair(Card(a), Card(b)) : std::make_pair(Card(b), Card(a));
        ++k;
      }
    for (int a = 0; a < 52; ++a) index[a][a] = -1;
  }
};

const ComboTable& table() {
  static const ComboTable t;
  return t;
}

enum class Kind { Pair, Suited, Offsuit, Any };

struct HandClass {
  int hi, lo;  // ranks, hi >= lo
  Kind kind;
};

void add_class(std::vector<int>& out, const HandClass& h) {
  for (int s1 = 0; s1 < 4; ++s1)
    for (int s2 = 0; s2 < 4; ++s2) {
      if (h.kind == Kind::Pair) {
        if (s1 < s2) out.push_back(combo_index(make_card(h.hi, s1), make_card(h.hi, s2)));
        continue;
      }
      bool suited = s1 == s2;
      if (h.kind == Kind::Suited && !suited) continue;
      if (h.kind == Kind::Offsuit && suited) continue;
      out.push_back(combo_index(make_card(h.hi, s1), make_card(h.lo, s2)));
    }
}

// Parse "AK", "AKs", "AKo", "TT" (no + or -).
HandClass parse_class(const std::string& t, const std::string& token) {
  if (t.size() < 2 || t.size() > 3) throw std::invalid_argument("cannot read hand '" + token + "'");
  int a = parse_rank(t[0]), b = parse_rank(t[1]);
  if (a < 0 || b < 0) throw std::invalid_argument("bad rank in '" + token + "'");
  HandClass h{a > b ? a : b, a > b ? b : a, Kind::Any};
  if (a == b) {
    if (t.size() == 3) throw std::invalid_argument("a pair cannot be suited or offsuit: '" + token + "'");
    h.kind = Kind::Pair;
  } else if (t.size() == 3) {
    char s = static_cast<char>(t[2] | 0x20);
    if (s == 's') h.kind = Kind::Suited;
    else if (s == 'o') h.kind = Kind::Offsuit;
    else throw std::invalid_argument("expected 's' or 'o' in '" + token + "'");
  }
  return h;
}

std::vector<int> expand_token(const std::string& t, const std::string& token) {
  std::vector<int> out;
  if (t.size() == 4) {
    int c1 = parse_card(t.substr(0, 2)), c2 = parse_card(t.substr(2, 2));
    if (c1 >= 0 && c2 >= 0) {
      if (c1 == c2) throw std::invalid_argument("same card twice in '" + token + "'");
      out.push_back(combo_index(Card(c1), Card(c2)));
      return out;
    }
  }
  size_t dash = t.find('-');
  if (dash != std::string::npos) {
    HandClass a = parse_class(t.substr(0, dash), token), b = parse_class(t.substr(dash + 1), token);
    if (a.kind != b.kind) throw std::invalid_argument("both ends must be the same type in '" + token + "'");
    if (a.kind == Kind::Pair) {
      int lo = std::min(a.hi, b.hi), hi = std::max(a.hi, b.hi);
      for (int r = lo; r <= hi; ++r) add_class(out, {r, r, Kind::Pair});
    } else {
      if (a.hi != b.hi) throw std::invalid_argument("first rank must match in '" + token + "'");
      int lo = std::min(a.lo, b.lo), hi = std::max(a.lo, b.lo);
      for (int r = lo; r <= hi; ++r) add_class(out, {a.hi, r, a.kind});
    }
    return out;
  }
  if (!t.empty() && t.back() == '+') {
    HandClass h = parse_class(t.substr(0, t.size() - 1), token);
    if (h.kind == Kind::Pair) {
      for (int r = h.hi; r <= 12; ++r) add_class(out, {r, r, Kind::Pair});
    } else {
      for (int r = h.lo; r < h.hi; ++r) add_class(out, {h.hi, r, h.kind});
    }
    return out;
  }
  add_class(out, parse_class(t, token));
  return out;
}

}  // namespace

int Range::num_combos() const {
  int n = 0;
  for (float w : weight) n += w > 0.f;
  return n;
}

int combo_index(Card a, Card b) { return table().index[a][b]; }
std::pair<Card, Card> combo_cards(int index) { return table().cards[index]; }
std::string combo_str(int index) {
  auto c = combo_cards(index);
  return card_str(c.first) + card_str(c.second);
}

int grid_cell(int index) {
  auto c = combo_cards(index);
  int hi = rank_of(c.first), lo = rank_of(c.second);
  int row_hi = 12 - hi, row_lo = 12 - lo;
  if (hi == lo) return row_hi * 13 + row_hi;
  if (suit_of(c.first) == suit_of(c.second)) return row_hi * 13 + row_lo;  // suited: upper right
  return row_lo * 13 + row_hi;                                              // offsuit: lower left
}

std::string grid_label(int cell) {
  static const char kR[] = "AKQJT98765432";
  int row = cell / 13, col = cell % 13;
  if (row == col) return {kR[row], kR[row]};
  if (row < col) return {kR[row], kR[col], 's'};
  return {kR[col], kR[row], 'o'};
}

Range parse_range(const std::string& text) {
  Range r;
  std::string token;
  auto flush = [&]() {
    if (token.empty()) return;
    std::string t = token;
    float w = 1.f;
    size_t colon = t.find(':');
    if (colon != std::string::npos) {
      std::string ws = t.substr(colon + 1);
      char* end = nullptr;
      double v = std::strtod(ws.c_str(), &end);
      if (ws.empty() || *end != '\0' || v < 0 || v > 1)
        throw std::invalid_argument("weight must be between 0 and 1 in '" + token + "'");
      w = static_cast<float>(v);
      t = t.substr(0, colon);
    }
    for (int idx : expand_token(t, token)) r.weight[idx] = w;
    token.clear();
  };
  for (char ch : text) {
    if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') flush();
    else token += ch;
  }
  flush();
  return r;
}

}  // namespace poker
