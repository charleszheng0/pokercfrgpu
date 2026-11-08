#include "game.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace poker {

namespace {

constexpr double kEps = 1e-9;
constexpr size_t kMaxNodes = 30'000'000;

// DCFR parameters from Brown & Sandholm (2019): alpha = 1.5, beta = 0, gamma = 2.
constexpr double kAlpha = 1.5, kBeta = 0.0, kGamma = 2.0;

std::string fmt(double v, int prec = 2) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", prec, v);
  std::string s = buf;
  if (s.find('.') != std::string::npos) {
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
  }
  if (s == "-0") s = "0";
  return s;
}

std::string json_num(double v, int prec = 4) {
  if (!std::isfinite(v)) return "0";
  return fmt(v, prec);
}

inline bool hand_has(const HandInfo& h, Card c) { return h.c1 == c || h.c2 == c; }

// The 24 permutations of the four suits, in lexicographic order (id 0 = identity).
struct SuitPerms {
  int8_t p[24][4];
  int inv[24];
};
const SuitPerms& suit_perms() {
  static const SuitPerms sp = [] {
    SuitPerms s{};
    int a[4] = {0, 1, 2, 3}, k = 0;
    do {
      for (int i = 0; i < 4; ++i) s.p[k][i] = static_cast<int8_t>(a[i]);
      ++k;
    } while (std::next_permutation(a, a + 4));
    for (int x = 0; x < 24; ++x)
      for (int y = 0; y < 24; ++y) {
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok &= s.p[y][s.p[x][i]] == i;
        if (ok) s.inv[x] = y;
      }
    return s;
  }();
  return sp;
}
inline Card perm_card(int k, Card c) { return make_card(rank_of(c), suit_perms().p[k][suit_of(c)]); }
// id of the permutation "apply b, then a"
int compose(int a, int b) {
  const SuitPerms& sp = suit_perms();
  for (int k = 0; k < 24; ++k) {
    bool ok = true;
    for (int i = 0; i < 4; ++i) ok &= sp.p[k][i] == sp.p[a][sp.p[b][i]];
    if (ok) return k;
  }
  return 0;
}

}  // namespace

// ---------------------------------------------------------------- tree build

Game::Game(const GameConfig& cfg) : cfg_(cfg) {
  const int bs = static_cast<int>(cfg_.board.size());
  if (bs < 3 || bs > 5) throw std::invalid_argument("board must have 3, 4 or 5 cards");
  if (!(cfg_.pot > 0)) throw std::invalid_argument("pot must be positive");
  if (cfg_.stack < 0) throw std::invalid_argument("stack must not be negative");

  uint64_t board_mask = 0;
  for (Card c : cfg_.board) board_mask |= 1ULL << c;
  for (int p = 0; p < 2; ++p) {
    for (int i = 0; i < kNumCombos; ++i) {
      float w = cfg_.range[p].weight[i];
      if (w <= 0) continue;
      auto cc = combo_cards(i);
      if ((board_mask >> cc.first & 1) || (board_mask >> cc.second & 1)) continue;
      hands_[p].push_back({cc.first, cc.second, w, static_cast<uint16_t>(i)});
    }
    if (hands_[p].empty())
      throw std::invalid_argument(std::string(p == 0 ? "OOP" : "IP") +
                                  " range is empty after removing board cards");
  }
  for (int p = 0; p < 2; ++p) {
    std::vector<int32_t> pos(kNumCombos, -1);
    for (size_t j = 0; j < hands_[1 - p].size(); ++j) pos[hands_[1 - p][j].combo] = static_cast<int32_t>(j);
    same_combo_[p].resize(hands_[p].size());
    for (size_t i = 0; i < hands_[p].size(); ++i) same_combo_[p][i] = pos[hands_[p][i].combo];
  }
  if (pair_weight_sum() <= 0) throw std::invalid_argument("the two ranges never fit together (card conflicts)");
  prepare_symmetry();

  boards_.push_back(cfg_.board);
  BuildState st{0, bs, {cfg_.pot / 2, cfg_.pot / 2}, 0, 0};
  nodes_.emplace_back();
  build_decision(0, st);

  for (auto& n : nodes_) {
    if (n.type != NodeType::Action) continue;
    n.data = arena_size_;
    n.slot = static_cast<uint32_t>(num_action_nodes_);
    arena_size_ += static_cast<uint64_t>(n.num_children) * hands_[n.player].size();
    ++num_action_nodes_;
  }
  compressed_ = cfg_.compress == 1;
  if (cfg_.compress < 0) {
    compressed_ = false;
    if (memory_bytes() / 1e9 > cfg_.mem_limit_gb) compressed_ = true;  // 32-bit does not fit: try 16-bit
  }
  const double gb = memory_bytes() / 1e9;
  if (gb > cfg_.mem_limit_gb) {
    std::ostringstream os;
    os << "tree needs " << fmt(gb, 2) << " GB" << (compressed_ ? " even with 16-bit storage" : "") << " (limit "
       << fmt(cfg_.mem_limit_gb, 2) << " GB). Use fewer bet sizes, smaller ranges, or a later street.";
    throw std::runtime_error(os.str());
  }
  if (compressed_) {
    regrets16_.assign(arena_size_, 0);
    strategy_sum16_.assign(arena_size_, 0);
    regret_scale_.assign(num_action_nodes_, 0.f);
    strategy_scale_.assign(num_action_nodes_, 0.f);
  } else {
    regrets_.assign(arena_size_, 0.f);
    strategy_sum_.assign(arena_size_, 0.f);
  }
  prepare_showdowns();

  int threads = cfg_.threads > 0 ? cfg_.threads : static_cast<int>(std::thread::hardware_concurrency());
  pool_ = std::make_unique<ThreadPool>(std::max(1, threads));
}

double Game::memory_bytes() const {
  if (compressed_)
    return static_cast<double>(arena_size_) * 2 * sizeof(int16_t) + static_cast<double>(num_action_nodes_) * 2 * sizeof(float);
  return static_cast<double>(arena_size_) * 2 * sizeof(float);
}

// Find the suit permutations that leave both ranges and the starting board unchanged.
// Only these can make two dealt cards equivalent.
void Game::prepare_symmetry() {
  uint64_t board_mask = 0;
  for (Card c : cfg_.board) board_mask |= 1ULL << c;
  for (int k = 0; k < 24; ++k) {
    bool ok = k == 0 || cfg_.isomorphism;
    for (Card c : cfg_.board) ok = ok && (board_mask >> perm_card(k, c) & 1);
    for (int p = 0; p < 2 && ok; ++p)
      for (int i = 0; i < kNumCombos && ok; ++i) {
        auto cc = combo_cards(i);
        ok = cfg_.range[p].weight[i] == cfg_.range[p].weight[combo_index(perm_card(k, cc.first), perm_card(k, cc.second))];
      }
    usable_[k] = ok;
    if (!ok) continue;
    for (int p = 0; p < 2; ++p) {
      std::vector<int32_t> pos(kNumCombos, -1);
      for (size_t h = 0; h < hands_[p].size(); ++h) pos[hands_[p][h].combo] = static_cast<int32_t>(h);
      hand_perm_[k][p].resize(hands_[p].size());
      for (size_t h = 0; h < hands_[p].size(); ++h) {
        const int32_t j = pos[combo_index(perm_card(k, hands_[p][h].c1), perm_card(k, hands_[p][h].c2))];
        if (j < 0) throw std::logic_error("suit symmetry: missing image hand");
        hand_perm_[k][p][h] = static_cast<uint16_t>(j);
      }
    }
  }
}

// Usable permutations that also leave this board unchanged (as a set).
std::vector<int> Game::symmetry_group(const std::vector<Card>& board) const {
  uint64_t mask = 0;
  for (Card c : board) mask |= 1ULL << c;
  std::vector<int> g;
  for (int k = 0; k < 24; ++k) {
    if (!usable_[k]) continue;
    bool ok = true;
    for (Card c : board) ok = ok && (mask >> perm_card(k, c) & 1);
    if (ok) g.push_back(k);
  }
  return g;
}

std::vector<ActionInfo> Game::legal_actions(const BuildState& st) const {
  const int a = st.actor, o = 1 - a;
  const int street = st.board_size - 3;
  const double half = cfg_.pot / 2;
  const double rem_a = cfg_.stack + half - st.contrib[a];
  const double rem_o = cfg_.stack + half - st.contrib[o];
  const double to_call = st.contrib[o] - st.contrib[a];
  const double pot_now = st.contrib[0] + st.contrib[1];
  std::vector<ActionInfo> acts;
  std::vector<double> sizes;  // chips added by the actor

  auto push_size = [&](double add) {
    if (add >= rem_a - kEps || add >= cfg_.allin_threshold * rem_a) add = rem_a;
    sizes.push_back(add);
  };

  if (to_call <= kEps) {
    acts.push_back({ActKind::Check, 0});
    if (rem_a > kEps && rem_o > kEps) {
      for (double f : cfg_.sizes[a][street].bet) push_size(f * pot_now);
      if (cfg_.add_allin) sizes.push_back(rem_a);
    }
  } else {
    acts.push_back({ActKind::Fold, 0});
    acts.push_back({ActKind::Call, std::min(to_call, rem_a)});
    if (st.raises < cfg_.max_raises && rem_a > to_call + kEps && rem_o > kEps) {
      for (double f : cfg_.sizes[a][street].raise) {
        double inc = f * (pot_now + to_call);
        push_size(to_call + std::max(inc, to_call));  // min raise = size of the last bet
      }
      if (cfg_.add_allin) sizes.push_back(rem_a);
    }
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end(), [](double x, double y) { return std::fabs(x - y) < 1e-6; }),
              sizes.end());
  for (double s : sizes) {
    ActKind k = s >= rem_a - kEps ? ActKind::AllIn : (to_call <= kEps ? ActKind::Bet : ActKind::Raise);
    acts.push_back({k, s});
  }
  return acts;
}

void Game::build_decision(uint32_t idx, const BuildState& st) {
  if (nodes_.size() > kMaxNodes)
    throw std::runtime_error("tree is too large (over 30M nodes). Use fewer bet sizes or a later street.");
  const int a = st.actor, o = 1 - a;
  std::vector<ActionInfo> acts = legal_actions(st);
  {
    Node& n = nodes_[idx];
    n.type = NodeType::Action;
    n.player = static_cast<uint8_t>(a);
    n.board_size = static_cast<uint8_t>(st.board_size);
    n.board_id = st.board_id;
    n.contrib[0] = st.contrib[0];
    n.contrib[1] = st.contrib[1];
    n.num_children = static_cast<uint32_t>(acts.size());
    n.first_child = static_cast<uint32_t>(nodes_.size());
    n.first_action = static_cast<uint32_t>(actions_.size());
  }
  const uint32_t first = static_cast<uint32_t>(nodes_.size());
  actions_.insert(actions_.end(), acts.begin(), acts.end());
  nodes_.resize(nodes_.size() + acts.size());
  const double to_call = st.contrib[o] - st.contrib[a];

  for (size_t i = 0; i < acts.size(); ++i) {
    const uint32_t child = first + static_cast<uint32_t>(i);
    BuildState ns = st;
    switch (acts[i].kind) {
      case ActKind::Fold: {
        Node& n = nodes_[child];
        n.type = NodeType::Fold;
        n.player = static_cast<uint8_t>(a);
        n.board_size = static_cast<uint8_t>(st.board_size);
        n.board_id = st.board_id;
        n.contrib[0] = st.contrib[0];
        n.contrib[1] = st.contrib[1];
        break;
      }
      case ActKind::Check:
        if (a == 0) {  // OOP checks: IP acts next
          ns.actor = 1;
          build_decision(child, ns);
        } else {  // IP checks back: street ends
          build_street_end(child, ns);
        }
        break;
      case ActKind::Call:
        ns.contrib[a] += acts[i].amount;
        build_street_end(child, ns);
        break;
      default:  // Bet, Raise, AllIn
        ns.contrib[a] += acts[i].amount;
        if (to_call > kEps) ns.raises = st.raises + 1;
        ns.actor = o;
        build_decision(child, ns);
        break;
    }
  }
}

void Game::build_street_end(uint32_t idx, const BuildState& st) {
  {
    Node& n = nodes_[idx];
    n.board_size = static_cast<uint8_t>(st.board_size);
    n.board_id = st.board_id;
    n.contrib[0] = st.contrib[0];
    n.contrib[1] = st.contrib[1];
    if (st.board_size == 5) {
      n.type = NodeType::Showdown;
      return;
    }
    n.type = NodeType::Chance;
  }
  const double half = cfg_.pot / 2;
  const bool all_in = cfg_.stack + half - st.contrib[0] <= kEps || cfg_.stack + half - st.contrib[1] <= kEps;
  const std::vector<Card> board = boards_[st.board_id];
  uint64_t used = 0;
  for (Card c : board) used |= 1ULL << c;
  // Group the dealable cards by suit symmetry. Only the lowest card of each group (the
  // representative) gets a subtree; the others reuse it through a suit permutation.
  const std::vector<int> group = symmetry_group(board);
  std::vector<Card> deal;  // representatives
  std::vector<ChanceEntry> entries;
  for (int c = 0; c < kNumCards; ++c) {
    if (used >> c & 1) continue;
    Card rep = static_cast<Card>(c);
    for (int k : group) rep = std::min(rep, perm_card(k, static_cast<Card>(c)));
    if (rep == c) deal.push_back(rep);
  }
  for (int c = 0; c < kNumCards; ++c) {
    if (used >> c & 1) continue;
    Card rep = static_cast<Card>(c);
    for (int k : group) rep = std::min(rep, perm_card(k, static_cast<Card>(c)));
    int perm = -1;
    for (int k : group)
      if (perm_card(k, rep) == c) {
        perm = k;
        break;
      }
    const auto child = std::lower_bound(deal.begin(), deal.end(), rep) - deal.begin();
    entries.push_back({static_cast<Card>(c), static_cast<uint8_t>(perm), static_cast<uint16_t>(child)});
  }
  chance_total_ += entries.size();
  chance_solved_ += deal.size();

  const uint32_t first = static_cast<uint32_t>(nodes_.size());
  nodes_[idx].num_children = static_cast<uint32_t>(deal.size());
  nodes_[idx].first_child = first;
  nodes_[idx].first_action = static_cast<uint32_t>(chance_entries_.size());
  chance_entries_.insert(chance_entries_.end(), entries.begin(), entries.end());
  nodes_.resize(nodes_.size() + deal.size());
  for (size_t i = 0; i < deal.size(); ++i) {
    const uint32_t child = first + static_cast<uint32_t>(i);
    std::vector<Card> nb = board;
    nb.push_back(deal[i]);
    boards_.push_back(nb);
    BuildState ns = st;
    ns.board_id = static_cast<uint32_t>(boards_.size() - 1);
    ns.board_size = st.board_size + 1;
    ns.actor = 0;
    ns.raises = 0;
    if (all_in) build_street_end(child, ns);
    else build_decision(child, ns);
    nodes_[child].dealt = deal[i];
  }
}

void Game::prepare_showdowns() {
  showdown_index_.assign(boards_.size(), -1);
  for (size_t b = 0; b < boards_.size(); ++b) {
    if (boards_[b].size() != 5) continue;
    showdown_index_[b] = static_cast<int32_t>(showdowns_.size());
    showdowns_.emplace_back();
  }
  // Each river board is independent: fill them in parallel.
  std::vector<size_t> river_boards;
  for (size_t b = 0; b < boards_.size(); ++b)
    if (showdown_index_[b] >= 0) river_boards.push_back(b);
  int threads = cfg_.threads > 0 ? cfg_.threads : static_cast<int>(std::thread::hardware_concurrency());
  ThreadPool pool(std::max(1, threads));
  pool.parallel_for(static_cast<int>(river_boards.size()), [&](int k) {
    const auto& board = boards_[river_boards[k]];
    Showdown& sd = showdowns_[showdown_index_[river_boards[k]]];
    uint64_t used = 0;
    for (Card c : board) used |= 1ULL << c;
    for (int p = 0; p < 2; ++p) {
      std::vector<std::pair<uint32_t, uint16_t>> v;
      Card cards[7];
      for (int i = 0; i < 5; ++i) cards[i] = board[i];
      for (size_t h = 0; h < hands_[p].size(); ++h) {
        const HandInfo& hi = hands_[p][h];
        if ((used >> hi.c1 & 1) || (used >> hi.c2 & 1)) continue;
        cards[5] = hi.c1;
        cards[6] = hi.c2;
        v.emplace_back(eval_hand(cards, 7), static_cast<uint16_t>(h));
      }
      std::sort(v.begin(), v.end());
      for (auto& e : v) {
        sd.strength[p].push_back(e.first);
        sd.order[p].push_back(e.second);
      }
    }
  });
}

// ---------------------------------------------------------------- strategies

// 16-bit layout: each node keeps one scale per array; value = q * scale, where the
// largest magnitude in the node maps to the largest integer.
void Game::load_regrets(const Node& n, float* dst) const {
  const size_t len = static_cast<size_t>(n.num_children) * hands_[n.player].size();
  if (!compressed_) {
    std::copy_n(&regrets_[n.data], len, dst);
    return;
  }
  const float sc = regret_scale_[n.slot];
  const int16_t* q = &regrets16_[n.data];
  for (size_t i = 0; i < len; ++i) dst[i] = static_cast<float>(q[i]) * sc;
}

void Game::store_regrets(const Node& n, const float* src) {
  const size_t len = static_cast<size_t>(n.num_children) * hands_[n.player].size();
  if (!compressed_) {
    std::copy_n(src, len, &regrets_[n.data]);
    return;
  }
  float m = 0;
  for (size_t i = 0; i < len; ++i) m = std::max(m, std::fabs(src[i]));
  regret_scale_[n.slot] = m / 32767.f;
  const float inv = m > 0 ? 32767.f / m : 0.f;
  int16_t* q = &regrets16_[n.data];
  for (size_t i = 0; i < len; ++i) q[i] = static_cast<int16_t>(std::lrintf(src[i] * inv));
}

void Game::load_strategy_sum(const Node& n, float* dst) const {
  const size_t len = static_cast<size_t>(n.num_children) * hands_[n.player].size();
  if (!compressed_) {
    std::copy_n(&strategy_sum_[n.data], len, dst);
    return;
  }
  const float sc = strategy_scale_[n.slot];
  const uint16_t* q = &strategy_sum16_[n.data];
  for (size_t i = 0; i < len; ++i) dst[i] = static_cast<float>(q[i]) * sc;
}

void Game::store_strategy_sum(const Node& n, const float* src) {
  const size_t len = static_cast<size_t>(n.num_children) * hands_[n.player].size();
  if (!compressed_) {
    std::copy_n(src, len, &strategy_sum_[n.data]);
    return;
  }
  float m = 0;
  for (size_t i = 0; i < len; ++i) m = std::max(m, src[i]);
  strategy_scale_[n.slot] = m / 65535.f;
  const float inv = m > 0 ? 65535.f / m : 0.f;
  uint16_t* q = &strategy_sum16_[n.data];
  for (size_t i = 0; i < len; ++i) q[i] = static_cast<uint16_t>(std::lrintf(std::max(src[i], 0.f) * inv));
}

void Game::current_strategy(const Node& n, float* sigma) const {
  const size_t nh = hands_[n.player].size(), na = n.num_children;
  const float* r = sigma;
  if (compressed_) load_regrets(n, sigma);  // sigma holds the regrets first, then the strategy
  else r = &regrets_[n.data];               // 32-bit: read in place, no copy
  for (size_t h = 0; h < nh; ++h) {
    float sum = 0;
    for (size_t a = 0; a < na; ++a) sum += std::max(r[a * nh + h], 0.f);
    if (sum > 0) {
      for (size_t a = 0; a < na; ++a) sigma[a * nh + h] = std::max(r[a * nh + h], 0.f) / sum;
    } else {
      for (size_t a = 0; a < na; ++a) sigma[a * nh + h] = 1.f / static_cast<float>(na);
    }
  }
}

void Game::average_strategy(const Node& n, float* sigma) const {
  const size_t nh = hands_[n.player].size(), na = n.num_children;
  load_strategy_sum(n, sigma);
  for (size_t h = 0; h < nh; ++h) {
    float sum = 0;
    for (size_t a = 0; a < na; ++a) sum += sigma[a * nh + h];
    for (size_t a = 0; a < na; ++a) sigma[a * nh + h] = sum > 0 ? sigma[a * nh + h] / sum : 1.f / static_cast<float>(na);
  }
}

// ---------------------------------------------------------------- terminal values

// out[h] = total opponent reach of hands that share no card with hero hand h.
void Game::compat_sums(int p, const float* opp, float* out) const {
  const auto& oh = hands_[1 - p];
  double total = 0, card[kNumCards] = {0};
  for (size_t o = 0; o < oh.size(); ++o) {
    total += opp[o];
    card[oh[o].c1] += opp[o];
    card[oh[o].c2] += opp[o];
  }
  const auto& hh = hands_[p];
  for (size_t h = 0; h < hh.size(); ++h) {
    double v = total - card[hh[h].c1] - card[hh[h].c2];
    int32_t s = same_combo_[p][h];
    if (s >= 0) v += opp[s];  // removed twice above
    out[h] = static_cast<float>(v);
  }
}

void Game::fold_values(const Node& n, int p, const float* opp, float* out) const {
  const int f = n.player;
  const double payoff = (p == f) ? -n.contrib[f] : n.contrib[f];
  compat_sums(p, opp, out);
  for (size_t h = 0; h < hands_[p].size(); ++h) out[h] = static_cast<float>(out[h] * payoff);
}

// O(n) showdown: both lists are sorted by strength once per board, then one sweep
// up (hands we beat) and one sweep down (hands that beat us). Card removal uses
// per-card running sums.
void Game::showdown_values(const Node& n, int p, const float* opp, float* out) const {
  const Showdown& sd = showdowns_[showdown_index_[n.board_id]];
  const auto& hs = sd.strength[p];
  const auto& ho = sd.order[p];
  const auto& os = sd.strength[1 - p];
  const auto& oo = sd.order[1 - p];
  const auto& hh = hands_[p];
  const auto& oh = hands_[1 - p];
  const double c = n.contrib[0];
  std::fill(out, out + hh.size(), 0.f);
  thread_local std::vector<double> win;  // keep the first sweep in double: win - lose can cancel
  win.assign(ho.size(), 0.0);

  double sum = 0, card[kNumCards] = {0};
  size_t j = 0;
  for (size_t i = 0; i < ho.size(); ++i) {
    while (j < oo.size() && os[j] < hs[i]) {
      const float r = opp[oo[j]];
      sum += r;
      card[oh[oo[j]].c1] += r;
      card[oh[oo[j]].c2] += r;
      ++j;
    }
    const HandInfo& h = hh[ho[i]];
    win[i] = sum - card[h.c1] - card[h.c2];
  }
  sum = 0;
  std::fill(card, card + kNumCards, 0.0);
  j = oo.size();
  for (size_t i = ho.size(); i-- > 0;) {
    while (j > 0 && os[j - 1] > hs[i]) {
      --j;
      const float r = opp[oo[j]];
      sum += r;
      card[oh[oo[j]].c1] += r;
      card[oh[oo[j]].c2] += r;
    }
    const HandInfo& h = hh[ho[i]];
    out[ho[i]] = static_cast<float>(c * (win[i] - (sum - card[h.c1] - card[h.c2])));
  }
}

// ---------------------------------------------------------------- traversal

void Game::walk(uint32_t idx, int p, const float* opp, float* out, Mode mode) {
  const Node& n = nodes_[idx];
  const size_t nh = hands_[p].size();
  const size_t no = hands_[1 - p].size();

  switch (n.type) {
    case NodeType::Fold:
      fold_values(n, p, opp, out);
      return;
    case NodeType::Showdown:
      showdown_values(n, p, opp, out);
      return;

    case NodeType::Chance: {
      const size_t nc = n.num_children;
      std::vector<float> vals(nc * nh, 0.f);
      std::vector<float> reach(nc * no);
      std::vector<char> live(nc, 0);
      for (size_t k = 0; k < nc; ++k) {
        const Card d = nodes_[n.first_child + k].dealt;
        float* r = &reach[k * no];
        bool any = false;
        for (size_t o = 0; o < no; ++o) {
          r[o] = hand_has(hands_[1 - p][o], d) ? 0.f : opp[o];
          any |= r[o] > 0;
        }
        live[k] = any;
      }
      pool_->parallel_for(static_cast<int>(nc), [&](int k) {
        if (live[k]) walk(n.first_child + k, p, &reach[k * no], &vals[k * nh], mode);
      });
      const float scale = 1.f / static_cast<float>(48 - n.board_size);  // cards left after board + 4 hole cards
      std::fill(out, out + nh, 0.f);
      // Every dealable card adds its value. A card without its own subtree takes the values
      // of its representative r, read through the suit permutation sigma (sigma(r) = card):
      // value[h] = value_r[sigma^-1(h)].
      const ChanceEntry* ce = &chance_entries_[n.first_action];
      const size_t ne = static_cast<size_t>(kNumCards - n.board_size);
      for (size_t e = 0; e < ne; ++e) {
        if (!live[ce[e].child]) continue;
        const Card d = ce[e].card;
        const float* v = &vals[ce[e].child * nh];
        if (ce[e].perm == 0) {
          for (size_t h = 0; h < nh; ++h)
            if (!hand_has(hands_[p][h], d)) out[h] += v[h];
        } else {
          const uint16_t* hp = hand_perm_[suit_perms().inv[ce[e].perm]][p].data();
          for (size_t h = 0; h < nh; ++h)
            if (!hand_has(hands_[p][h], d)) out[h] += v[hp[h]];
        }
      }
      for (size_t h = 0; h < nh; ++h) out[h] *= scale;
      return;
    }

    case NodeType::Action:
      break;
  }

  const size_t na = n.num_children;
  if (n.player == p) {
    std::vector<float> vals(na * nh);
    for (size_t a = 0; a < na; ++a) walk(n.first_child + static_cast<uint32_t>(a), p, opp, &vals[a * nh], mode);
    if (mode == Mode::BestResponse) {
      for (size_t h = 0; h < nh; ++h) {
        float best = vals[h];
        for (size_t a = 1; a < na; ++a) best = std::max(best, vals[a * nh + h]);
        out[h] = best;
      }
      return;
    }
    std::vector<float> sigma(na * nh);
    if (mode == Mode::Train) current_strategy(n, sigma.data());
    else average_strategy(n, sigma.data());
    std::fill(out, out + nh, 0.f);
    for (size_t a = 0; a < na; ++a)
      for (size_t h = 0; h < nh; ++h) out[h] += sigma[a * nh + h] * vals[a * nh + h];
    if (mode == Mode::Train) {
      thread_local std::vector<float> rbuf;
      float* r;
      if (compressed_) {
        rbuf.resize(na * nh);
        r = rbuf.data();
        load_regrets(n, r);
      } else {
        r = &regrets_[n.data];  // 32-bit: update in place
      }
      for (size_t a = 0; a < na; ++a)
        for (size_t h = 0; h < nh; ++h) {
          float& x = r[a * nh + h];
          x *= x > 0 ? disc_pos_ : disc_neg_;
          x += vals[a * nh + h] - out[h];
        }
      if (compressed_) store_regrets(n, r);
    }
    return;
  }

  // Opponent node: split the opponent reach by the opponent strategy.
  std::vector<float> sigma(na * no);
  if (mode == Mode::Train) {
    current_strategy(n, sigma.data());
    thread_local std::vector<float> sbuf;
    float* s;
    if (compressed_) {
      sbuf.resize(na * no);
      s = sbuf.data();
      load_strategy_sum(n, s);
    } else {
      s = &strategy_sum_[n.data];  // 32-bit: update in place
    }
    for (size_t a = 0; a < na; ++a)
      for (size_t o = 0; o < no; ++o) s[a * no + o] = s[a * no + o] * disc_strat_ + opp[o] * sigma[a * no + o];
    if (compressed_) store_strategy_sum(n, s);
  } else {
    average_strategy(n, sigma.data());
  }
  std::fill(out, out + nh, 0.f);
  std::vector<float> reach(no), tmp(nh);
  for (size_t a = 0; a < na; ++a) {
    bool any = false;
    for (size_t o = 0; o < no; ++o) {
      reach[o] = opp[o] * sigma[a * no + o];
      any |= reach[o] > 0;
    }
    if (!any) continue;
    walk(n.first_child + static_cast<uint32_t>(a), p, reach.data(), tmp.data(), mode);
    for (size_t h = 0; h < nh; ++h) out[h] += tmp[h];
  }
}

void Game::run_iteration() {
  ++iteration_;
  const double t = iteration_;
  const double ta = std::pow(t, kAlpha), tb = std::pow(t, kBeta);
  disc_pos_ = static_cast<float>(ta / (ta + 1));
  disc_neg_ = static_cast<float>(tb / (tb + 1));
  disc_strat_ = static_cast<float>(std::pow(t / (t + 1), kGamma));
  for (int p = 0; p < 2; ++p) {
    std::vector<float> opp(hands_[1 - p].size()), out(hands_[p].size());
    for (size_t o = 0; o < opp.size(); ++o) opp[o] = hands_[1 - p][o].weight;
    walk(0, p, opp.data(), out.data(), Mode::Train);
  }
}

double Game::pair_weight_sum() const {
  std::vector<float> opp(hands_[1].size()), comp(hands_[0].size());
  for (size_t o = 0; o < opp.size(); ++o) opp[o] = hands_[1][o].weight;
  compat_sums(0, opp.data(), comp.data());
  double z = 0;
  for (size_t h = 0; h < comp.size(); ++h) z += static_cast<double>(hands_[0][h].weight) * comp[h];
  return z;
}

double Game::exploitability() {
  const double z = pair_weight_sum();
  double br = 0;
  for (int p = 0; p < 2; ++p) {
    std::vector<float> opp(hands_[1 - p].size()), out(hands_[p].size());
    for (size_t o = 0; o < opp.size(); ++o) opp[o] = hands_[1 - p][o].weight;
    walk(0, p, opp.data(), out.data(), Mode::BestResponse);
    for (size_t h = 0; h < out.size(); ++h) br += static_cast<double>(hands_[p][h].weight) * out[h];
  }
  return br / z / 2;
}

std::array<double, 2> Game::expected_values() {
  const double z = pair_weight_sum();
  std::array<double, 2> ev{};
  for (int p = 0; p < 2; ++p) {
    std::vector<float> opp(hands_[1 - p].size()), out(hands_[p].size());
    for (size_t o = 0; o < opp.size(); ++o) opp[o] = hands_[1 - p][o].weight;
    walk(0, p, opp.data(), out.data(), Mode::Average);
    double v = 0;
    for (size_t h = 0; h < out.size(); ++h) v += static_cast<double>(hands_[p][h].weight) * out[h];
    ev[p] = v / z + cfg_.pot / 2;  // zero-sum value + own half of the starting pot
  }
  return ev;
}

double Game::self_check_terminals(int max_nodes, unsigned seed) {
  uint32_t state = seed ? seed : 1;
  auto rnd = [&] {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
  };
  double worst = 0;
  int checked = 0;
  for (uint32_t i = 0; i < nodes_.size() && checked < max_nodes; ++i) {
    const Node& n = nodes_[i];
    if (n.type != NodeType::Showdown && n.type != NodeType::Fold) continue;
    ++checked;
    const auto& board = boards_[n.board_id];
    uint64_t used = 0;
    for (Card c : board) used |= 1ULL << c;
    for (int p = 0; p < 2; ++p) {
      const auto& hh = hands_[p];
      const auto& oh = hands_[1 - p];
      std::vector<float> opp(oh.size()), fast(hh.size());
      for (size_t o = 0; o < oh.size(); ++o)
        opp[o] = ((used >> oh[o].c1 & 1) || (used >> oh[o].c2 & 1)) ? 0.f : rnd();
      if (n.type == NodeType::Showdown) showdown_values(n, p, opp.data(), fast.data());
      else fold_values(n, p, opp.data(), fast.data());
      Card cards[7];
      for (int k = 0; k < 5 && k < static_cast<int>(board.size()); ++k) cards[k] = board[k];
      for (size_t h = 0; h < hh.size(); ++h) {
        if ((used >> hh[h].c1 & 1) || (used >> hh[h].c2 & 1)) continue;
        double slow = 0;
        for (size_t o = 0; o < oh.size(); ++o) {
          if (opp[o] == 0) continue;
          if (hand_has(oh[o], hh[h].c1) || hand_has(oh[o], hh[h].c2)) continue;
          double payoff;
          if (n.type == NodeType::Fold) {
            payoff = (p == n.player) ? -n.contrib[n.player] : n.contrib[n.player];
          } else {
            cards[5] = hh[h].c1, cards[6] = hh[h].c2;
            uint32_t a = eval_hand(cards, 7);
            cards[5] = oh[o].c1, cards[6] = oh[o].c2;
            uint32_t b = eval_hand(cards, 7);
            payoff = a > b ? n.contrib[0] : a < b ? -n.contrib[0] : 0;
          }
          slow += opp[o] * payoff;
        }
        worst = std::max(worst, std::fabs(slow - fast[h]) / std::max(1.0, std::fabs(slow)));
      }
    }
  }
  return worst;
}

// ---------------------------------------------------------------- queries

std::string Game::action_label(const ActionInfo& a) const {
  switch (a.kind) {
    case ActKind::Fold: return "Fold";
    case ActKind::Check: return "Check";
    case ActKind::Call: return "Call " + fmt(a.amount);
    case ActKind::Bet: return "Bet " + fmt(a.amount);
    case ActKind::Raise: return "Raise " + fmt(a.amount);
    case ActKind::AllIn: return "All-in " + fmt(a.amount);
  }
  return "?";
}

std::string Game::node_json(const std::vector<std::string>& path) { return to_json(node_view(path)); }

NodeView Game::node_view(const std::vector<std::string>& path) {
  std::vector<float> reach[2];
  for (int p = 0; p < 2; ++p) {
    reach[p].resize(hands_[p].size());
    for (size_t h = 0; h < hands_[p].size(); ++h) reach[p][h] = hands_[p][h].weight;
  }
  std::vector<std::string> history;
  uint32_t idx = 0;
  // The tree stores one subtree per group of suit-symmetric cards. tau maps the cards the
  // user sees to the cards inside the stored subtree (internal = perm_card(tau, shown)).
  int tau = 0;
  std::vector<Card> shown_board = cfg_.board;
  for (const std::string& tok : path) {
    const Node& n = nodes_[idx];
    if (n.type == NodeType::Action) {
      char* end = nullptr;
      long a = std::strtol(tok.c_str(), &end, 10);
      if (tok.empty() || *end != '\0' || a < 0 || a >= static_cast<long>(n.num_children))
        throw std::invalid_argument("bad action '" + tok + "' in path");
      std::vector<float> sigma(n.num_children * hands_[n.player].size());
      average_strategy(n, sigma.data());
      const size_t nh = hands_[n.player].size();
      for (size_t h = 0; h < nh; ++h) reach[n.player][h] *= sigma[a * nh + h];
      history.push_back(std::string(n.player == 0 ? "OOP " : "IP ") + action_label(actions_[n.first_action + a]));
      idx = n.first_child + static_cast<uint32_t>(a);
    } else if (n.type == NodeType::Chance) {
      int c = parse_card(tok);
      if (c < 0) throw std::invalid_argument("bad card '" + tok + "' in path");
      const Card ci = perm_card(tau, static_cast<Card>(c));
      const ChanceEntry* ce = &chance_entries_[n.first_action];
      const ChanceEntry* hit = nullptr;
      for (int e = 0; e < kNumCards - n.board_size; ++e)
        if (ce[e].card == ci) hit = &ce[e];
      if (!hit) throw std::invalid_argument("card " + tok + " is already on the board");
      // Move into the representative's subtree: hand h there = hand sigma(h) here.
      const int sigma = hit->perm;
      if (sigma != 0)
        for (int p = 0; p < 2; ++p) {
          std::vector<float> moved(reach[p].size());
          for (size_t h = 0; h < moved.size(); ++h) moved[h] = reach[p][hand_perm_[sigma][p][h]];
          reach[p] = std::move(moved);
        }
      const uint32_t next = n.first_child + hit->child;
      for (int p = 0; p < 2; ++p)
        for (size_t h = 0; h < hands_[p].size(); ++h)
          if (hand_has(hands_[p][h], nodes_[next].dealt)) reach[p][h] = 0;
      tau = compose(suit_perms().inv[sigma], tau);
      shown_board.push_back(static_cast<Card>(c));
      history.push_back("deal " + tok);
      idx = next;
    } else {
      throw std::invalid_argument("path goes past the end of the hand");
    }
  }

  const Node& n = nodes_[idx];
  NodeView v;
  const int shown = suit_perms().inv[tau];  // internal card -> card the user sees
  v.board = shown_board;
  v.pot = n.contrib[0] + n.contrib[1];
  v.contrib[0] = n.contrib[0];
  v.contrib[1] = n.contrib[1];
  v.history = history;
  if (n.type == NodeType::Fold) {
    v.type = "fold";
    v.winner = 1 - n.player;
    return v;
  }
  if (n.type == NodeType::Showdown) {
    v.type = "showdown";
    return v;
  }
  if (n.type == NodeType::Chance) {
    v.type = "chance";
    uint64_t on_board = 0;
    for (Card c : shown_board) on_board |= 1ULL << c;
    for (int c = 0; c < kNumCards; ++c)
      if (!(on_board >> c & 1)) v.cards.push_back(static_cast<Card>(c));
    return v;
  }

  // Action node: strategy and EV of every hand of the actor that can reach this node.
  v.type = "action";
  const int a = n.player, o = 1 - a;
  v.player = a;
  const size_t nh = hands_[a].size(), na = n.num_children;
  std::vector<float> sigma(na * nh), vals(na * nh), comp(nh);
  average_strategy(n, sigma.data());
  for (size_t k = 0; k < na; ++k) walk(n.first_child + static_cast<uint32_t>(k), a, reach[o].data(), &vals[k * nh], Mode::Average);
  compat_sums(a, reach[o].data(), comp.data());
  const double own = n.contrib[a];  // chips already in: EV shown = chips won from the pot from here
  for (size_t k = 0; k < na; ++k) {
    v.action_info.push_back(actions_[n.first_action + k]);
    v.actions.push_back(action_label(actions_[n.first_action + k]));
  }
  uint64_t used = 0;
  for (Card c : boards_[n.board_id]) used |= 1ULL << c;
  v.freq.assign(na, 0);
  for (size_t h = 0; h < nh; ++h) {
    const HandInfo& hi = hands_[a][h];
    const double w = reach[a][h];
    if (w <= 1e-7 || (used >> hi.c1 & 1) || (used >> hi.c2 & 1) || comp[h] <= 0) continue;
    HandView hv;
    hv.combo = combo_index(perm_card(shown, hi.c1), perm_card(shown, hi.c2));
    hv.w = w;
    hv.e = 0;
    for (size_t k = 0; k < na; ++k) {
      hv.s.push_back(sigma[k * nh + h]);
      hv.ev.push_back(vals[k * nh + h] / comp[h] + own);
      hv.e += hv.s.back() * hv.ev.back();
      v.freq[k] += w * sigma[k * nh + h];
    }
    v.combos += w;
    v.hands.push_back(std::move(hv));
  }
  for (double& f : v.freq) f = v.combos > 0 ? f / v.combos : 0;
  return v;
}

std::string Game::to_json(const NodeView& v) {
  std::ostringstream js;
  js << "{\"board\":\"" << cards_str(v.board) << "\",\"pot\":" << json_num(v.pot) << ",\"contrib\":["
     << json_num(v.contrib[0]) << "," << json_num(v.contrib[1]) << "],\"history\":[";
  for (size_t i = 0; i < v.history.size(); ++i) js << (i ? "," : "") << "\"" << v.history[i] << "\"";
  js << "]";
  if (v.type == "fold") {
    js << ",\"type\":\"fold\",\"winner\":" << v.winner << "}";
    return js.str();
  }
  if (v.type == "showdown") {
    js << ",\"type\":\"showdown\"}";
    return js.str();
  }
  if (v.type == "chance") {
    js << ",\"type\":\"chance\",\"cards\":[";
    for (size_t i = 0; i < v.cards.size(); ++i) js << (i ? "," : "") << "\"" << card_str(v.cards[i]) << "\"";
    js << "]}";
    return js.str();
  }
  const size_t na = v.actions.size();
  std::vector<double> cell_w(169, 0), cell_ev(169, 0), cell_s(169 * na, 0);
  std::ostringstream hands_js;
  for (size_t i = 0; i < v.hands.size(); ++i) {
    const HandView& h = v.hands[i];
    hands_js << (i ? "," : "") << "{\"c\":\"" << combo_str(h.combo) << "\",\"w\":" << json_num(h.w) << ",\"s\":[";
    for (size_t k = 0; k < na; ++k) hands_js << (k ? "," : "") << json_num(h.s[k]);
    hands_js << "],\"ev\":[";
    for (size_t k = 0; k < na; ++k) hands_js << (k ? "," : "") << json_num(h.ev[k], 3);
    hands_js << "],\"e\":" << json_num(h.e, 3) << "}";
    const int cell = grid_cell(h.combo);
    cell_w[cell] += h.w;
    cell_ev[cell] += h.w * h.e;
    for (size_t k = 0; k < na; ++k) cell_s[cell * na + k] += h.w * h.s[k];
  }
  js << ",\"type\":\"action\",\"player\":" << v.player << ",\"actions\":[";
  for (size_t k = 0; k < na; ++k) js << (k ? "," : "") << "\"" << v.actions[k] << "\"";
  js << "],\"freq\":[";
  for (size_t k = 0; k < na; ++k) js << (k ? "," : "") << json_num(v.freq[k]);
  js << "],\"combos\":" << json_num(v.combos, 2) << ",\"grid\":[";
  for (int c = 0; c < 169; ++c) {
    js << (c ? "," : "") << "{\"l\":\"" << grid_label(c) << "\",\"w\":" << json_num(cell_w[c]);
    if (cell_w[c] > 0) {
      js << ",\"ev\":" << json_num(cell_ev[c] / cell_w[c], 3) << ",\"s\":[";
      for (size_t k = 0; k < na; ++k) js << (k ? "," : "") << json_num(cell_s[c * na + k] / cell_w[c]);
      js << "]";
    }
    js << "}";
  }
  js << "],\"hands\":[" << hands_js.str() << "]}";
  return js.str();
}

}  // namespace poker
