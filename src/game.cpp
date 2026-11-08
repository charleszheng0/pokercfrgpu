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

  boards_.push_back(cfg_.board);
  BuildState st{0, bs, {cfg_.pot / 2, cfg_.pot / 2}, 0, 0};
  nodes_.emplace_back();
  build_decision(0, st);

  for (auto& n : nodes_) {
    if (n.type != NodeType::Action) continue;
    n.data = arena_size_;
    arena_size_ += static_cast<uint64_t>(n.num_children) * hands_[n.player].size();
    ++num_action_nodes_;
  }
  const double gb = memory_bytes() / 1e9;
  if (gb > cfg_.mem_limit_gb) {
    std::ostringstream os;
    os << "tree needs " << fmt(gb, 2) << " GB (limit " << fmt(cfg_.mem_limit_gb, 2)
       << " GB). Use fewer bet sizes, smaller ranges, or a later street.";
    throw std::runtime_error(os.str());
  }
  regrets_.assign(arena_size_, 0.f);
  strategy_sum_.assign(arena_size_, 0.f);
  prepare_showdowns();

  int threads = cfg_.threads > 0 ? cfg_.threads : static_cast<int>(std::thread::hardware_concurrency());
  pool_ = std::make_unique<ThreadPool>(std::max(1, threads));
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
  std::vector<Card> deal;
  for (int c = 0; c < kNumCards; ++c)
    if (!(used >> c & 1)) deal.push_back(static_cast<Card>(c));

  const uint32_t first = static_cast<uint32_t>(nodes_.size());
  nodes_[idx].num_children = static_cast<uint32_t>(deal.size());
  nodes_[idx].first_child = first;
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

void Game::current_strategy(const Node& n, float* sigma) const {
  const size_t nh = hands_[n.player].size(), na = n.num_children;
  const float* r = &regrets_[n.data];
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
  const float* s = &strategy_sum_[n.data];
  for (size_t h = 0; h < nh; ++h) {
    float sum = 0;
    for (size_t a = 0; a < na; ++a) sum += s[a * nh + h];
    for (size_t a = 0; a < na; ++a)
      sigma[a * nh + h] = sum > 0 ? s[a * nh + h] / sum : 1.f / static_cast<float>(na);
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
      for (size_t k = 0; k < nc; ++k) {
        if (!live[k]) continue;
        const Card d = nodes_[n.first_child + k].dealt;
        const float* v = &vals[k * nh];
        for (size_t h = 0; h < nh; ++h)
          if (!hand_has(hands_[p][h], d)) out[h] += v[h];
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
      float* r = &regrets_[n.data];
      for (size_t a = 0; a < na; ++a)
        for (size_t h = 0; h < nh; ++h) {
          float& x = r[a * nh + h];
          x *= x > 0 ? disc_pos_ : disc_neg_;
          x += vals[a * nh + h] - out[h];
        }
    }
    return;
  }

  // Opponent node: split the opponent reach by the opponent strategy.
  std::vector<float> sigma(na * no);
  if (mode == Mode::Train) {
    current_strategy(n, sigma.data());
    float* s = &strategy_sum_[n.data];
    for (size_t a = 0; a < na; ++a)
      for (size_t o = 0; o < no; ++o) s[a * no + o] = s[a * no + o] * disc_strat_ + opp[o] * sigma[a * no + o];
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

std::string Game::node_json(const std::vector<std::string>& path) {
  std::vector<float> reach[2];
  for (int p = 0; p < 2; ++p) {
    reach[p].resize(hands_[p].size());
    for (size_t h = 0; h < hands_[p].size(); ++h) reach[p][h] = hands_[p][h].weight;
  }
  std::vector<std::string> history;
  uint32_t idx = 0;
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
      uint32_t next = 0;
      for (uint32_t k = 0; k < n.num_children; ++k)
        if (nodes_[n.first_child + k].dealt == c) next = n.first_child + k;
      if (next == 0) throw std::invalid_argument("card " + tok + " is already on the board");
      for (int p = 0; p < 2; ++p)
        for (size_t h = 0; h < hands_[p].size(); ++h)
          if (hand_has(hands_[p][h], static_cast<Card>(c))) reach[p][h] = 0;
      history.push_back("deal " + tok);
      idx = next;
    } else {
      throw std::invalid_argument("path goes past the end of the hand");
    }
  }

  const Node& n = nodes_[idx];
  std::ostringstream js;
  js << "{\"board\":\"" << cards_str(boards_[n.board_id]) << "\",\"pot\":" << json_num(n.contrib[0] + n.contrib[1])
     << ",\"contrib\":[" << json_num(n.contrib[0]) << "," << json_num(n.contrib[1]) << "],\"history\":[";
  for (size_t i = 0; i < history.size(); ++i) js << (i ? "," : "") << "\"" << history[i] << "\"";
  js << "]";

  if (n.type == NodeType::Fold) {
    js << ",\"type\":\"fold\",\"winner\":" << (1 - n.player) << "}";
    return js.str();
  }
  if (n.type == NodeType::Showdown) {
    js << ",\"type\":\"showdown\"}";
    return js.str();
  }
  if (n.type == NodeType::Chance) {
    js << ",\"type\":\"chance\",\"cards\":[";
    for (uint32_t k = 0; k < n.num_children; ++k)
      js << (k ? "," : "") << "\"" << card_str(nodes_[n.first_child + k].dealt) << "\"";
    js << "]}";
    return js.str();
  }

  // Action node: strategy and EV of every hand of the actor that can reach this node.
  const int a = n.player, o = 1 - a;
  const size_t nh = hands_[a].size(), na = n.num_children;
  std::vector<float> sigma(na * nh), vals(na * nh), comp(nh);
  average_strategy(n, sigma.data());
  for (size_t k = 0; k < na; ++k) walk(n.first_child + static_cast<uint32_t>(k), a, reach[o].data(), &vals[k * nh], Mode::Average);
  compat_sums(a, reach[o].data(), comp.data());
  const double own = n.contrib[a];  // chips already in: EV shown = chips won from the pot from here

  uint64_t used = 0;
  for (Card c : boards_[n.board_id]) used |= 1ULL << c;

  std::vector<double> freq(na, 0), cell_w(169, 0), cell_ev(169, 0), cell_s(169 * na, 0);
  double total = 0;
  std::ostringstream hands_js;
  bool first = true;
  for (size_t h = 0; h < nh; ++h) {
    const HandInfo& hi = hands_[a][h];
    const double w = reach[a][h];
    if (w <= 1e-7 || (used >> hi.c1 & 1) || (used >> hi.c2 & 1) || comp[h] <= 0) continue;
    double ev_h = 0;
    hands_js << (first ? "" : ",") << "{\"c\":\"" << combo_str(hi.combo) << "\",\"w\":" << json_num(w) << ",\"s\":[";
    first = false;
    for (size_t k = 0; k < na; ++k) hands_js << (k ? "," : "") << json_num(sigma[k * nh + h]);
    hands_js << "],\"ev\":[";
    for (size_t k = 0; k < na; ++k) {
      const double ev = vals[k * nh + h] / comp[h] + own;
      ev_h += sigma[k * nh + h] * ev;
      hands_js << (k ? "," : "") << json_num(ev, 3);
    }
    hands_js << "],\"e\":" << json_num(ev_h, 3) << "}";
    const int cell = grid_cell(hi.combo);
    total += w;
    cell_w[cell] += w;
    cell_ev[cell] += w * ev_h;
    for (size_t k = 0; k < na; ++k) {
      freq[k] += w * sigma[k * nh + h];
      cell_s[cell * na + k] += w * sigma[k * nh + h];
    }
  }
  js << ",\"type\":\"action\",\"player\":" << a << ",\"actions\":[";
  for (size_t k = 0; k < na; ++k) js << (k ? "," : "") << "\"" << action_label(actions_[n.first_action + k]) << "\"";
  js << "],\"freq\":[";
  for (size_t k = 0; k < na; ++k) js << (k ? "," : "") << json_num(total > 0 ? freq[k] / total : 0);
  js << "],\"combos\":" << json_num(total, 2) << ",\"grid\":[";
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
