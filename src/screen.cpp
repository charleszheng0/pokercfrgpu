// pokerscreen: a free screen of every flop (no solving). For each of the 1,755 distinct
// flops (suit renaming counts as the same flop) and each preset, it writes one CSV row with:
//   weight      how many of the 22,100 dealt flops map to this flop
//   texture     monotone > paired > wet > low > dry_high (first class that matches)
//   ip_equity   IP range equity against the OOP range, all-in to the river
//   oop_nuts    share of the OOP range with at least 80% equity against the IP range
//   ip_nuts     the same for IP;  nut_adv = ip_nuts - oop_nuts
// Equity uses --runouts random turn and river pairs per flop (fixed seed, same for all presets).
//
//   pokerscreen --out data/flop_screen.csv --runouts 60 --seed 7
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "presets.hpp"
#include "range.hpp"
#include "spot.hpp"

using namespace poker;

namespace {

// wet = the three ranks fit in a window of 5 (a straight is possible, ace counts low too),
// or two cards share a suit and are at most 2 ranks apart.
std::string texture(const std::vector<Card>& f) {
  int r[3] = {rank_of(f[0]), rank_of(f[1]), rank_of(f[2])};
  int s[3] = {suit_of(f[0]), suit_of(f[1]), suit_of(f[2])};
  std::sort(r, r + 3);
  if (s[0] == s[1] && s[1] == s[2]) return "monotone";
  if (r[0] == r[1] || r[1] == r[2]) return "paired";
  bool straight = r[2] - r[0] <= 4;
  if (r[2] == 12) {  // ace low
    int lo[3] = {-1, r[0], r[1]};
    straight = straight || lo[2] - lo[0] <= 4;
  }
  bool suited_close = false;
  for (int a = 0; a < 3; ++a)
    for (int b = a + 1; b < 3; ++b)
      if (suit_of(f[a]) == suit_of(f[b]) && std::abs(rank_of(f[a]) - rank_of(f[b])) <= 2) suited_close = true;
  if (straight || suited_close) return "wet";
  if (r[2] <= 7) return "low";  // top card 9 or lower
  return "dry_high";
}

struct Side {
  std::vector<int> combo;  // combos with weight > 0 that do not touch the flop
  std::vector<float> w;
};

// Equity of each hand of `me` against `opp` on one 5-card board, added into eq[] (wins + ties/2,
// divided by the compatible opponent weight). str[c] = strength of combo c on this board.
void add_equity(const Side& me, const Side& opp, const std::vector<uint32_t>& str, const std::vector<char>& dead,
                std::vector<double>& eq) {
  std::vector<int> oi, mi;
  for (size_t j = 0; j < opp.combo.size(); ++j)
    if (!dead[opp.combo[j]]) oi.push_back(static_cast<int>(j));
  for (size_t i = 0; i < me.combo.size(); ++i)
    if (!dead[me.combo[i]]) mi.push_back(static_cast<int>(i));
  auto by_str_o = [&](int a, int b) { return str[opp.combo[a]] < str[opp.combo[b]]; };
  auto by_str_m = [&](int a, int b) { return str[me.combo[a]] < str[me.combo[b]]; };
  std::sort(oi.begin(), oi.end(), by_str_o);
  std::sort(mi.begin(), mi.end(), by_str_m);
  // Total and per-card opponent weight, then two sweeps: strictly lower, and lower or equal.
  double tot = 0, card_tot[52] = {};
  std::vector<double> opp_by_combo(kNumCombos, 0.0);
  for (int j : oi) {
    auto c = combo_cards(opp.combo[j]);
    tot += opp.w[j], card_tot[c.first] += opp.w[j], card_tot[c.second] += opp.w[j];
    opp_by_combo[opp.combo[j]] = opp.w[j];
  }
  auto sweep = [&](bool or_equal, std::vector<double>& out) {
    double below = 0, card_below[52] = {};
    std::vector<char> in(kNumCombos, 0);
    size_t p = 0;
    for (int i : mi) {
      const uint32_t s = str[me.combo[i]];
      while (p < oi.size() && (str[opp.combo[oi[p]]] < s || (or_equal && str[opp.combo[oi[p]]] == s))) {
        const int j = oi[p++];
        auto c = combo_cards(opp.combo[j]);
        below += opp.w[j], card_below[c.first] += opp.w[j], card_below[c.second] += opp.w[j];
        in[opp.combo[j]] = 1;
      }
      auto c = combo_cards(me.combo[i]);
      const int same = me.combo[i];
      out[i] = below - card_below[c.first] - card_below[c.second] + (in[same] ? opp_by_combo[same] : 0.0);
    }
  };
  std::vector<double> lt(me.combo.size()), le(me.combo.size());
  sweep(false, lt);
  sweep(true, le);
  for (int i : mi) {
    auto c = combo_cards(me.combo[i]);
    const double compat = tot - card_tot[c.first] - card_tot[c.second] + opp_by_combo[me.combo[i]];
    if (compat > 0) eq[i] += (lt[i] + (le[i] - lt[i]) * 0.5) / compat;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  const std::string out_path = field(f, "out", "data/flop_screen.csv");
  const int runouts = static_cast<int>(parse_number(f, "runouts", 60));
  const unsigned seed = static_cast<unsigned>(parse_number(f, "seed", 7));
  const double nut_eq = parse_number(f, "nut_equity", 0.80);

  // 1. Distinct flops and their weights.
  std::map<std::string, int> weight;
  for (int a = 0; a < 52; ++a)
    for (int b = a + 1; b < 52; ++b)
      for (int c = b + 1; c < 52; ++c)
        ++weight[canonical_board_key({static_cast<Card>(a), static_cast<Card>(b), static_cast<Card>(c)})];
  std::vector<std::string> keys;
  for (auto& kv : weight) keys.push_back(kv.first);
  std::printf("distinct flops: %zu\n", keys.size());

  const auto& ps = presets();
  std::vector<Range> ranges[2];
  for (const auto& p : ps) ranges[0].push_back(parse_range(p.oop)), ranges[1].push_back(parse_range(p.ip));

  struct Row {
    double ip_eq = 0, oop_nuts = 0, ip_nuts = 0;
  };
  std::vector<std::vector<Row>> rows(keys.size(), std::vector<Row>(ps.size()));
  std::atomic<size_t> next{0};
  auto t0 = std::chrono::steady_clock::now();
  auto work = [&]() {
    std::vector<uint32_t> str(kNumCombos);
    std::vector<char> dead(kNumCombos);
    for (size_t k; (k = next++) < keys.size();) {
      const std::vector<Card> flop = parse_cards(keys[k]);
      bool on_flop[52] = {};
      for (Card c : flop) on_flop[c] = true;
      // Random turn and river pairs, the same list for every preset.
      std::mt19937 rng(seed + static_cast<unsigned>(k));
      std::vector<int> deck;
      for (int c = 0; c < 52; ++c)
        if (!on_flop[c]) deck.push_back(c);
      std::vector<std::pair<Card, Card>> runs;
      for (int r = 0; r < runouts; ++r) {
        std::shuffle(deck.begin(), deck.end(), rng);
        runs.push_back({static_cast<Card>(deck[0]), static_cast<Card>(deck[1])});
      }
      std::vector<Side> side[2];
      for (size_t pi = 0; pi < ps.size(); ++pi)
        for (int p = 0; p < 2; ++p) {
          Side s;
          for (int c = 0; c < kNumCombos; ++c) {
            auto cc = combo_cards(c);
            if (ranges[p][pi].weight[c] > 0 && !on_flop[cc.first] && !on_flop[cc.second])
              s.combo.push_back(c), s.w.push_back(ranges[p][pi].weight[c]);
          }
          side[p].push_back(s);
        }
      std::vector<std::vector<double>> eq[2];
      for (int p = 0; p < 2; ++p)
        for (size_t pi = 0; pi < ps.size(); ++pi) eq[p].push_back(std::vector<double>(side[p][pi].combo.size(), 0.0));
      std::vector<int> seen[2];  // runouts where each hand was live
      for (const auto& run : runs) {
        Card board[7] = {flop[0], flop[1], flop[2], run.first, run.second, 0, 0};
        for (int c = 0; c < kNumCombos; ++c) {
          auto cc = combo_cards(c);
          dead[c] = cc.first == run.first || cc.first == run.second || cc.second == run.first || cc.second == run.second;
          if (on_flop[cc.first] || on_flop[cc.second] || dead[c]) continue;
          board[5] = cc.first, board[6] = cc.second;
          str[c] = eval_hand(board, 7);
        }
        for (size_t pi = 0; pi < ps.size(); ++pi) {
          add_equity(side[0][pi], side[1][pi], str, dead, eq[0][pi]);
          add_equity(side[1][pi], side[0][pi], str, dead, eq[1][pi]);
        }
      }
      // Live count per combo: runouts that do not use its cards.
      std::vector<int> live(kNumCombos, 0);
      for (int c = 0; c < kNumCombos; ++c) {
        auto cc = combo_cards(c);
        for (const auto& run : runs)
          if (cc.first != run.first && cc.first != run.second && cc.second != run.first && cc.second != run.second) ++live[c];
      }
      for (size_t pi = 0; pi < ps.size(); ++pi) {
        Row& row = rows[k][pi];
        double wsum[2] = {0, 0}, eqsum[2] = {0, 0}, nuts[2] = {0, 0};
        for (int p = 0; p < 2; ++p) {
          const Side& s = side[p][pi];
          for (size_t i = 0; i < s.combo.size(); ++i) {
            if (live[s.combo[i]] == 0) continue;
            const double e = eq[p][pi][i] / live[s.combo[i]];
            wsum[p] += s.w[i], eqsum[p] += s.w[i] * e;
            if (e >= nut_eq) nuts[p] += s.w[i];
          }
        }
        row.ip_eq = eqsum[1] / wsum[1];
        row.oop_nuts = nuts[0] / wsum[0], row.ip_nuts = nuts[1] / wsum[1];
      }
    }
  };
  std::vector<std::thread> th;
  const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned t = 0; t < nt; ++t) th.emplace_back(work);
  for (auto& t : th) t.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  FILE* out = std::fopen(out_path.c_str(), "w");
  if (!out) {
    std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
    return 2;
  }
  std::fprintf(out, "preset,flop,weight,texture,top_rank,ip_equity,oop_nuts,ip_nuts,nut_adv\n");
  for (size_t k = 0; k < keys.size(); ++k) {
    const auto flop = parse_cards(keys[k]);
    int top = 0;
    for (Card c : flop) top = std::max(top, rank_of(c));
    for (size_t pi = 0; pi < ps.size(); ++pi) {
      const Row& r = rows[k][pi];
      std::fprintf(out, "%s,%s,%d,%s,%c,%.4f,%.4f,%.4f,%.4f\n", ps[pi].name.c_str(), keys[k].c_str(), weight[keys[k]],
                   texture(flop).c_str(), "23456789TJQKA"[top], r.ip_eq, r.oop_nuts, r.ip_nuts, r.ip_nuts - r.oop_nuts);
    }
  }
  std::fclose(out);
  std::printf("wrote %zu rows to %s in %.1f s (%u threads, %d runouts per flop)\n", keys.size() * ps.size(),
              out_path.c_str(), secs, nt, runouts);

  // 2. Pick a small typical set per preset (--pick_srp per single-raised preset, --pick_3bet per 3-bet preset).
  // Groups = texture class. Slots per group follow the deal weight (largest
  // remainder, at least 1 for a group with 1% or more of the deals). Inside a group the flops are sorted by
  // nut advantage and picked at evenly spaced weighted quantiles, so one group gives different flops.
  const int pick_srp = static_cast<int>(parse_number(f, "pick_srp", 12));
  const int pick_3bet = static_cast<int>(parse_number(f, "pick_3bet", 12));
  const std::string list_path = field(f, "list", "data/flop_list.txt");
  struct Pick { size_t preset; std::string flop; std::string group; double share; };
  std::vector<std::vector<Pick>> per(ps.size());
  for (size_t pi = 0; pi < ps.size(); ++pi) {
    const int want = ps[pi].name.find("3-bet") != std::string::npos ? pick_3bet : pick_srp;
    std::map<std::string, std::vector<size_t>> groups;
    for (size_t k = 0; k < keys.size(); ++k) {
      const auto flop = parse_cards(keys[k]);
      int top = 0;
      for (Card c : flop) top = std::max(top, rank_of(c));
      (void)top;
      groups[texture(flop)].push_back(k);
    }
    struct G { std::string name; double share; int slots; double rem; };
    std::vector<G> gs;
    for (auto& [name, ks] : groups) {
      double w = 0;
      for (size_t k : ks) w += weight[keys[k]];
      const double share = w / 22100.0, exact = share * want;
      gs.push_back({name, share, static_cast<int>(exact), exact - static_cast<int>(exact)});
    }
    for (auto& g : gs)
      if (g.slots == 0 && g.share >= 0.01) g.slots = 1, g.rem = -1;
    int used = 0;
    for (auto& g : gs) used += g.slots;
    std::sort(gs.begin(), gs.end(), [](const G& a, const G& b) { return a.rem > b.rem; });
    for (size_t i = 0; used < want && i < gs.size(); ++i)
      if (gs[i].rem >= 0) ++gs[i].slots, ++used;
    while (used > want) {  // too many minimum slots: drop from the smallest groups with one slot
      auto it = std::min_element(gs.begin(), gs.end(), [](const G& a, const G& b) {
        return (a.slots ? a.share : 9) < (b.slots ? b.share : 9);
      });
      --it->slots, --used;
    }
    std::sort(gs.begin(), gs.end(), [](const G& a, const G& b) { return a.share > b.share; });
    for (const auto& g : gs) {
      auto ks = groups[g.name];
      std::sort(ks.begin(), ks.end(), [&](size_t a, size_t b) {
        const double na = rows[a][pi].ip_nuts - rows[a][pi].oop_nuts, nb = rows[b][pi].ip_nuts - rows[b][pi].oop_nuts;
        return na < nb;
      });
      double tw = 0;
      for (size_t k : ks) tw += weight[keys[k]];
      for (int j = 0; j < g.slots; ++j) {
        const double q = (j + 0.5) / g.slots * tw;
        double cum = 0;
        for (size_t k : ks) {
          cum += weight[keys[k]];
          if (cum >= q) { per[pi].push_back({pi, keys[k], g.name, g.share}); break; }
        }
      }
    }
  }
  // Within a preset: the first flop of every group (common groups first), then every second flop, and so on.
  for (auto& v : per) {
    std::map<std::string, int> seen;
    std::vector<std::pair<int, size_t>> order;  // (pass, position)
    for (size_t i = 0; i < v.size(); ++i) order.push_back({seen[v[i].group]++, i});
    std::stable_sort(order.begin(), order.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::vector<Pick> nv;
    for (auto& o : order) nv.push_back(v[o.second]);
    v = nv;
  }
  // Interleave presets so a partial run still covers every preset.
  FILE* lf = std::fopen(list_path.c_str(), "w");
  if (!lf) {
    std::fprintf(stderr, "error: cannot write %s\n", list_path.c_str());
    return 2;
  }
  std::fprintf(lf, "# preset|flop|group|group share of deals. Made by pokerscreen.\n");
  size_t total = 0;
  for (size_t round = 0;; ++round) {
    bool any = false;
    for (size_t pi = 0; pi < ps.size(); ++pi)
      if (round < per[pi].size()) {
        const Pick& p = per[pi][round];
        std::fprintf(lf, "%s|%s|%s|%.3f\n", ps[pi].name.c_str(), p.flop.c_str(), p.group.c_str(), p.share);
        any = true, ++total;
      }
    if (!any) break;
  }
  std::fclose(lf);
  std::printf("picked %zu flops into %s\n", total, list_path.c_str());
  return 0;
}
