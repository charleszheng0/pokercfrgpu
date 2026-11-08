// pokerbatch: solve many spots and store the decision points for the trainer.
//
//   pokerbatch --db data/spots.db --count 50 --street river --seed 1
//
// For each spot: pick a preset (house ranges) and a random board that the database does not
// have yet (suit renaming counts as the same board), solve until the exploitability gate is
// met, reject the solve if it is not, then score every decision node of the first street
// (up to --depth actions deep) and store it. The flop (and the turn, for river spots) are taken as
// checked through, so the starting ranges stay the preflop ranges. The story says so.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "game.hpp"
#include "presets.hpp"
#include "spot.hpp"
#include "store.hpp"

#ifndef POKER_VERSION
#define POKER_VERSION "dev"
#endif

using namespace poker;

namespace {

struct Scores {
  double mixed = 0, ev_loss = 0, bluff = 0, fundamental = 0, aggression = 0;
  std::string category;
  double score = 0;
};

bool aggressive(ActKind k) { return k == ActKind::Bet || k == ActKind::Raise || k == ActKind::AllIn; }

// Spot scores (docs/STORE.md has the definitions). All are weighted by the reach of each hand.
//   mixed       share of hands whose most frequent action is below 70%
//   ev_loss     mean EV gap between the best and the second-best action, % of the pot
//   bluff       share of the bet/raise frequency that comes from weak hands
//               (weak = the passive action, check or call, wins less than 15% of the pot)
//   fundamental share of hands with a near-pure strategy (top action at least 90%)
Scores score_node(const NodeView& v) {
  Scores sc;
  const size_t na = v.actions.size();
  int passive = -1;
  for (size_t k = 0; k < na; ++k)
    if (v.action_info[k].kind == ActKind::Check || v.action_info[k].kind == ActKind::Call) passive = static_cast<int>(k);
  double wsum = 0, aggr_all = 0, aggr_weak = 0;
  for (const HandView& h : v.hands) {
    double top = 0, best = -1e18, second = -1e18, aggr = 0;
    for (size_t k = 0; k < na; ++k) {
      top = std::max(top, h.s[k]);
      if (h.ev[k] > best) second = best, best = h.ev[k];
      else if (h.ev[k] > second) second = h.ev[k];
      if (aggressive(v.action_info[k].kind)) aggr += h.s[k];
    }
    wsum += h.w;
    if (top < 0.7) sc.mixed += h.w;
    if (top >= 0.9) sc.fundamental += h.w;
    if (na > 1) sc.ev_loss += h.w * (best - second) / v.pot * 100;
    aggr_all += h.w * aggr;
    if (passive >= 0 && h.ev[passive] < 0.15 * v.pot) aggr_weak += h.w * aggr;
  }
  if (wsum > 0) {
    sc.mixed /= wsum;
    sc.fundamental /= wsum;
    sc.ev_loss /= wsum;
    sc.aggression = aggr_all / wsum;
  }
  sc.bluff = aggr_all > 0 ? aggr_weak / aggr_all : 0;
  // One category per spot, checked in this order.
  if (sc.aggression >= 0.15 && sc.bluff >= 0.3) sc.category = "bluff", sc.score = sc.bluff;
  else if (sc.mixed >= 0.35) sc.category = "mixed", sc.score = sc.mixed;
  else if (sc.ev_loss >= 5) sc.category = "ev_loss", sc.score = sc.ev_loss;
  else sc.category = "fundamental", sc.score = sc.fundamental;
  return sc;
}

std::string json_str_array(const std::vector<std::string>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) o += (i ? ",\"" : "\"") + v[i] + "\"";
  return o + "]";
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
  std::string o;
  for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  const std::string db_path = field(f, "db", "data/spots.db");
  const int count = static_cast<int>(parse_number(f, "count", 10));
  const unsigned seed = static_cast<unsigned>(parse_number(f, "seed", 1));
  const std::string street_opt = field(f, "street", "river");  // river, turn, mix
  const std::string preset_opt = field(f, "preset", "all");
  const int max_iters = static_cast<int>(parse_number(f, "iters", 1500));
  const double max_db_gb = parse_number(f, "max_db_gb", 4);  // stop before the disk fills
  const size_t max_depth = static_cast<size_t>(parse_number(f, "depth", 4));  // actions on this street

  try {
    Store store(db_path);
    // Older rows: fill in the board key, then remove later copies of the same spot.
    auto missing = store.solves_without_key();
    if (!missing.empty()) {
      store.begin();
      for (auto& [id, b] : missing) store.set_board_key(id, canonical_board_key(parse_cards(b)));
      store.commit();
    }
    if (f.count("compact")) {
      std::printf("compacting... ");
      std::fflush(stdout);
      std::printf("%lld rows converted, database now %.2f GB\n", store.compact(), store.file_gb());
      if (count == 0) return 0;
    }
    auto removed = store.remove_duplicates();
    if (removed.first > 0)
      std::printf("removed %lld duplicate solves (%lld spots)\n", removed.first, removed.second);
    std::set<std::string> taken;  // keys used in this run (also the rejected ones)
    std::mt19937 rng(seed);
    const auto& ps = presets();
    int ok = 0, rejected = 0;
    for (int i = 0; i < count; ++i) {
      if (store.file_gb() > max_db_gb) {
        std::printf("database is %.2f GB, over --max_db_gb %.2f: stopping\n", store.file_gb(), max_db_gb);
        return 3;
      }
      const Preset& pr = preset_opt == "all" ? ps[(seed + static_cast<unsigned>(i)) % ps.size()] : [&]() -> const Preset& {
        for (const auto& p : ps)
          if (p.name == preset_opt) return p;
        throw std::invalid_argument("unknown preset: " + preset_opt);
      }();
      const bool river = street_opt == "river" || (street_opt == "mix" && rng() % 3 != 0);
      const double target = parse_number(f, "target", river ? 0.3 : 0.5);
      const std::string street = river ? "river" : "turn";
      std::vector<Card> board;
      std::string key;
      for (int tries = 0;; ++tries) {
        std::vector<int> deck(kNumCards);
        for (int c = 0; c < kNumCards; ++c) deck[c] = c;
        std::shuffle(deck.begin(), deck.end(), rng);
        board.assign(deck.begin(), deck.begin() + (river ? 5 : 4));
        key = canonical_board_key(board);
        if (!taken.count(pr.name + street + key) && !store.has_solve(pr.name, street, key)) break;
        if (tries > 2000) throw std::runtime_error("no new board found for " + pr.name + " " + street);
      }
      taken.insert(pr.name + street + key);

      std::map<std::string, std::string> fields = {
          {"board", cards_str(board)}, {"pot", std::to_string(pr.pot)}, {"stack", std::to_string(pr.stack)},
          {"oop", pr.oop}, {"ip", pr.ip}, {"bets", field(f, "bets", "33,75")}, {"raises", field(f, "raises", "60")},
          {"threads", field(f, "threads", "0")}, {"mem_gb", field(f, "mem_gb", "3")}};
      GameConfig cfg = config_from_fields(fields);
      cfg.board = board;

      auto t0 = std::chrono::steady_clock::now();
      Game g(cfg);
      double expl = 100;
      int it = 0;
      while (it < max_iters) {
        for (int k = 0; k < (river ? 25 : 20) && it < max_iters; ++k, ++it) g.run_iteration();
        expl = g.exploitability() / cfg.pot * 100;
        if (expl <= target) break;
      }
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      auto ev = g.expected_values();

      SolveRow sr;
      sr.preset = pr.name, sr.oop_pos = pr.oop_pos, sr.ip_pos = pr.ip_pos, sr.board = cards_str(board);
      sr.board_key = key, sr.street = street, sr.oop_range = pr.oop, sr.ip_range = pr.ip;
      sr.config = "bets=" + fields["bets"] + ";raises=" + fields["raises"] + ";max_raises=3;allin=1;allin_threshold=0.67;" +
                  "target=" + std::to_string(target) + ";compress=" + (g.compressed() ? "16" : "32");
      sr.solver_version = POKER_VERSION, sr.pot = pr.pot, sr.stack = pr.stack, sr.iterations = it, sr.seconds = secs;
      sr.exploitability_pct = expl, sr.ev_oop = ev[0], sr.ev_ip = ev[1], sr.memory_mb = g.memory_bytes() / 1e6;
      sr.status = expl <= target ? "ok" : "rejected";

      // Story shown to the player. Flop (and turn) are checked through in this model.
      std::vector<std::string> story = {"Preflop: " + pr.preflop};
      const std::string fl = card_str(board[0]) + card_str(board[1]) + card_str(board[2]);
      story.push_back("Flop " + fl + ": " + pr.oop_pos + " checks, " + pr.ip_pos + " checks");
      if (river) story.push_back("Turn " + card_str(board[3]) + ": " + pr.oop_pos + " checks, " + pr.ip_pos + " checks");
      story.push_back(std::string(river ? "River " : "Turn ") + card_str(board.back()));

      int stored = 0;
      store.begin();
      const long long solve_id = store.add_solve(sr);
      if (sr.status == "ok") {
        // Every decision node of this street, breadth first, up to max_depth actions deep:
        // the first action, the reply to a check or bet, facing a raise, and so on. Lines
        // through an all-in are not followed (the reply is only call or fold).
        std::vector<std::vector<std::string>> queue = {{}};
        for (size_t qi = 0; qi < queue.size(); ++qi) {
          const std::vector<std::string> path = queue[qi];
          NodeView v = g.node_view(path);
          if (v.type != "action") continue;  // street ended (chance, showdown) or fold
          if (path.size() < max_depth)
            for (size_t k = 0; k < v.actions.size(); ++k) {
              const ActKind kind = v.action_info[k].kind;
              if (kind == ActKind::Check || kind == ActKind::Bet || kind == ActKind::Raise) {
                auto next = path;
                next.push_back(std::to_string(k));
                queue.push_back(next);
              }
            }
          if (v.hands.empty()) continue;
          Scores sc = score_node(v);
          SpotRow row;
          row.solve_id = solve_id, row.path = join(path, ","), row.line = join(v.history, " > ");
          row.actor = v.player, row.actor_pos = v.player == 0 ? pr.oop_pos : pr.ip_pos;
          row.num_actions = static_cast<int>(v.actions.size()), row.combos = v.combos;
          row.mixed = sc.mixed, row.ev_loss = sc.ev_loss, row.bluff = sc.bluff, row.fundamental = sc.fundamental;
          row.category = sc.category, row.score = sc.score, row.story_json = json_str_array(story);
          row.node_json = Game::to_json(v, /*with_grid=*/false);
          store.add_spot(row);
          ++stored;
        }
        ++ok;
      } else {
        ++rejected;
      }
      store.commit();
      std::printf("[%d/%d] %-16s %-6s %-14s %4d it %6.2f s  expl %.3f%%  %s  %d spots\n", i + 1, count, pr.name.c_str(),
                  sr.street.c_str(), sr.board.c_str(), it, secs, expl, sr.status.c_str(), stored);
      std::fflush(stdout);
    }
    std::printf("done: %d ok, %d rejected\n%s\n", ok, rejected, store.stats_json().c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
