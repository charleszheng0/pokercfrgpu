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
#include <functional>
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

// "OOP Bet 2.5" -> "BB bets 2.5" (position names, plain verbs).
std::string say(const std::string& h, const Preset& pr) {
  const size_t sp = h.find(' ');
  const std::string who = h.substr(0, sp) == "OOP" ? pr.oop_pos : pr.ip_pos;
  std::string act = h.substr(sp + 1), amt;
  const size_t sp2 = act.find(' ');
  if (sp2 != std::string::npos) amt = act.substr(sp2 + 1), act = act.substr(0, sp2);
  static const std::map<std::string, std::string> verb = {{"Check", "checks"}, {"Call", "calls"}, {"Bet", "bets"},
                                                          {"Raise", "raises"}, {"All-in", "goes all-in"}, {"Fold", "folds"}};
  auto it = verb.find(act);
  std::string out = who + " " + (it == verb.end() ? act : it->second);
  if (!amt.empty() && act != "Call") out += " " + amt + " bb";
  return out;
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
  const unsigned seed = static_cast<unsigned>(parse_number(f, "seed", 1));
  const std::string street_opt = field(f, "street", "river");  // river, turn, mix, flop
  const bool flop_mode = street_opt == "flop";
  const std::string preset_opt = field(f, "preset", "all");
  const int max_iters = static_cast<int>(parse_number(f, "iters", 1500));
  const double max_db_gb = parse_number(f, "max_db_gb", 4);  // stop before the disk fills
  // Stuck-flop stop: after this many seconds, stop a solve whose error made no new low in the last
  // stall_iters iterations. 0 = off. Separate limit for 3-bet presets (their solves are ~15x faster).
  const double stuck_secs = parse_number(f, "stuck_secs", 0);
  const double stuck_secs_3bet = parse_number(f, "stuck_secs_3bet", 0);
  const int stall_iters = static_cast<int>(parse_number(f, "stall_iters", 100));
  const size_t max_depth = static_cast<size_t>(parse_number(f, "depth", 4));  // actions on this street
  // Continuations: cards stored per line that ends a street, and actions deep on that street.
  const int turn_cards = static_cast<int>(parse_number(f, "turn_cards", 3));
  const size_t turn_depth = static_cast<size_t>(parse_number(f, "turn_depth", 3));
  const int river_cards = static_cast<int>(parse_number(f, "river_cards", 2));
  const size_t river_depth = static_cast<size_t>(parse_number(f, "river_depth", 2));
  // Flop solves are big: one bet size and at most one raise per street by default (see poker-metrics).
  const std::string bets = field(f, "bets", flop_mode ? "33" : "33,75");
  const std::string raises = field(f, "raises", "60");
  const std::string max_raises = field(f, "max_raises", flop_mode ? "1" : "3");
  // Flop mode: the jobs come from a list file, one "preset|flop|..." per line (pokerscreen writes it).
  std::vector<std::pair<std::string, std::string>> flop_jobs;
  if (flop_mode) {
    FILE* lf = std::fopen(field(f, "flops", "data/flop_list.txt").c_str(), "r");
    if (!lf) {
      std::fprintf(stderr, "error: cannot read the flop list (--flops)\n");
      return 2;
    }
    char buf[512];
    while (std::fgets(buf, sizeof buf, lf)) {
      std::string l = buf;
      if (l.empty() || l[0] == '#') continue;
      const size_t a = l.find('|'), b = l.find('|', a + 1);
      if (a == std::string::npos) continue;
      flop_jobs.push_back({l.substr(0, a), l.substr(a + 1, (b == std::string::npos ? l.size() : b) - a - 1)});
    }
    std::fclose(lf);
  }
  const int count = flop_mode ? static_cast<int>(parse_number(f, "count", static_cast<double>(flop_jobs.size())))
                              : static_cast<int>(parse_number(f, "count", 10));

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
    auto preset_named = [&](const std::string& name) -> const Preset& {
      for (const auto& p : ps)
        if (p.name == name) return p;
      throw std::invalid_argument("unknown preset: " + name);
    };
    for (int i = 0; i < count && (!flop_mode || i < static_cast<int>(flop_jobs.size())); ++i) {
      if (store.file_gb() > max_db_gb) {
        std::printf("database is %.2f GB, over --max_db_gb %.2f: stopping\n", store.file_gb(), max_db_gb);
        return 3;
      }
      const Preset& pr = flop_mode ? preset_named(flop_jobs[i].first)
                         : preset_opt == "all" ? ps[(seed + static_cast<unsigned>(i)) % ps.size()]
                                               : preset_named(preset_opt);
      const bool river = street_opt == "river" || (street_opt == "mix" && rng() % 3 != 0);
      const double target = parse_number(f, "target", river ? 0.3 : 0.5);
      const std::string street = flop_mode ? "flop" : river ? "river" : "turn";
      std::vector<Card> board;
      std::string key;
      if (flop_mode) {
        board = parse_cards(flop_jobs[i].second);
        key = canonical_board_key(board);
        if (store.has_solve(pr.name, street, key)) {
          std::printf("[%d/%d] %-16s flop   %-14s already solved: skip\n", i + 1, count, pr.name.c_str(), cards_str(board).c_str());
          std::fflush(stdout);
          continue;
        }
      }
      for (int tries = 0; !flop_mode; ++tries) {
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
          {"oop", pr.oop}, {"ip", pr.ip}, {"bets", bets}, {"raises", raises}, {"max_raises", max_raises},
          {"threads", field(f, "threads", "0")}, {"mem_gb", field(f, "mem_gb", flop_mode ? "5" : "3")}};
      GameConfig cfg = config_from_fields(fields);
      cfg.board = board;

      auto t0 = std::chrono::steady_clock::now();
      Game g(cfg);
      if (flop_mode) {
        std::printf("[%d/%d] %-16s flop   %-14s tree %.0f MB, solving...\n", i + 1, count, pr.name.c_str(),
                    cards_str(board).c_str(), g.memory_bytes() / 1e6);
        std::fflush(stdout);
      }
      double expl = 100, best_expl = 1e9;
      int it = 0, best_it = 0;
      const double stuck_limit = pr.name.find("3-bet") != std::string::npos ? stuck_secs_3bet : stuck_secs;
      while (it < max_iters) {
        for (int k = 0; k < (river ? 25 : 20) && it < max_iters; ++k, ++it) g.run_iteration();
        expl = g.exploitability() / cfg.pot * 100;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (flop_mode) {
          std::printf("      it %4d  expl %.3f%%  %.0f s\n", it, expl, el);
          std::fflush(stdout);
        }
        if (expl <= target) break;
        if (expl < best_expl) best_expl = expl, best_it = it;
        if (stuck_limit > 0 && el > stuck_limit && it - best_it >= stall_iters) {
          std::printf("      stuck: no new low since it %d (best %.3f%%) after %.0f s: stop\n", best_it, best_expl, el);
          std::fflush(stdout);
          break;
        }
      }
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      auto ev = g.expected_values();

      SolveRow sr;
      sr.preset = pr.name, sr.oop_pos = pr.oop_pos, sr.ip_pos = pr.ip_pos, sr.board = cards_str(board);
      sr.board_key = key, sr.street = street, sr.oop_range = pr.oop, sr.ip_range = pr.ip;
      sr.config = "bets=" + fields["bets"] + ";raises=" + fields["raises"] + ";max_raises=" + max_raises + ";allin=1;allin_threshold=0.67;" +
                  "target=" + std::to_string(target) + ";compress=" + (g.compressed() ? "16" : "32");
      sr.solver_version = POKER_VERSION, sr.pot = pr.pot, sr.stack = pr.stack, sr.iterations = it, sr.seconds = secs;
      sr.exploitability_pct = expl, sr.ev_oop = ev[0], sr.ev_ip = ev[1], sr.memory_mb = g.memory_bytes() / 1e6;
      sr.status = expl <= target ? "ok" : "rejected";

      // Story shown to the player. Flop (and turn) are checked through in this model.
      std::vector<std::string> story = {"Preflop: " + pr.preflop};
      const std::string fl = card_str(board[0]) + card_str(board[1]) + card_str(board[2]);
      if (flop_mode) {
        story.push_back("Flop " + fl);
      } else {
        story.push_back("Flop " + fl + ": " + pr.oop_pos + " checks, " + pr.ip_pos + " checks");
        if (river) story.push_back("Turn " + card_str(board[3]) + ": " + pr.oop_pos + " checks, " + pr.ip_pos + " checks");
        story.push_back(std::string(river ? "River " : "Turn ") + card_str(board.back()));
      }
      const int root_street = flop_mode ? 0 : river ? 2 : 1;  // 0 flop, 1 turn, 2 river

      int stored = 0;
      store.begin();
      const long long solve_id = store.add_solve(sr);
      if (sr.status == "ok") {
        // Store every decision node below `prefix`, breadth first, up to `depth` more actions:
        // the first action, the reply to a check or bet, facing a raise, and so on. Lines
        // through an all-in are not followed (the reply is only call or fold). Paths that end
        // the street (a call, or a check behind) go to `ends`.
        auto store_street = [&](const std::vector<std::string>& prefix, size_t depth, const std::vector<std::string>& st,
                                std::vector<std::vector<std::string>>* ends) {
          std::vector<std::vector<std::string>> queue = {prefix};
          for (size_t qi = 0; qi < queue.size(); ++qi) {
            const std::vector<std::string> path = queue[qi];
            NodeView v = g.node_view(path);
            if (v.type != "action") continue;  // street ended (chance, showdown) or fold
            for (size_t k = 0; k < v.actions.size(); ++k) {
              const ActKind kind = v.action_info[k].kind;
              auto next = path;
              next.push_back(std::to_string(k));
              if (path.size() < prefix.size() + depth && (kind == ActKind::Check || kind == ActKind::Bet || kind == ActKind::Raise))
                queue.push_back(next);
              if (ends && path.size() < prefix.size() + 3 && (kind == ActKind::Call || kind == ActKind::Check) &&
                  g.node_view(next).type == "chance")
                ends->push_back(next);
            }
            if (v.hands.empty()) continue;
            Scores sc = score_node(v);
            SpotRow row;
            row.solve_id = solve_id, row.path = join(path, ","), row.line = join(v.history, " > ");
            row.actor = v.player, row.actor_pos = v.player == 0 ? pr.oop_pos : pr.ip_pos;
            row.num_actions = static_cast<int>(v.actions.size()), row.combos = v.combos;
            row.mixed = sc.mixed, row.ev_loss = sc.ev_loss, row.bluff = sc.bluff, row.fundamental = sc.fundamental;
            row.category = sc.category, row.score = sc.score, row.story_json = json_str_array(st);
            row.node_json = Game::to_json(v, /*with_grid=*/false);
            store.add_spot(row);
            ++stored;
          }
        };
        // Continue the common lines (up to 3 actions) onto the next street, for a few random cards
        // each, so the trainer can play a hand to the end: flop -> turn_cards turns -> river_cards rivers.
        static const char* kStreet[3] = {"Flop", "Turn", "River"};
        std::function<void(const std::vector<std::string>&, int, const std::vector<std::string>&)> run_street;
        run_street = [&](const std::vector<std::string>& prefix, int sidx, const std::vector<std::string>& st) {
          std::vector<std::vector<std::string>> ends;
          const size_t depth = prefix.empty() ? max_depth : sidx == 1 ? turn_depth : river_depth;
          store_street(prefix, depth, st, sidx == 2 ? nullptr : &ends);
          const int ncards = sidx == 0 ? turn_cards : river_cards;
          for (const auto& end : ends) {
            NodeView chance = g.node_view(end);
            // Story: the root story without its bare last line, then one line per street played.
            std::vector<std::vector<std::string>> seg(1);
            std::vector<std::string> dealt;
            for (const std::string& h : chance.history) {
              if (h.rfind("deal ", 0) == 0) seg.emplace_back(), dealt.push_back(h.substr(5));
              else seg.back().push_back(say(h, pr));
            }
            std::vector<std::string> base(story.begin(), story.end() - 1);
            for (size_t s2 = 0; s2 < seg.size(); ++s2) {
              const std::string label = s2 == 0 ? story.back() : std::string(kStreet[root_street + s2]) + " " + dealt[s2 - 1];
              base.push_back(label + ": " + join(seg[s2], ", "));
            }
            std::vector<Card> cards = chance.cards;
            std::shuffle(cards.begin(), cards.end(), rng);
            for (int c = 0; c < ncards && c < static_cast<int>(cards.size()); ++c) {
              auto st2 = base;
              st2.push_back(std::string(kStreet[sidx + 1]) + " " + card_str(cards[c]));
              auto next = end;
              next.push_back(card_str(cards[c]));
              run_street(next, sidx + 1, st2);
            }
          }
        };
        run_street({}, root_street, story);
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
