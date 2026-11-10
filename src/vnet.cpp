// pokervnet: depth-limited flop solving with a leaf value network (DeepStack / ReBeL style).
//
//   pokervnet gen   --out data/vnet --count 60 --seed 7      solve random flops, store leaf data
//   pokervnet train --data data/vnet --net data/vnet.net     train the value network
//   pokervnet eval  --preset "BB 3-bet vs BTN" --flop QcJd5d --net data/vnet.net
//
// The tree config is the one of the flop database (pokerbatch --street flop): bet 33%,
// raise 60%, one raise per street, all-in allowed.
#include <algorithm>
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
#include "value_net.hpp"
#include "vnet_data.hpp"

using namespace poker;

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

const Preset& preset_named(const std::string& name) {
  for (const auto& p : presets())
    if (p.name == name) return p;
  throw std::invalid_argument("unknown preset: " + name);
}

GameConfig flop_config(const Preset& pr, const std::vector<Card>& board, int depth_limit, int compress = -1) {
  std::map<std::string, std::string> fields = {
      {"board", cards_str(board)}, {"pot", std::to_string(pr.pot)}, {"stack", std::to_string(pr.stack)},
      {"oop", pr.oop},  {"ip", pr.ip}, {"bets", "33"}, {"raises", "60"}, {"max_raises", "1"},
      {"threads", "0"}, {"mem_gb", "5"}};
  GameConfig cfg = config_from_fields(fields);
  cfg.board = board;
  cfg.depth_limit = depth_limit;
  cfg.compress = compress;
  return cfg;
}

bool is_3bet(const Preset& pr) { return pr.name.find("3-bet") != std::string::npos; }

// Solve a full tree until the target exploitability (% of pot) or max_iters.
struct SolveLog {
  std::vector<int> iters;
  std::vector<double> secs, expl;
  std::vector<std::vector<float>> flop_strategy;  // first-street strategy at each check
};
double solve(Game& g, double pot, double target, int max_iters, int step, SolveLog* log, bool verbose) {
  const auto t0 = Clock::now();
  double expl = 100;
  int it = 0;
  double el_solve = 0;  // time in iterations only (exploitability checks not counted)
  while (it < max_iters) {
    const auto t1 = Clock::now();
    for (int k = 0; k < step && it < max_iters; ++k, ++it) g.run_iteration();
    el_solve += since(t1);
    expl = g.exploitability() / pot * 100;
    if (log) {
      log->iters.push_back(it);
      log->secs.push_back(el_solve);
      log->expl.push_back(expl);
      log->flop_strategy.push_back(g.first_street_strategy());
    }
    if (verbose) {
      std::printf("      it %4d  expl %7.3f%%  %6.1f s solve (%6.1f s wall)\n", it, expl, el_solve, since(t0));
      std::fflush(stdout);
    }
    if (expl <= target) break;
  }
  return expl;
}

std::set<std::string> list_flop_keys(const std::string& path) {
  std::set<std::string> keys;
  FILE* lf = std::fopen(path.c_str(), "r");
  if (!lf) return keys;
  char buf[512];
  while (std::fgets(buf, sizeof buf, lf)) {
    std::string l = buf;
    if (l.empty() || l[0] == '#') continue;
    const size_t a = l.find('|'), b = l.find('|', a + 1);
    if (a == std::string::npos) continue;
    keys.insert(canonical_board_key(parse_cards(l.substr(a + 1, (b == std::string::npos ? l.size() : b) - a - 1))));
  }
  std::fclose(lf);
  return keys;
}

// ---------------------------------------------------------------- gen

int cmd_gen(const std::map<std::string, std::string>& f) {
  const std::string out = field(f, "out", "data/vnet");
  const int count = static_cast<int>(parse_number(f, "count", 60));
  const unsigned seed = static_cast<unsigned>(parse_number(f, "seed", 7));
  const double target_srp = parse_number(f, "target", 1.0), target_3b = parse_number(f, "target3", 0.5);
  const int max_iters = static_cast<int>(parse_number(f, "iters", 400));
  // Test flops (the flop database list) are never used for training.
  const std::set<std::string> held_out = list_flop_keys(field(f, "exclude", "data/flop_list.txt"));
  std::mt19937 rng(seed);
  const auto& ps = presets();
  std::set<std::string> done;
  for (int i = 0; i < count; ++i) {
    // Two 3-bet flops for each single-raised flop: 3-bet solves are ~15x faster.
    const Preset& pr = f.count("preset") ? preset_named(f.at("preset"))
                                          : ps[i % 3 == 0 ? (i / 3) % 4 : 4 + (i % 3 == 1 ? 0 : 1)];
    std::vector<Card> board;
    std::string key;
    if (f.count("flop")) board = parse_cards(f.at("flop")), key = canonical_board_key(board);
    for (; !f.count("flop");) {
      std::vector<int> deck(kNumCards);
      for (int c = 0; c < kNumCards; ++c) deck[c] = c;
      std::shuffle(deck.begin(), deck.end(), rng);
      board.assign(deck.begin(), deck.begin() + 3);
      key = canonical_board_key(board);
      if (!held_out.count(key) && !done.count(pr.name + key)) break;
    }
    done.insert(pr.name + key);
    std::string name = pr.name + "_" + cards_str(board);
    for (char& c : name)
      if (c == ' ') c = '_';
    const std::string path = out + "/" + name + ".bin";
    if (FILE* e = std::fopen(path.c_str(), "rb")) {
      std::fclose(e);
      continue;
    }
    const auto t0 = Clock::now();
    Game g(flop_config(pr, board, 0, static_cast<int>(parse_number(f, "compress", 1))));
    std::printf("[%d/%d] %-16s %-8s tree %5.0f MB, solving...\n", i + 1, count, pr.name.c_str(), cards_str(board).c_str(),
                g.memory_bytes() / 1e6);
    std::fflush(stdout);
    const double expl = solve(g, pr.pot, is_3bet(pr) ? target_3b : target_srp, max_iters, 20, nullptr, false);
    auto samples = g.street_end_samples();
    VnetFile d;
    d.preset = pr.name, d.board = board, d.pot = pr.pot, d.stack = pr.stack, d.expl_pct = expl;
    d.iterations = g.iterations();
    for (int p = 0; p < 2; ++p)
      for (const auto& h : g.hands(p)) d.combos[p].push_back(h.combo);
    for (auto& s : samples) {
      VnetLeaf l;
      l.contrib = s.info.contrib, l.stack_left = s.info.stack_left, l.all_in = s.all_in;
      for (int p = 0; p < 2; ++p) l.reach[p] = std::move(s.reach[p]), l.value[p] = std::move(s.value[p]);
      d.leaves.push_back(std::move(l));
    }
    write_vnet_file(path, d);
    std::printf("[%d/%d] %-16s %-8s %4d it  expl %.3f%%  %6.1f s  %zu leaves\n", i + 1, count, pr.name.c_str(),
                cards_str(board).c_str(), g.iterations(), expl, since(t0), d.leaves.size());
    std::fflush(stdout);
  }
  return 0;
}

// ---------------------------------------------------------------- train

// Rows for one data file: features and targets (y - d) of every hand with compatible
// opponent reach, at every leaf that is not all-in.
void file_rows(const VnetFile& d, std::vector<float>& X, std::vector<float>& Y) {
  std::vector<std::pair<Card, Card>> hs[2];
  for (int p = 0; p < 2; ++p)
    for (uint16_t c : d.combos[p]) hs[p].push_back(combo_cards(c));
  FlopRunouts R(d.board, hs);
  LeafFeatures F;
  for (const auto& l : d.leaves) {
    if (l.all_in) continue;
    const float* const r[2] = {l.reach[0].data(), l.reach[1].data()};
    compute_leaf_features(R, r, l.contrib, l.stack_left, F);
    for (int p = 0; p < 2; ++p) {
      double tot = 0;
      for (float x : l.reach[1 - p]) tot += x;
      for (size_t h = 0; h < hs[p].size(); ++h) {
        if (F.mass[p][h] <= 1e-6 * tot) continue;
        const double y = l.value[p][h] / (l.contrib * F.mass[p][h]);
        X.insert(X.end(), &F.x[p][h * kLeafFeatures], &F.x[p][(h + 1) * kLeafFeatures]);
        Y.push_back(static_cast<float>(std::clamp(y - F.d[p][h], -4.0, 4.0)));
      }
    }
  }
}

std::vector<std::string> list_bin_files(const std::string& dir) {
  std::vector<std::string> out;
  std::string cmd = "ls " + dir + "/*.bin 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return out;
  char buf[1024];
  while (std::fgets(buf, sizeof buf, p)) {
    std::string l = buf;
    while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
    if (!l.empty()) out.push_back(l);
  }
  pclose(p);
  std::sort(out.begin(), out.end());
  return out;
}

int cmd_train(const std::map<std::string, std::string>& f) {
  const std::string dir = field(f, "data", "data/vnet"), out = field(f, "net", "data/vnet.net");
  const int valid_every = static_cast<int>(parse_number(f, "valid_every", 8));  // every k-th file: validation
  const int hidden = static_cast<int>(parse_number(f, "hidden", 64));
  const int layers = static_cast<int>(parse_number(f, "layers", 2));
  TrainOptions opt;
  opt.epochs = static_cast<int>(parse_number(f, "epochs", 30));
  opt.lr = parse_number(f, "lr", 1e-3);
  opt.batch = static_cast<int>(parse_number(f, "batch", 1024));
  std::vector<std::string> files;
  {
    std::string rest = dir;
    while (!rest.empty()) {
      const size_t c = rest.find(',');
      const auto part = list_bin_files(rest.substr(0, c));
      files.insert(files.end(), part.begin(), part.end());
      rest = c == std::string::npos ? "" : rest.substr(c + 1);
    }
  }
  if (files.empty()) throw std::runtime_error("no data files in " + dir);
  std::vector<float> X, Y, Xv, Yv;
  int nt = 0, nv = 0;
  const auto t0 = Clock::now();
  for (size_t i = 0; i < files.size(); ++i) {
    const VnetFile d = read_vnet_file(files[i]);
    const bool valid = valid_every > 0 && static_cast<int>(i % valid_every) == valid_every - 1;
    file_rows(d, valid ? Xv : X, valid ? Yv : Y);
    (valid ? nv : nt)++;
  }
  std::printf("%d train files (%zu rows), %d validation files (%zu rows), features %.1f s\n", nt, Y.size(), nv,
              Yv.size(), since(t0));
  auto stats = [](const std::vector<float>& y) {
    double s = 0, a = 0;
    for (float v : y) s += static_cast<double>(v) * v, a += std::fabs(v);
    return std::make_pair(y.empty() ? 0 : s / y.size(), y.empty() ? 0 : a / y.size());
  };
  const auto bt = stats(Y);
  std::printf("check-down only (no network): train mse %.5f, mean abs error %.2f%% of the leaf pot\n", bt.first,
              bt.second * 50);
  const auto bv = stats(Yv);
  std::printf("check-down only (no network): valid mse %.5f, mean abs error %.2f%% of the leaf pot\n", bv.first,
              bv.second * 50);
  std::vector<int> sizes = {kLeafFeatures};
  for (int l = 0; l < layers; ++l) sizes.push_back(hidden);
  sizes.push_back(1);
  Mlp net;
  net.init(sizes, 1);
  std::printf("network %d", kLeafFeatures);
  for (size_t l = 1; l < sizes.size(); ++l) std::printf("-%d", sizes[l]);
  std::printf(", %zu parameters\n", net.num_params());
  const auto t1 = Clock::now();
  const double tr = train_mlp(net, X, Y, Xv, Yv, opt);
  std::printf("network: train mse %.5f\n", tr);
  double se = 0, ae = 0;
  for (size_t i = 0; i < Yv.size(); ++i) {
    const double e = net.forward(&Xv[i * kLeafFeatures]) - Yv[i];
    se += e * e, ae += std::fabs(e);
  }
  if (!Yv.empty())
    std::printf("network: valid mse %.5f, mean abs error %.2f%% of the leaf pot (train %.0f s)\n", se / Yv.size(),
                ae / Yv.size() * 50, since(t1));
  net.save(out);
  std::printf("saved %s\n", out.c_str());
  return 0;
}

// ---------------------------------------------------------------- eval

struct DlResult {
  double secs = 0, internal_expl = 0;
  std::vector<float> flop;
};

DlResult depth_limited_solve(const Preset& pr, const std::vector<Card>& board, const Mlp* net, int iters) {
  const auto t0 = Clock::now();
  Game g(flop_config(pr, board, 1));
  NetLeafEvaluator ev(g, net);
  g.set_leaf_evaluator(&ev);
  for (int i = 0; i < iters; ++i) g.run_iteration();
  DlResult r;
  r.secs = since(t0);
  r.internal_expl = g.exploitability() / pr.pot * 100;
  r.flop = g.first_street_strategy();
  return r;
}

// Exploitability (% of pot) of a first-street strategy: lock it into the full tree, solve the
// rest, and keep the lowest exploitability seen (an upper bound for the locked strategy).
double locked_exploitability(Game& g, const std::vector<float>& flop, double pot, int max_iters, double* secs) {
  const auto t0 = Clock::now();
  g.reset();
  g.lock_first_street(flop);
  double best = 1e9, prev = 1e9;
  for (int it = 0; it < max_iters;) {
    for (int k = 0; k < 20 && it < max_iters; ++k, ++it) g.run_iteration();
    const double e = g.exploitability() / pot * 100;
    best = std::min(best, e);
    if (it >= 100 && prev - e < 0.01) break;  // the rest of the tree has converged
    prev = e;
  }
  if (secs) *secs = since(t0);
  return best;
}

int cmd_eval(const std::map<std::string, std::string>& f) {
  const Preset& pr = preset_named(field(f, "preset", "BB 3-bet vs BTN"));
  const std::vector<Card> board = parse_cards(field(f, "flop", "QcJd5d"));
  const int dl_iters = static_cast<int>(parse_number(f, "dl_iters", 300));
  const int full_iters = static_cast<int>(parse_number(f, "iters", 600));
  const double target = parse_number(f, "target", 0.5);
  const int lock_iters = static_cast<int>(parse_number(f, "lock_iters", 600));
  Mlp net;
  const std::string net_path = field(f, "net", "data/vnet.net");
  const bool have_net = net_path != "none";
  if (have_net) net.load(net_path);
  const bool dl_only = field(f, "dl_only", "0") == "1";  // timing of the depth-limited solves only
  std::printf("== %s  %s\n", pr.name.c_str(), cards_str(board).c_str());

  std::printf("depth-limited, check-down leaves ... ");
  std::fflush(stdout);
  const DlResult eq = depth_limited_solve(pr, board, nullptr, dl_iters);
  std::printf("%.1f s\n", eq.secs);
  std::printf("depth-limited, value network leaves ... ");
  std::fflush(stdout);
  const DlResult nn = depth_limited_solve(pr, board, have_net ? &net : nullptr, dl_iters);
  std::printf("%.1f s\n", nn.secs);
  if (dl_only) return 0;
  if (field(f, "only_nn", "0") == "1") {  // score the network strategy only (no DCFR baseline)
    Game g(flop_config(pr, board, 0));
    double ls = 0;
    const double e = locked_exploitability(g, nn.flop, pr.pot, lock_iters, &ls);
    std::printf("depth-limited + value net: flop-locked expl %.3f%% (DL internal %.3f%%, lock check %.0f s)\n", e,
                nn.internal_expl, ls);
    return 0;
  }

  std::printf("full DCFR to %.2f%%:\n", target);
  Game g(flop_config(pr, board, 0));
  SolveLog log;
  const double full_expl = solve(g, pr.pot, target, full_iters, 20, &log, true);
  const double full_secs = log.secs.back();
  // DCFR flop strategy at the same solve time as the network version.
  size_t k_same = 0;
  while (k_same + 1 < log.secs.size() && log.secs[k_same] < nn.secs) ++k_same;

  struct Cand {
    std::string name;
    double secs;
    const std::vector<float>* flop;
  };
  std::vector<Cand> cands = {{"depth-limited + value net", nn.secs, &nn.flop},
                             {"depth-limited + check-down", eq.secs, &eq.flop},
                             {"DCFR at it " + std::to_string(log.iters[k_same]), log.secs[k_same], &log.flop_strategy[k_same]},
                             {"DCFR full (it " + std::to_string(log.iters.back()) + ")", full_secs, &log.flop_strategy.back()}};
  std::printf("\n%-30s %9s %22s\n", "flop strategy from", "solve s", "flop-locked expl % pot");
  for (const auto& c : cands) {
    double ls = 0;
    const double e = locked_exploitability(g, *c.flop, pr.pot, lock_iters, &ls);
    std::printf("%-30s %9.1f %21.3f%%   (lock check %.0f s)\n", c.name.c_str(), c.secs, e, ls);
    std::fflush(stdout);
  }
  std::printf("full DCFR whole-tree exploitability at the end: %.3f%% after %.1f s\n", full_expl, full_secs);
  // Time for full DCFR to reach the whole-tree exploitability that each depth-limited strategy has.
  std::printf("RESULT %s|%s|nn_secs=%.2f|eq_secs=%.2f|full_secs=%.2f|full_expl=%.3f\n", pr.name.c_str(),
              cards_str(board).c_str(), nn.secs, eq.secs, full_secs, full_expl);
  for (size_t k = 0; k < log.iters.size(); ++k)
    std::printf("CURVE %d %.2f %.3f\n", log.iters[k], log.secs[k], log.expl[k]);
  return 0;
}

// ---------------------------------------------------------------- gen2: random ranges

// One leaf per solve: the flop is checked through with random ranges and a random pot, then
// the turn and river are solved in full. Ranges: the preset ranges, reweighted by a random
// function of each hand's check-down value (like flop betting does), with random noise.
void shape_range(std::vector<float>& w, const std::vector<float>& d, std::mt19937& rng) {
  std::uniform_real_distribution<double> U(0, 1);
  const int type = static_cast<int>(U(rng) * 5);
  const double c = -0.6 + 1.2 * U(rng), k = 2 + 10 * U(rng), width = 0.2 + 0.6 * U(rng);
  std::normal_distribution<double> noise(0, 0.4);
  float mx = 0;
  for (size_t h = 0; h < w.size(); ++h) {
    const double x = d[h];
    double f = 1;
    switch (type) {
      case 0: f = 1; break;                                                       // unchanged
      case 1: f = 1 / (1 + std::exp(-k * (x - c))); break;                        // strong hands stay
      case 2: f = 1 / (1 + std::exp(k * (x - c))); break;                         // capped: strong hands leave
      case 3: f = std::exp(-(x - c) * (x - c) / (2 * width * width)); break;      // condensed
      default: f = 1 - 0.95 * std::exp(-(x - c) * (x - c) / (2 * width * width));  // polarized
    }
    w[h] = static_cast<float>(w[h] * (0.02 + f) * std::exp(noise(rng)));
    mx = std::max(mx, w[h]);
  }
  if (mx > 0)
    for (auto& x : w) x /= mx;
}

int cmd_gen2(const std::map<std::string, std::string>& f) {
  const std::string out = field(f, "out", "data/vnet2");
  const int count = static_cast<int>(parse_number(f, "count", 1000));
  const unsigned seed = static_cast<unsigned>(parse_number(f, "seed", 11));
  const double target = parse_number(f, "target", 0.7);
  const int threads = static_cast<int>(parse_number(f, "threads", 0));
  const std::set<std::string> held_out = list_flop_keys(field(f, "exclude", "data/flop_list.txt"));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0, 1);
  const auto& ps = presets();
  for (int i = 0; i < count; ++i) {
    const Preset& pr = ps[rng() % ps.size()];
    std::vector<Card> board;
    for (;;) {
      std::vector<int> deck(kNumCards);
      for (int c = 0; c < kNumCards; ++c) deck[c] = c;
      std::shuffle(deck.begin(), deck.end(), rng);
      board.assign(deck.begin(), deck.begin() + 3);
      if (!held_out.count(canonical_board_key(board))) break;
    }
    // Pot at the end of the flop: 1x (checked) to 4.5x the starting pot, log-uniform.
    const double mult = std::exp(U(rng) * std::log(4.5));
    const double pot = pr.pot * mult, stack = pr.stack - (pot - pr.pot) / 2;
    if (stack < 0.1 * pot) continue;
    GameConfig cfg = flop_config(pr, board, 0);
    cfg.pot = pot, cfg.stack = stack, cfg.check_first_street = true, cfg.threads = threads;
    // Hands of the base ranges, check-down values, random reshaping.
    std::vector<std::pair<Card, Card>> hs[2];
    std::vector<int> combo[2];
    std::vector<float> w[2];
    uint64_t used = 0;
    for (Card c : board) used |= 1ULL << c;
    for (int p = 0; p < 2; ++p)
      for (int ci = 0; ci < kNumCombos; ++ci) {
        const float x = cfg.range[p].weight[ci];
        auto cc = combo_cards(ci);
        if (x <= 0 || (used >> cc.first & 1) || (used >> cc.second & 1)) continue;
        hs[p].push_back(cc), combo[p].push_back(ci), w[p].push_back(x);
      }
    FlopRunouts R(board, hs);
    LeafFeatures F;
    const float* const r0[2] = {w[0].data(), w[1].data()};
    compute_leaf_features(R, r0, pot / 2, stack, F);
    for (int p = 0; p < 2; ++p) {
      shape_range(w[p], F.d[p], rng);
      for (int ci = 0; ci < kNumCombos; ++ci) cfg.range[p].weight[ci] = 0;
      for (size_t h = 0; h < w[p].size(); ++h) cfg.range[p].weight[combo[p][h]] = std::max(w[p][h], 1e-4f);
    }
    char name[256];
    std::snprintf(name, sizeof name, "%s/r%05d_%s.bin", out.c_str(), i, cards_str(board).c_str());
    if (FILE* e = std::fopen(name, "rb")) {
      std::fclose(e);
      continue;
    }
    const auto t0 = Clock::now();
    Game g(cfg);
    const bool verbose = field(f, "verbose", "0") == "1";
    if (verbose) std::printf("[%d] %s %s tree %.0f MB\n", i, pr.name.c_str(), cards_str(board).c_str(), g.memory_bytes() / 1e6);
    const double expl = solve(g, pot, target, 400, 20, nullptr, verbose);
    auto samples = g.street_end_samples();
    VnetFile d;
    d.preset = "rand:" + pr.name, d.board = board, d.pot = pot, d.stack = stack, d.expl_pct = expl;
    d.iterations = g.iterations();
    for (int p = 0; p < 2; ++p)
      for (const auto& h : g.hands(p)) d.combos[p].push_back(h.combo);
    for (auto& s : samples) {
      VnetLeaf l;
      l.contrib = s.info.contrib, l.stack_left = s.info.stack_left, l.all_in = s.all_in;
      for (int p = 0; p < 2; ++p) l.reach[p] = std::move(s.reach[p]), l.value[p] = std::move(s.value[p]);
      d.leaves.push_back(std::move(l));
    }
    write_vnet_file(name, d);
    std::printf("[%d] %-16s %-8s pot %5.1f stack %5.1f  %3d it  expl %.3f%%  %5.1f s  %.0f MB\n", i, pr.name.c_str(),
                cards_str(board).c_str(), pot, stack, g.iterations(), expl, since(t0), g.memory_bytes() / 1e6);
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: pokervnet gen|train|eval [--key value]...\n");
    return 2;
  }
  const std::string cmd = argv[1];
  std::map<std::string, std::string> f;
  for (int i = 2; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  try {
    if (cmd == "gen") return cmd_gen(f);
    if (cmd == "gen2") return cmd_gen2(f);
    if (cmd == "train") return cmd_train(f);
    if (cmd == "eval") return cmd_eval(f);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
  return 2;
}

