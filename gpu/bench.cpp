// pokergpu: benchmark the level-by-level solver (CUDA or CPU back end) against Game (CPU DCFR).
//
//   pokergpu --spot river|turn|flop3bet|flopsrp [--target 0.5] [--iters 1000] [--cpu 1]
//
// Prints, for each solver: iterations, seconds (iterations only, checks not counted), and the
// exploitability in % of the pot, every --step iterations.
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <map>
#include <string>
#include <thread>

#include "flat_solver.hpp"
#include "game.hpp"
#include "presets.hpp"
#include "spot.hpp"

using namespace poker;
using Clock = std::chrono::steady_clock;

static GameConfig spot_config(const std::string& spot) {
  const auto& ps = presets();
  const Preset& srp = ps[0];   // BTN vs BB SRP
  const Preset& b3 = ps[4];    // BB 3-bet vs BTN
  std::map<std::string, std::string> f;
  auto base = [&](const Preset& pr, const std::string& board, const std::string& bets, const std::string& mr) {
    f = {{"board", board}, {"pot", std::to_string(pr.pot)}, {"stack", std::to_string(pr.stack)}, {"oop", pr.oop},
         {"ip", pr.ip}, {"bets", bets}, {"raises", "60"}, {"max_raises", mr}, {"threads", "0"}, {"mem_gb", "12"},
         {"compress", "0"}};
  };
  if (spot == "river") base(srp, "Ks 9d 5c 2h 2d", "33,75", "3");
  else if (spot == "turn") base(srp, "Ks 9d 5c 2h", "33,75", "3");
  else if (spot == "flop3bet") base(b3, "Ac 9d 8c", "33", "1");
  else if (spot == "flopsrp") base(srp, "Jc Td 3d", "33", "1");
  else throw std::invalid_argument("unknown --spot " + spot);
  return config_from_fields(f);
}

template <class S>
static void run(const char* name, S& s, double pot, double target, int iters, int step) {
  double secs = 0, e = 100;
  int it = 0;
  while (it < iters) {
    const auto t0 = Clock::now();
    for (int k = 0; k < step && it < iters; ++k, ++it) s.run_iteration();
    secs += std::chrono::duration<double>(Clock::now() - t0).count();
    e = s.exploitability() / pot * 100;
    std::printf("  %-10s it %5d  %8.2f s  expl %7.3f%%\n", name, it, secs, e);
    std::fflush(stdout);
    if (e <= target) break;
  }
  std::printf("RESULT %s it=%d secs=%.3f expl=%.4f\n", name, it, secs, e);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  try {
    const std::string spot = field(f, "spot", "river");
    const double target = parse_number(f, "target", 0.5);
    const int iters = static_cast<int>(parse_number(f, "iters", 1000));
    const int step = static_cast<int>(parse_number(f, "step", 20));
    GameConfig cfg = spot_config(spot);
    std::printf("spot %s: building tree...\n", spot.c_str());
    Game g(cfg);
    std::printf("  %zu nodes, %d / %d hands, regrets %.2f GB (32-bit)\n", g.num_nodes(), g.num_hands(0), g.num_hands(1),
                g.memory_bytes() / 1e9);
    const double need = 2.0 * g.num_nodes() * std::max(g.num_hands(0), g.num_hands(1)) * 4 / 1e9 + 2 * g.memory_bytes() / 1e9;
    std::printf("  level-by-level solver needs about %.2f GB\n", need);
    if (parse_number(f, "dry", 0) != 0) return 0;
    if (parse_number(f, "flat", 1) != 0) {
      const FlatTree t = g.export_flat();
      FlatSolver fs(t);
      std::printf("level-by-level solver (%s back end), %.2f GB:\n", FlatSolver::backend(), fs.memory_gb());
      if (parse_number(f, "profile", 0) != 0) fs.set_profile(true);
      run(FlatSolver::backend(), fs, cfg.pot, target, iters, step);
      if (parse_number(f, "profile", 0) != 0) std::printf("profile:\n%s", fs.profile_report().c_str());
    }
    if (parse_number(f, "cpu", 1) != 0) {
      std::printf("Game (CPU DCFR, depth-first, %u threads):\n", std::thread::hardware_concurrency());
      run("cpu-dfs", g, cfg.pot, target, iters, step);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
