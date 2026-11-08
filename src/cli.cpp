// pokersolve: solve one spot from the command line.
//
// Example:
//   pokersolve --board "Ks 9d 5c 2h 2d" --pot 10 --stack 95 \
//     --oop "QQ-22,AJs-A2s,KTs+" --ip "AA-22,A2s+,KTo+" --bets 33,75 --target 0.3
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>

#include "game.hpp"
#include "spot.hpp"

using namespace poker;

int main(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
      std::fprintf(stderr, "usage: pokersolve --board B --pot P --stack S --oop R --ip R [--bets 33,75] "
                           "[--raises 60] [--iters N] [--target PCT] [--threads T] [--path 0,1]\n");
      return 2;
    }
    f[k.substr(2)] = argv[++i];
  }
  try {
    GameConfig cfg = config_from_fields(f);
    const int max_iters = static_cast<int>(parse_number(f, "iters", 1000));
    const double target_pct = parse_number(f, "target", 0.5);
    auto t0 = std::chrono::steady_clock::now();
    Game game(cfg);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("board %s  pot %.2f  stack %.2f\n", cards_str(cfg.board).c_str(), cfg.pot, cfg.stack);
    std::printf("hands: OOP %d, IP %d | nodes %zu (action %zu) | memory %.1f MB | build %.0f ms\n",
                game.num_hands(0), game.num_hands(1), game.num_nodes(), game.num_action_nodes(),
                game.memory_bytes() / 1e6, std::chrono::duration<double, std::milli>(t1 - t0).count());
    double expl_pct = 100;
    int next_check = 10;
    for (int it = 1; it <= max_iters; ++it) {
      game.run_iteration();
      if (it == next_check || it == max_iters) {
        expl_pct = game.exploitability() / cfg.pot * 100;
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        std::printf("iter %5d  exploitability %.3f%% of pot  (%.2f s)\n", it, expl_pct, sec);
        if (expl_pct <= target_pct) break;
        next_check += it < 100 ? 10 : 50;
      }
    }
    auto ev = game.expected_values();
    std::printf("EV: OOP %.3f  IP %.3f  (sum %.3f = pot)\n", ev[0], ev[1], ev[0] + ev[1]);
    std::vector<std::string> path;
    std::string ps = field(f, "path", "");
    std::string tok;
    for (char c : ps + ",") {
      if (c == ',') {
        if (!tok.empty()) path.push_back(tok);
        tok.clear();
      } else {
        tok += c;
      }
    }
    if (f.count("json")) std::cout << game.node_json(path) << "\n";
    return expl_pct <= target_pct ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
