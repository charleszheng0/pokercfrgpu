// Acceptance tests from SPEC.md. Run: ./build/tests
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>

#include "game.hpp"
#include "spot.hpp"

using namespace poker;

static int g_fail = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    if (!(cond)) {                                        \
      ++g_fail;                                           \
      std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);  \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

static double secs_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

static uint32_t eval_str(const std::string& s) {
  auto c = parse_cards(s);
  return eval_hand(c.data(), static_cast<int>(c.size()));
}

void test_evaluator() {
  std::printf("[1] evaluator: all 2,598,960 five-card hands\n");
  long count[9] = {0};
  Card h[5];
  for (int a = 0; a < 52; ++a)
    for (int b = a + 1; b < 52; ++b)
      for (int c = b + 1; c < 52; ++c)
        for (int d = c + 1; d < 52; ++d)
          for (int e = d + 1; e < 52; ++e) {
            h[0] = Card(a), h[1] = Card(b), h[2] = Card(c), h[3] = Card(d), h[4] = Card(e);
            ++count[hand_category(eval_hand(h, 5))];
          }
  const long expect[9] = {1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40};
  for (int i = 0; i < 9; ++i)
    CHECK(count[i] == expect[i], "%s: %ld, expected %ld", category_name(i), count[i], expect[i]);
  CHECK(eval_str("Ah2c3d4s5h") < eval_str("2h3c4d5s6h"), "wheel must lose to six-high straight");
  CHECK(eval_str("AhAcKdKs2h") > eval_str("AhAcQdQsKh"), "AAKK beats AAQQ");
  CHECK(eval_str("AhAcKdQs9h") > eval_str("AhAcKdQs8h"), "kicker");
  CHECK(eval_str("KhKcKdQsQh2c3d") == eval_str("KhKcKdQsQh"), "7-card same as best 5");
  CHECK(eval_str("9h9c9d8s8h8c2d") > eval_str("9h9c9d7s7h7c2d"), "trips+trips full house picks best pair");
}

void test_ranges() {
  std::printf("[2] range parser\n");
  CHECK(parse_range("AA").num_combos() == 6, "AA");
  CHECK(parse_range("AKs").num_combos() == 4, "AKs");
  CHECK(parse_range("AKo").num_combos() == 12, "AKo");
  CHECK(parse_range("AK").num_combos() == 16, "AK");
  CHECK(parse_range("22+").num_combos() == 78, "22+");
  CHECK(parse_range("A2s+").num_combos() == 48, "A2s+");
  CHECK(parse_range("QQ-99").num_combos() == 24, "QQ-99");
  CHECK(parse_range("A5s-A2s").num_combos() == 16, "A5s-A2s");
  CHECK(parse_range("AhKh").num_combos() == 1, "AhKh");
  Range r = parse_range("AA,KK:0.25");
  CHECK(std::fabs(r.weight[combo_index(make_card(11, 0), make_card(11, 1))] - 0.25f) < 1e-6, "weight");
  bool threw = false;
  try {
    parse_range("AXs");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw, "bad rank must throw");
  CHECK(grid_label(grid_cell(combo_index(make_card(12, 3), make_card(11, 3)))) == "AKs", "grid AKs");
  CHECK(grid_label(grid_cell(combo_index(make_card(12, 3), make_card(11, 2)))) == "AKo", "grid AKo");
  // board cards are removed when the game is built
  GameConfig cfg;
  cfg.board = parse_cards("Ah Kd 7c 2s 3h");
  cfg.range[0] = parse_range("AA");
  cfg.range[1] = parse_range("KK");
  cfg.sizes[0][2].bet = {1.0};
  Game g(cfg);
  CHECK(g.num_hands(0) == 3 && g.num_hands(1) == 3, "AA and KK lose 3 combos each to the board");
}

// Polarized river game: OOP has nuts (77, a set) or air (65s), IP has bluff-catchers (QJs).
// OOP may bet pot; IP may only call or fold. Theory: IP calls 50%; bluffs are 1/3 of bets.
void test_toy_river() {
  std::printf("[3] toy river game (nuts + air vs bluff-catchers)\n");
  GameConfig cfg;
  cfg.board = parse_cards("Ks Qd 7h 4c 2d");
  cfg.pot = 10;
  cfg.stack = 100;
  cfg.range[0] = parse_range("77,65s");
  cfg.range[1] = parse_range("QJs");
  cfg.sizes[0][2].bet = {1.0};  // only OOP may bet
  cfg.add_allin = false;
  cfg.max_raises = 0;
  Game g(cfg);
  for (int i = 0; i < 2000; ++i) g.run_iteration();
  double expl = g.exploitability() / cfg.pot * 100;
  std::string root = g.node_json({});
  std::string facing = g.node_json({"1"});  // OOP bets
  std::printf("  exploitability %.4f%% of pot\n", expl);
  CHECK(expl < 0.1, "exploitability %.4f%% >= 0.1%%", expl);

  // Read frequencies from the JSON (simple scan, no JSON library).
  auto read_freq = [](const std::string& js) {
    std::vector<double> v;
    size_t p = js.find("\"freq\":[");
    if (p == std::string::npos) return v;
    p += 8;
    while (js[p] != ']') {
      v.push_back(std::strtod(js.c_str() + p, nullptr));
      p = js.find_first_of(",]", p);
      if (js[p] == ',') ++p;
    }
    return v;
  };
  // Facing the bet: actions are Fold, Call.
  auto f = read_freq(facing);
  CHECK(f.size() == 2, "IP should have fold/call");
  if (f.size() == 2) {
    std::printf("  IP call frequency %.3f (theory 0.500)\n", f[1]);
    CHECK(std::fabs(f[1] - 0.5) < 0.05, "call freq %.3f not near 0.5", f[1]);
  }
  // Bluff share: OOP air combos betting / all OOP combos betting.
  // 77 has 3 combos (7h on board), 65s has 4. Bet frequency over the root = (3 + b) / 7.
  auto r = read_freq(root);
  if (r.size() == 2) {
    double bets = r[1] * 7.0;
    double bluffs = bets - 3.0;
    double share = bluffs / bets;
    std::printf("  OOP bets %.2f combos, bluff share %.3f (theory 0.333)\n", bets, share);
    CHECK(std::fabs(share - 1.0 / 3) < 0.05, "bluff share %.3f not near 1/3", share);
  }
}

static std::map<std::string, std::string> srp_spot(const std::string& board) {
  return {{"board", board},
          {"pot", "10"},
          {"stack", "95"},
          {"oop", "TT-22,AJs-A2s,KTs+,K9s-K6s,QTs+,Q9s,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo+,QTo+,JTo"},
          {"ip", "AA-22,A2s+,K7s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,A8o+,KTo+,QTo+,JTo"},
          {"bets", "33,75"},
          {"raises", "60"}};
}

// Fast terminal math (sorted sweep + card-removal sums) must equal a slow pairwise loop.
void test_terminals() {
  std::printf("[0] fast showdown/fold values == slow pairwise loop\n");
  for (const char* board : {"Ks 9d 5c 2h 2d", "Ah Kh Qh Jh 2c", "8s 8d 8c 7h 7d", "Ks 9d 5c 2h"}) {
    GameConfig cfg = config_from_fields({{"board", board},
                                         {"oop", "22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32"},
                                         {"ip", "22+,A2s+,K5s+,Q8s+,T9s,A8o+,KJo+"},
                                         {"bets", "50"},
                                         {"raises", "100"}});
    Game g(cfg);
    double err = g.self_check_terminals(120, 7);
    std::printf("  board %-16s max relative error %.2e\n", board, err);
    CHECK(err < 1e-4, "terminal values differ on %s: %g", board, err);
  }
}

void test_river_speed() {
  std::printf("[4] river solve, two bet sizes\n");
  GameConfig cfg = config_from_fields(srp_spot("Ks 9d 5c 2h 2d"));
  auto t0 = std::chrono::steady_clock::now();
  Game g(cfg);
  double expl = 100;
  int it = 0;
  while (it < 2000) {
    for (int k = 0; k < 25; ++k, ++it) g.run_iteration();
    expl = g.exploitability() / cfg.pot * 100;
    if (expl < 0.5) break;
  }
  double s = secs_since(t0);
  auto ev = g.expected_values();
  std::printf("  hands %d vs %d, %d iterations, exploitability %.3f%%, %.2f s, EV sum %.3f\n", g.num_hands(0),
              g.num_hands(1), it, expl, s, ev[0] + ev[1]);
  CHECK(expl < 0.5, "exploitability %.3f%%", expl);
  CHECK(s < 2.0, "took %.2f s", s);
  CHECK(std::fabs(ev[0] + ev[1] - cfg.pot) < 0.01 * cfg.pot, "EVs must add up to the pot");
}

void test_turn_speed() {
  std::printf("[5] turn solve, two bet sizes\n");
  auto f = srp_spot("Ks 9d 5c 2h");
  GameConfig cfg = config_from_fields(f);
  auto t0 = std::chrono::steady_clock::now();
  Game g(cfg);
  std::printf("  nodes %zu, memory %.0f MB\n", g.num_nodes(), g.memory_bytes() / 1e6);
  double expl = 100;
  int it = 0;
  while (it < 1000) {
    for (int k = 0; k < 20; ++k, ++it) g.run_iteration();
    expl = g.exploitability() / cfg.pot * 100;
    if (expl < 0.5) break;
  }
  double s = secs_since(t0);
  std::printf("  hands %d vs %d, %d iterations, exploitability %.3f%%, %.1f s\n", g.num_hands(0), g.num_hands(1), it,
              expl, s);
  CHECK(expl < 0.5, "exploitability %.3f%%", expl);
  CHECK(s < 60.0, "took %.1f s", s);
}

// Suit isomorphism must not change the game: the same iterations with and without it give
// the same exploitability, EVs and strategies, also at a card that has no subtree of its own.
void test_isomorphism() {
  std::printf("[6] suit isomorphism == full tree\n");
  auto f = srp_spot("Ks 9s 5s 2d");  // clubs and hearts are absent: they are symmetric
  f["bets"] = "50";
  f["raises"] = "none";
  f["max_raises"] = "0";
  f["allin"] = "0";
  GameConfig a = config_from_fields(f);
  f["iso"] = "0";
  GameConfig b = config_from_fields(f);
  Game ga(a), gb(b);
  std::printf("  dealt cards solved: %zu of %zu (iso) vs %zu of %zu (full)\n", ga.chance_cards_solved(),
              ga.chance_cards_total(), gb.chance_cards_solved(), gb.chance_cards_total());
  CHECK(ga.chance_cards_solved() < gb.chance_cards_solved(), "isomorphism should skip cards on this board");
  for (int i = 0; i < 40; ++i) ga.run_iteration(), gb.run_iteration();
  double ea = ga.exploitability(), eb = gb.exploitability();
  auto va = ga.expected_values(), vb = gb.expected_values();
  std::printf("  exploitability %.6f vs %.6f, EV OOP %.6f vs %.6f\n", ea, eb, va[0], vb[0]);
  CHECK(std::fabs(ea - eb) < 1e-3 * a.pot, "exploitability differs");
  CHECK(std::fabs(va[0] - vb[0]) < 1e-3, "EV differs");
  // Both suits of the river: one is the representative, the other uses the permutation.
  for (const char* card : {"7c", "7h"}) {
    std::vector<std::string> path = {"0", "0", card};  // check, check, river card
    std::string ja = ga.node_json(path), jb = gb.node_json(path);
    auto freq = [](const std::string& js) { size_t p = js.find("\"freq\":"); return js.substr(p, js.find(']', p) - p); };
    auto hand = [](const std::string& js, const std::string& c) {
      size_t p = js.find("{\"c\":\"" + c + "\"");
      return p == std::string::npos ? std::string("missing") : js.substr(p, js.find('}', p) - p);
    };
    std::printf("  river %s: %s | %s\n", card, freq(ja).c_str(), freq(jb).c_str());
    CHECK(freq(ja) == freq(jb), "frequencies differ at river %s", card);
    CHECK(hand(ja, "AhKh") == hand(jb, "AhKh") && hand(ja, "AcKc") == hand(jb, "AcKc"),
          "hand strategies differ at river %s", card);
    CHECK(ja.find("\"board\":\"Ks9s5s2d" + std::string(card) + "\"") != std::string::npos, "board label at %s", card);
  }
}

// 16-bit storage must still converge.
void test_compression() {
  std::printf("[7] 16-bit storage\n");
  GameConfig cfg = config_from_fields(srp_spot("Ks 9d 5c 2h 2d"));
  cfg.compress = 1;
  Game g(cfg);
  double expl = 100;
  int it = 0;
  while (it < 2000 && expl >= 0.5) {
    for (int k = 0; k < 25; ++k, ++it) g.run_iteration();
    expl = g.exploitability() / cfg.pot * 100;
  }
  std::printf("  memory %.1f MB (16-bit), %d iterations, exploitability %.3f%%\n", g.memory_bytes() / 1e6, it, expl);
  CHECK(g.compressed(), "storage should be 16-bit");
  CHECK(expl < 0.5, "16-bit exploitability %.3f%%", expl);
}

int main(int argc, char** argv) {
  bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  test_ranges();
  test_terminals();
  test_toy_river();
  test_compression();
  if (!quick) {
    test_isomorphism();
    test_evaluator();
    test_river_speed();
    test_turn_speed();
  }
  std::printf(g_fail ? "\n%d CHECK(S) FAILED\n" : "\nALL TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
