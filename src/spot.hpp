// Turn text fields (from the CLI or the web form) into a GameConfig.
#pragma once
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "game.hpp"

namespace poker {

inline std::vector<double> parse_sizes(const std::string& text, const std::string& field) {
  std::vector<double> out;
  std::string tok;
  std::istringstream is(text);
  while (std::getline(is, tok, ',')) {
    size_t a = tok.find_first_not_of(" %"), b = tok.find_last_not_of(" %");
    if (a == std::string::npos) continue;
    tok = tok.substr(a, b - a + 1);
    char* end = nullptr;
    double v = std::strtod(tok.c_str(), &end);
    if (*end != '\0' || v <= 0 || v > 1000)
      throw std::invalid_argument(field + ": '" + tok + "' is not a size in % of pot (example: 33, 75)");
    out.push_back(v / 100.0);
  }
  return out;
}

inline double parse_number(const std::map<std::string, std::string>& f, const std::string& key, double def) {
  auto it = f.find(key);
  if (it == f.end() || it->second.empty()) return def;
  char* end = nullptr;
  double v = std::strtod(it->second.c_str(), &end);
  if (*end != '\0') throw std::invalid_argument(key + ": '" + it->second + "' is not a number");
  return v;
}

inline std::string field(const std::map<std::string, std::string>& f, const std::string& key, const std::string& def) {
  auto it = f.find(key);
  return it == f.end() || it->second.empty() ? def : it->second;
}

// Keys: board, pot, stack, oop, ip, bets, raises, (flop_bets, turn_bets, river_bets,
// flop_raises, ...), max_raises, allin (0/1), threads, mem_gb.
inline GameConfig config_from_fields(const std::map<std::string, std::string>& f) {
  GameConfig cfg;
  cfg.board = parse_cards(field(f, "board", ""));
  cfg.pot = parse_number(f, "pot", 10);
  cfg.stack = parse_number(f, "stack", 95);
  try {
    cfg.range[0] = parse_range(field(f, "oop", ""));
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string("OOP range: ") + e.what());
  }
  try {
    cfg.range[1] = parse_range(field(f, "ip", ""));
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string("IP range: ") + e.what());
  }
  // Most specific key wins: oop_river_bets > river_bets > oop_bets > bets. "none" = no sizes.
  const char* streets[3] = {"flop", "turn", "river"};
  const char* players[2] = {"oop", "ip"};
  for (int p = 0; p < 2; ++p)
    for (int s = 0; s < 3; ++s)
      for (const char* kind : {"bets", "raises"}) {
        const std::string k = kind, pl = players[p], st = streets[s];
        std::string v = field(f, k, k == "bets" ? "33,75" : "60");
        v = field(f, pl + "_" + k, v);
        v = field(f, st + "_" + k, v);
        v = field(f, pl + "_" + st + "_" + k, v);
        auto sizes = v == "none" ? std::vector<double>{} : parse_sizes(v, k == "bets" ? "bet sizes" : "raise sizes");
        (k == "bets" ? cfg.sizes[p][s].bet : cfg.sizes[p][s].raise) = sizes;
      }
  cfg.max_raises = static_cast<int>(parse_number(f, "max_raises", 3));
  cfg.add_allin = parse_number(f, "allin", 1) != 0;
  cfg.threads = static_cast<int>(parse_number(f, "threads", 0));
  cfg.mem_limit_gb = parse_number(f, "mem_gb", 3.0);
  return cfg;
}

}  // namespace poker
