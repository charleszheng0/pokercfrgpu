// Training database (SQLite, one file). See docs/STORE.md for the schema and why.
//
// Stability rule: rows hold hands as combo text ("AhKd") and lines as text
// ("OOP Check > IP Bet 1.82"), never the solver's internal array layout. So changes to the
// solver (suit symmetry, 16-bit storage, CUDA) do not break stored data.
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

struct sqlite3;

namespace poker {

// 1: node stored as JSON text. 2: node stored as a zlib blob ("Z1" + 4-byte length + data),
// JSON without the 13x13 grid. Readers accept both.
constexpr int kStoreFormat = 2;

struct SolveRow {
  std::string preset, oop_pos, ip_pos, board, board_key, street, oop_range, ip_range, config, solver_version, status;
  double pot = 0, stack = 0, seconds = 0, exploitability_pct = 0, ev_oop = 0, ev_ip = 0, memory_mb = 0;
  int iterations = 0;
};

struct SpotRow {
  long long solve_id = 0;
  std::string path, line, actor_pos, category, story_json, node_json;
  int actor = 0, num_actions = 0;
  double combos = 0, mixed = 0, ev_loss = 0, bluff = 0, fundamental = 0, score = 0;
};

class Store {
 public:
  explicit Store(const std::string& path);  // opens or creates; throws std::runtime_error
  ~Store();
  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;

  long long add_solve(const SolveRow& r);
  long long add_spot(const SpotRow& r);
  // Duplicate control: board_key = canonical_board_key(board) (suit renaming, card order).
  bool has_solve(const std::string& preset, const std::string& street, const std::string& board_key);
  std::vector<std::pair<long long, std::string>> solves_without_key();  // (id, board)
  void set_board_key(long long id, const std::string& key);
  // Delete later copies of an (preset, street, board_key) solve and their spots.
  // Returns {solves removed, spots removed}.
  std::pair<long long, long long> remove_duplicates();
  // Convert format-1 rows to format 2, then VACUUM. Returns rows converted.
  long long compact();
  double file_gb() const;
  void begin();
  void commit();

  // One random spot as the JSON envelope of the play API, or "" when none matches.
  std::string random_spot_json(const std::string& category, const std::vector<long long>& exclude);
  std::string stats_json();
  // The stored node at one path of one solve (same JSON as random_spot_json), or "".
  std::string spot_json(long long solve_id, const std::string& path);
  // Cards stored right after `path` (e.g. the river cards kept for a turn line).
  std::vector<std::string> cards_after(long long solve_id, const std::string& path);

 private:
  void exec(const std::string& sql);
  std::string path_;
  sqlite3* db_ = nullptr;
};

}  // namespace poker
