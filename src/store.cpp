#include "store.hpp"

#include <sqlite3.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace poker {

namespace {

std::string esc(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (static_cast<unsigned char>(c) < 0x20) continue;
    o += c;
  }
  return o;
}

std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.4g", v);
  return b;
}

// Small RAII wrapper for a prepared statement.
struct Stmt {
  sqlite3_stmt* s = nullptr;
  Stmt(sqlite3* db, const std::string& sql) {
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &s, nullptr) != SQLITE_OK)
      throw std::runtime_error(std::string("sqlite prepare: ") + sqlite3_errmsg(db));
  }
  ~Stmt() { sqlite3_finalize(s); }
  void text(int i, const std::string& v) { sqlite3_bind_text(s, i, v.c_str(), -1, SQLITE_TRANSIENT); }
  void real(int i, double v) { sqlite3_bind_double(s, i, v); }
  void integer(int i, long long v) { sqlite3_bind_int64(s, i, v); }
  std::string col_text(int i) {
    const unsigned char* t = sqlite3_column_text(s, i);
    return t ? reinterpret_cast<const char*>(t) : "";
  }
};

const char* kSchema = R"SQL(
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS solves(
  id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP,
  preset TEXT, oop_pos TEXT, ip_pos TEXT, board TEXT, street TEXT, pot REAL, stack REAL,
  oop_range TEXT, ip_range TEXT, config TEXT, solver_version TEXT, iterations INTEGER,
  seconds REAL, exploitability_pct REAL, ev_oop REAL, ev_ip REAL, memory_mb REAL, status TEXT);
CREATE TABLE IF NOT EXISTS spots(
  id INTEGER PRIMARY KEY, solve_id INTEGER REFERENCES solves(id),
  path TEXT, line TEXT, actor INTEGER, actor_pos TEXT, num_actions INTEGER, combos REAL,
  mixed REAL, ev_loss REAL, bluff REAL, fundamental REAL, category TEXT, score REAL,
  story TEXT, node TEXT);
CREATE INDEX IF NOT EXISTS spots_by_category ON spots(category, score);
)SQL";

}  // namespace

Store::Store(const std::string& path) {
  if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
    std::string e = db_ ? sqlite3_errmsg(db_) : "out of memory";
    sqlite3_close(db_);
    throw std::runtime_error("cannot open database " + path + ": " + e);
  }
  sqlite3_busy_timeout(db_, 5000);  // the trainer server may read while a batch writes
  exec("PRAGMA journal_mode=WAL;");
  exec(kSchema);
  Stmt st(db_, "INSERT OR IGNORE INTO meta(key, value) VALUES('format_version', ?)");
  st.text(1, std::to_string(kStoreFormat));
  sqlite3_step(st.s);
}

Store::~Store() { sqlite3_close(db_); }

void Store::exec(const std::string& sql) {
  char* err = nullptr;
  if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
    std::string e = err ? err : "?";
    sqlite3_free(err);
    throw std::runtime_error("sqlite: " + e);
  }
}

void Store::begin() { exec("BEGIN"); }
void Store::commit() { exec("COMMIT"); }

long long Store::add_solve(const SolveRow& r) {
  Stmt st(db_,
          "INSERT INTO solves(preset, oop_pos, ip_pos, board, street, pot, stack, oop_range, ip_range, config,"
          " solver_version, iterations, seconds, exploitability_pct, ev_oop, ev_ip, memory_mb, status)"
          " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
  st.text(1, r.preset), st.text(2, r.oop_pos), st.text(3, r.ip_pos), st.text(4, r.board), st.text(5, r.street);
  st.real(6, r.pot), st.real(7, r.stack), st.text(8, r.oop_range), st.text(9, r.ip_range), st.text(10, r.config);
  st.text(11, r.solver_version), st.integer(12, r.iterations), st.real(13, r.seconds);
  st.real(14, r.exploitability_pct), st.real(15, r.ev_oop), st.real(16, r.ev_ip), st.real(17, r.memory_mb);
  st.text(18, r.status);
  if (sqlite3_step(st.s) != SQLITE_DONE) throw std::runtime_error(std::string("sqlite insert: ") + sqlite3_errmsg(db_));
  return sqlite3_last_insert_rowid(db_);
}

long long Store::add_spot(const SpotRow& r) {
  Stmt st(db_,
          "INSERT INTO spots(solve_id, path, line, actor, actor_pos, num_actions, combos, mixed, ev_loss, bluff,"
          " fundamental, category, score, story, node) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
  st.integer(1, r.solve_id), st.text(2, r.path), st.text(3, r.line), st.integer(4, r.actor), st.text(5, r.actor_pos);
  st.integer(6, r.num_actions), st.real(7, r.combos), st.real(8, r.mixed), st.real(9, r.ev_loss);
  st.real(10, r.bluff), st.real(11, r.fundamental), st.text(12, r.category), st.real(13, r.score);
  st.text(14, r.story_json), st.text(15, r.node_json);
  if (sqlite3_step(st.s) != SQLITE_DONE) throw std::runtime_error(std::string("sqlite insert: ") + sqlite3_errmsg(db_));
  return sqlite3_last_insert_rowid(db_);
}

std::string Store::random_spot_json(const std::string& category, const std::vector<long long>& exclude) {
  std::string sql =
      "SELECT s.id, v.preset, v.street, v.board, v.pot, v.stack, v.oop_pos, v.ip_pos, s.story, s.path, s.category,"
      " s.mixed, s.ev_loss, s.bluff, s.fundamental, v.exploitability_pct, s.node, s.line"
      " FROM spots s JOIN solves v ON v.id = s.solve_id WHERE 1=1";
  if (!category.empty() && category != "any") sql += " AND s.category = ?1";
  if (!exclude.empty()) {
    sql += " AND s.id NOT IN (";
    for (size_t i = 0; i < exclude.size(); ++i) sql += (i ? "," : "") + std::to_string(exclude[i]);
    sql += ")";
  }
  sql += " ORDER BY RANDOM() LIMIT 1";
  Stmt st(db_, sql);
  if (!category.empty() && category != "any") st.text(1, category);
  if (sqlite3_step(st.s) != SQLITE_ROW) return "";
  std::ostringstream js;
  js << "{\"id\":" << sqlite3_column_int64(st.s, 0) << ",\"preset\":\"" << esc(st.col_text(1)) << "\",\"street\":\""
     << esc(st.col_text(2)) << "\",\"board\":\"" << esc(st.col_text(3)) << "\",\"pot\":" << num(sqlite3_column_double(st.s, 4))
     << ",\"stack\":" << num(sqlite3_column_double(st.s, 5)) << ",\"oop_pos\":\"" << esc(st.col_text(6))
     << "\",\"ip_pos\":\"" << esc(st.col_text(7)) << "\",\"story\":" << st.col_text(8) << ",\"path\":\""
     << esc(st.col_text(9)) << "\",\"line\":\"" << esc(st.col_text(17)) << "\",\"category\":\"" << esc(st.col_text(10))
     << "\",\"scores\":{\"mixed\":" << num(sqlite3_column_double(st.s, 11))
     << ",\"ev_loss\":" << num(sqlite3_column_double(st.s, 12)) << ",\"bluff\":" << num(sqlite3_column_double(st.s, 13))
     << ",\"fundamental\":" << num(sqlite3_column_double(st.s, 14))
     << "},\"exploitability_pct\":" << num(sqlite3_column_double(st.s, 15)) << ",\"node\":" << st.col_text(16) << "}";
  return js.str();
}

std::string Store::stats_json() {
  std::ostringstream js;
  Stmt a(db_, "SELECT COUNT(*) FROM spots");
  sqlite3_step(a.s);
  js << "{\"spots\":" << sqlite3_column_int64(a.s, 0);
  Stmt b(db_, "SELECT COUNT(*), SUM(status='ok'), AVG(CASE WHEN status='ok' THEN exploitability_pct END) FROM solves");
  sqlite3_step(b.s);
  js << ",\"solves\":" << sqlite3_column_int64(b.s, 0) << ",\"solves_ok\":" << sqlite3_column_int64(b.s, 1)
     << ",\"mean_exploitability_pct\":" << num(sqlite3_column_double(b.s, 2)) << ",\"by_category\":{";
  Stmt c(db_, "SELECT category, COUNT(*) FROM spots GROUP BY category ORDER BY category");
  bool first = true;
  while (sqlite3_step(c.s) == SQLITE_ROW) {
    js << (first ? "" : ",") << "\"" << esc(c.col_text(0)) << "\":" << sqlite3_column_int64(c.s, 1);
    first = false;
  }
  js << "}}";
  return js.str();
}

}  // namespace poker
