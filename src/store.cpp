#include "store.hpp"

#include <sqlite3.h>
#include <zlib.h>

#include <cstdio>
#include <filesystem>
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

// Node blob: "Z1", 4-byte little-endian JSON length, zlib data. Plain JSON text is format 1.
std::string pack_node(const std::string& json) {
  uLongf n = compressBound(static_cast<uLong>(json.size()));
  std::string out(6 + n, '\0');
  out[0] = 'Z', out[1] = '1';
  const uint32_t len = static_cast<uint32_t>(json.size());
  for (int i = 0; i < 4; ++i) out[2 + i] = static_cast<char>(len >> (8 * i) & 0xff);
  if (compress2(reinterpret_cast<Bytef*>(&out[6]), &n, reinterpret_cast<const Bytef*>(json.data()),
                static_cast<uLong>(json.size()), 6) != Z_OK)
    throw std::runtime_error("zlib compress failed");
  out.resize(6 + n);
  return out;
}

std::string unpack_node(const void* data, int size) {
  const char* p = static_cast<const char*>(data);
  if (size < 6 || p[0] != 'Z' || p[1] != '1') return std::string(p, p + size);  // format 1: plain JSON
  uint32_t len = 0;
  for (int i = 0; i < 4; ++i) len |= static_cast<uint32_t>(static_cast<unsigned char>(p[2 + i])) << (8 * i);
  std::string out(len, '\0');
  uLongf n = len;
  if (uncompress(reinterpret_cast<Bytef*>(&out[0]), &n, reinterpret_cast<const Bytef*>(p + 6),
                 static_cast<uLong>(size - 6)) != Z_OK || n != len)
    throw std::runtime_error("zlib uncompress failed");
  return out;
}

// Format 1 rows carry the grid; the trainer does not need it.
std::string drop_grid(const std::string& js) {
  const size_t a = js.find(",\"grid\":["), b = js.find("],\"hands\":");
  if (a == std::string::npos || b == std::string::npos || b < a) return js;
  return js.substr(0, a) + ",\"grid\":[" + js.substr(b);
}

const char* kSchema = R"SQL(
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS solves(
  id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP,
  preset TEXT, oop_pos TEXT, ip_pos TEXT, board TEXT, board_key TEXT, street TEXT, pot REAL, stack REAL,
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

Store::Store(const std::string& path) : path_(path) {
  if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
    std::string e = db_ ? sqlite3_errmsg(db_) : "out of memory";
    sqlite3_close(db_);
    throw std::runtime_error("cannot open database " + path + ": " + e);
  }
  sqlite3_busy_timeout(db_, 5000);  // the trainer server may read while a batch writes
  exec("PRAGMA journal_mode=WAL;");
  exec(kSchema);
  // Databases from before board_key existed: add the column (pokerbatch fills it in).
  bool has_key = false;
  {
    Stmt cols(db_, "PRAGMA table_info(solves)");
    while (sqlite3_step(cols.s) == SQLITE_ROW) has_key |= cols.col_text(1) == "board_key";
  }
  if (!has_key) exec("ALTER TABLE solves ADD COLUMN board_key TEXT");
  exec("CREATE INDEX IF NOT EXISTS solves_by_key ON solves(preset, street, board_key)");
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
          " solver_version, iterations, seconds, exploitability_pct, ev_oop, ev_ip, memory_mb, status, board_key)"
          " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
  st.text(1, r.preset), st.text(2, r.oop_pos), st.text(3, r.ip_pos), st.text(4, r.board), st.text(5, r.street);
  st.real(6, r.pot), st.real(7, r.stack), st.text(8, r.oop_range), st.text(9, r.ip_range), st.text(10, r.config);
  st.text(11, r.solver_version), st.integer(12, r.iterations), st.real(13, r.seconds);
  st.real(14, r.exploitability_pct), st.real(15, r.ev_oop), st.real(16, r.ev_ip), st.real(17, r.memory_mb);
  st.text(18, r.status);
  st.text(19, r.board_key);
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
  st.text(14, r.story_json);
  const std::string blob = pack_node(r.node_json);
  sqlite3_bind_blob(st.s, 15, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
  if (sqlite3_step(st.s) != SQLITE_DONE) throw std::runtime_error(std::string("sqlite insert: ") + sqlite3_errmsg(db_));
  return sqlite3_last_insert_rowid(db_);
}

double Store::file_gb() const {
  std::error_code ec;
  double total = 0;
  for (const char* ext : {"", "-wal"}) {
    auto sz = std::filesystem::file_size(path_ + ext, ec);
    if (!ec) total += static_cast<double>(sz);
  }
  return total / 1e9;
}

long long Store::compact() {
  long long converted = 0;
  std::vector<std::pair<long long, std::string>> rows;
  {
    Stmt st(db_, "SELECT id, node FROM spots WHERE substr(node, 1, 1) = '{'");
    while (sqlite3_step(st.s) == SQLITE_ROW) rows.emplace_back(sqlite3_column_int64(st.s, 0), st.col_text(1));
  }
  begin();
  for (auto& [id, js] : rows) {
    Stmt up(db_, "UPDATE spots SET node = ? WHERE id = ?");
    const std::string blob = pack_node(drop_grid(js));
    sqlite3_bind_blob(up.s, 1, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
    up.integer(2, id);
    sqlite3_step(up.s);
    ++converted;
  }
  exec("UPDATE meta SET value = '" + std::to_string(kStoreFormat) + "' WHERE key = 'format_version'");
  commit();
  exec("PRAGMA wal_checkpoint(TRUNCATE)");
  exec("VACUUM");
  return converted;
}

bool Store::has_solve(const std::string& preset, const std::string& street, const std::string& board_key) {
  Stmt st(db_, "SELECT 1 FROM solves WHERE preset=? AND street=? AND board_key=? AND status='ok' LIMIT 1");
  st.text(1, preset), st.text(2, street), st.text(3, board_key);
  return sqlite3_step(st.s) == SQLITE_ROW;
}

std::vector<std::pair<long long, std::string>> Store::solves_without_key() {
  std::vector<std::pair<long long, std::string>> out;
  Stmt st(db_, "SELECT id, board FROM solves WHERE board_key IS NULL OR board_key = ''");
  while (sqlite3_step(st.s) == SQLITE_ROW) out.emplace_back(sqlite3_column_int64(st.s, 0), st.col_text(1));
  return out;
}

void Store::set_board_key(long long id, const std::string& key) {
  Stmt st(db_, "UPDATE solves SET board_key=? WHERE id=?");
  st.text(1, key), st.integer(2, id);
  sqlite3_step(st.s);
}

std::pair<long long, long long> Store::remove_duplicates() {
  const std::string dup =
      "SELECT s.id FROM solves s WHERE EXISTS (SELECT 1 FROM solves t WHERE t.preset = s.preset AND"
      " t.street = s.street AND t.board_key = s.board_key AND t.status = 'ok' AND t.id < s.id)";
  begin();
  exec("DELETE FROM spots WHERE solve_id IN (" + dup + ")");
  const long long spots = sqlite3_changes(db_);
  exec("DELETE FROM solves WHERE id IN (" + dup + ")");
  const long long solves = sqlite3_changes(db_);
  commit();
  return {solves, spots};
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
     << "},\"exploitability_pct\":" << num(sqlite3_column_double(st.s, 15)) << ",\"node\":"
     << unpack_node(sqlite3_column_blob(st.s, 16), sqlite3_column_bytes(st.s, 16)) << "}";
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
