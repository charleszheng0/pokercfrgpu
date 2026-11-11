// pokerlab: a small local HTTP server for the solver UI. No dependencies.
//   GET  /                -> web/index.html
//   POST /api/solve       -> form fields (see spot.hpp) + iters, target. Queues a solve,
//                            returns {"ok":true,"id":"..."}. Each solve has its own id.
//   GET  /api/status?id=  -> progress of that solve (state queued/building/running/done/error/stopped)
//   POST /api/stop?id=    -> stop that solve early (or take it out of the line)
//   GET  /api/node?id=&path= -> strategy at a node of that solve (path: "1,0,Kh,2")
//   GET  /api/limits      -> the solve limits of this server
// Solves run one at a time per worker (--workers, default 1); the others wait in line.
// --public adds limits for a shared server: turn and river only, a memory, iteration and
// time cap per solve, and a cap on the line length. Run: pokerlab [port] [db] [--public] ...
//   GET  /play            -> web/play.html (trainer)
//   GET  /api/play/spot?category=&exclude=  -> one random stored spot (see docs/STORE.md)
//   GET  /api/play/stats  -> counts in the training database
//   GET  /api/play/node?solve=&path=   -> the stored node at a path (to continue a hand)
//   GET  /api/play/cards?solve=&path=  -> cards stored after a path ({"cards": [...]})
//   GET  /api/play/showdown?board=&a=&b= -> who wins with hands a and b on the board
//   GET  /api/play/range?board=&hero=&combos= -> made hand of each combo (and vs the hero on the river)
// --gpu 1: in a CUDA build (pokerlab_gpu) every solve runs on the GPU solver (gpu/).
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <random>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "any_solver.hpp"
#include "game.hpp"
#include "spot.hpp"
#include "store.hpp"

using namespace poker;

namespace {

std::string g_db_path = std::string(DATA_DIR) + "/spots.db";

std::string read_file(const std::string& p, bool* ok) {
  std::ifstream in(p, std::ios::binary);
  *ok = static_cast<bool>(in);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string url_decode(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') out += ' ';
    else if (s[i] == '%' && i + 2 < s.size()) {
      out += static_cast<char>(std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16));
      i += 2;
    } else out += s[i];
  }
  return out;
}

std::map<std::string, std::string> parse_form(const std::string& body) {
  std::map<std::string, std::string> f;
  std::istringstream is(body);
  std::string kv;
  while (std::getline(is, kv, '&')) {
    size_t eq = kv.find('=');
    if (eq == std::string::npos) continue;
    f[url_decode(kv.substr(0, eq))] = url_decode(kv.substr(eq + 1));
  }
  return f;
}

std::string json_str(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\', out += c;
    else if (c == '\n') out += "\\n";
    else if (static_cast<unsigned char>(c) >= 0x20) out += c;
  }
  return out + "\"";
}

// ---- solves: every POST /api/solve makes its own Solve with a random id. Worker threads take
// solves from the line one at a time. Finished solves stay for browsing until they get old or
// their trees use more than keep_mb together (oldest first).
struct Limits {
  bool is_public = false;
  int workers = 1;            // solves that run at the same time
  int threads = 0;            // solver threads per solve (0 = the form value, or all cores)
  double mem_gb = 1e9;        // tree memory cap per solve (the form's mem_gb is capped to this)
  int max_iters = 1000000;
  double max_seconds = 0;     // 0 = no time cap
  size_t max_line = 50;       // solves waiting plus running
  int min_board = 3;          // 4 = turn and river only
  double keep_mb = 4000;      // finished trees kept for browsing, all solves together
  double keep_minutes = 60;
  bool gpu = false;           // --gpu 1 in a CUDA build: solves run on the GPU solver
};
Limits g_lim;

struct Solve {
  std::mutex mu;  // guards game access and the fields below
  std::string id;
  GameConfig cfg;
  std::unique_ptr<Game> game;
  std::string state = "queued";  // queued, building, running, done, error, stopped, expired
  std::string error, note;
  std::string backend = "cpu";  // "cuda" when the solve ran on the GPU
  int iteration = 0, max_iters = 0;
  double expl_pct = -1, target_pct = 0.5, seconds = 0, pot = 0, game_mb = 0;
  double ev[2] = {0, 0};
  std::string history;  // JSON array of [iteration, exploitability %]
  std::atomic<bool> stop{false};
  std::chrono::steady_clock::time_point finished{};
  bool is_finished() const { return state == "done" || state == "error" || state == "stopped" || state == "expired"; }
};

// Lock order: g_jobs_mu first, then a Solve::mu. Never the other way round.
std::mutex g_jobs_mu;
std::condition_variable g_cv;
std::map<std::string, std::shared_ptr<Solve>> g_jobs;
std::deque<std::shared_ptr<Solve>> g_line;
size_t g_running = 0;

std::string new_id() {  // caller holds g_jobs_mu
  static std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string id;
  for (int k = 0; k < 2; ++k) {
    uint64_t v = rng();
    for (int i = 0; i < 12; ++i) id += hex[(v >> (4 * i)) & 15];
  }
  return id;
}

// Caller holds g_jobs_mu. Forget old finished solves; free the oldest trees over keep_mb.
void prune_locked() {
  const auto now = std::chrono::steady_clock::now();
  struct Kept { std::chrono::steady_clock::time_point t; double mb; std::shared_ptr<Solve> s; };
  std::vector<Kept> kept;
  for (auto it = g_jobs.begin(); it != g_jobs.end();) {
    auto s = it->second;
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->is_finished() && std::chrono::duration<double>(now - s->finished).count() > g_lim.keep_minutes * 60) {
      it = g_jobs.erase(it);
      continue;
    }
    if (s->game) kept.push_back({s->finished, s->game_mb, s});
    ++it;
  }
  std::sort(kept.begin(), kept.end(), [](const Kept& a, const Kept& b) { return a.t > b.t; });  // newest first
  double total = 0;
  for (auto& k : kept) {
    total += k.mb;
    if (total <= g_lim.keep_mb) continue;
    std::lock_guard<std::mutex> lk(k.s->mu);
    k.s->game.reset();
    k.s->state = "expired";
  }
}

void run_solve(Solve& s) {
  auto t0 = std::chrono::steady_clock::now();
  auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  auto finish = [&](const std::string& state) {  // caller holds s.mu
    s.state = state;
    s.seconds = secs();
    s.finished = std::chrono::steady_clock::now();
  };
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.stop) return finish("stopped");
    s.state = "building";
  }
  std::unique_ptr<Game> game;
  try {
    game = std::make_unique<Game>(s.cfg);
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lk(s.mu);
    s.error = e.what();
    return finish("error");
  }
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.state = "running";
  }
  Game* gm = game.get();
  AnySolver solver(*gm, g_lim.gpu);
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.backend = solver.on_flat() ? solver.backend() : "cpu";
    if (g_lim.gpu && !solver.on_flat()) s.note = "GPU not used: " + solver.fallback_reason();
  }
  int next_check = 5, checked = 0;
  bool time_hit = false;
  std::string hist;
  auto record = [&](int it) {
    double e = solver.exploitability() / s.cfg.pot * 100;
    checked = it;
    hist += (hist.empty() ? "" : ",") + std::string("[") + std::to_string(it) + "," + std::to_string(e) + "]";
    std::lock_guard<std::mutex> lk(s.mu);
    s.iteration = it;
    s.expl_pct = e;
    s.seconds = secs();
    s.history = hist;
    return e;
  };
  for (int it = 1; it <= s.max_iters && !s.stop; ++it) {
    solver.run_iteration();
    if (g_lim.max_seconds > 0 && secs() > g_lim.max_seconds) {
      time_hit = true;
      break;
    }
    if (it == next_check || it == s.max_iters) {
      if (record(it) <= s.target_pct) break;
      next_check += it < 50 ? 5 : it < 200 ? 25 : 50;
    }
  }
  if (solver.iterations() > 0 && checked != solver.iterations()) record(solver.iterations());  // stopped between checks
  solver.sync();  // GPU result into the Game, so /api/node reads it as usual (no-op on the CPU solver)
  auto ev = gm->expected_values();
  std::lock_guard<std::mutex> lk(s.mu);
  s.iteration = gm->iterations();
  s.ev[0] = ev[0];
  s.ev[1] = ev[1];
  s.game_mb = gm->memory_bytes() / 1e6;
  if (time_hit || s.stop) s.note = time_hit ? "time limit" : "stopped";
  s.game = std::move(game);
  finish("done");
}

void worker_loop() {
  for (;;) {
    std::shared_ptr<Solve> s;
    {
      std::unique_lock<std::mutex> lk(g_jobs_mu);
      g_cv.wait(lk, [] { return !g_line.empty(); });
      s = g_line.front();
      g_line.pop_front();
      ++g_running;
    }
    run_solve(*s);
    std::lock_guard<std::mutex> lk(g_jobs_mu);
    --g_running;
    prune_locked();
  }
}

// The solve with this id (nullptr if unknown) and its place in the line (1 = next, 0 = not waiting).
std::shared_ptr<Solve> find_solve(const std::string& id, size_t* place = nullptr) {
  std::lock_guard<std::mutex> lk(g_jobs_mu);
  auto it = g_jobs.find(id);
  if (it == g_jobs.end()) return nullptr;
  if (place) {
    *place = 0;
    for (size_t i = 0; i < g_line.size(); ++i)
      if (g_line[i] == it->second) *place = i + 1;
  }
  return it->second;
}

std::string status_json(Solve& s, size_t place) {
  std::lock_guard<std::mutex> lk(s.mu);
  std::ostringstream js;
  js << "{\"id\":\"" << s.id << "\",\"state\":\"" << s.state << "\",\"place\":" << place
     << ",\"error\":" << json_str(s.error) << ",\"note\":" << json_str(s.note) << ",\"iteration\":" << s.iteration
     << ",\"max_iters\":" << s.max_iters << ",\"expl_pct\":" << s.expl_pct << ",\"target_pct\":" << s.target_pct
     << ",\"seconds\":" << s.seconds << ",\"pot\":" << s.pot << ",\"backend\":" << json_str(s.backend) << ",\"ev\":[" << s.ev[0] << "," << s.ev[1]
     << "],\"history\":[" << s.history << "]";
  if (s.game)
    js << ",\"nodes\":" << s.game->num_nodes() << ",\"memory_mb\":" << s.game->memory_bytes() / 1e6
       << ",\"hands\":[" << s.game->num_hands(0) << "," << s.game->num_hands(1) << "]";
  js << "}";
  return js.str();
}

std::string limits_json() {
  std::ostringstream js;
  js << "{\"public\":" << (g_lim.is_public ? "true" : "false") << ",\"min_board\":" << g_lim.min_board
     << ",\"max_iters\":" << g_lim.max_iters << ",\"max_seconds\":" << g_lim.max_seconds
     << ",\"mem_gb\":" << (g_lim.mem_gb > 1e6 ? -1 : g_lim.mem_gb) << ",\"max_line\":" << g_lim.max_line
     << ",\"workers\":" << g_lim.workers << ",\"gpu\":" << (g_lim.gpu ? "true" : "false") << "}";
  return js.str();
}

struct Response {
  int code = 200;
  std::string type = "application/json";
  std::string body;
};

Response handle(const std::string& method, const std::string& target, const std::string& body) {
  std::string path = target, query;
  size_t q = target.find('?');
  if (q != std::string::npos) path = target.substr(0, q), query = target.substr(q + 1);

  if (method == "GET" && (path == "/" || path == "/index.html" || path == "/play" || path == "/play.html")) {
    const std::string file = path.rfind("/play", 0) == 0 ? "play.html" : "index.html";
    bool ok = false;
    std::string html = read_file(std::string(WEB_DIR) + "/" + file, &ok);
    if (!ok) return {404, "text/plain", "web/" + file + " not found"};
    return {200, "text/html; charset=utf-8", html};
  }
  // Sample spots for the trainer when no database exists. Only plain file names are allowed.
  if (method == "GET" && path.rfind("/fixtures/", 0) == 0) {
    const std::string name = path.substr(10);
    if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-.") != std::string::npos ||
        name.find("..") != std::string::npos)
      return {404, "text/plain", "not found"};
    bool ok = false;
    std::string js = read_file(std::string(WEB_DIR) + "/fixtures/" + name, &ok);
    return ok ? Response{200, "application/json", js} : Response{404, "text/plain", "not found"};
  }
  if (method == "GET" && path == "/api/play/showdown") {
    auto f = parse_form(query);
    try {
      std::vector<Card> board = parse_cards(f["board"]), a = parse_cards(f["a"]), b = parse_cards(f["b"]);
      if (board.size() != 5 || a.size() != 2 || b.size() != 2) throw std::invalid_argument("need a 5-card board and two 2-card hands");
      std::vector<Card> all = board;
      all.insert(all.end(), a.begin(), a.end());
      all.insert(all.end(), b.begin(), b.end());
      parse_cards(cards_str(all));  // throws on a duplicate card
      Card ca[7], cb[7];
      for (int i = 0; i < 5; ++i) ca[i] = cb[i] = board[i];
      ca[5] = a[0], ca[6] = a[1], cb[5] = b[0], cb[6] = b[1];
      const uint32_t sa = eval_hand(ca, 7), sb = eval_hand(cb, 7);
      std::ostringstream js;
      js << "{\"winner\":\"" << (sa > sb ? "a" : sb > sa ? "b" : "tie") << "\",\"a\":\"" << category_name(hand_category(sa))
         << "\",\"b\":\"" << category_name(hand_category(sb)) << "\"}";
      return {200, "application/json", js.str()};
    } catch (const std::exception& e) {
      return {400, "application/json", "{\"error\":" + json_str(e.what()) + "}"};
    }
  }
  if (method == "GET" && path == "/api/play/range") {
    // The made hand of each villain combo on this board; on a 5-card board also who wins against
    // the hero's hand (vs: 1 villain wins, 0 tie, -1 hero wins). Order follows combos=.
    auto f = parse_form(query);
    try {
      const std::vector<Card> board = parse_cards(f["board"]), hero = parse_cards(f["hero"]);
      if (board.size() < 3 || board.size() > 5 || hero.size() != 2) throw std::invalid_argument("need a 3-5 card board and a 2-card hero hand");
      std::vector<Card> bh = board;
      bh.insert(bh.end(), hero.begin(), hero.end());
      parse_cards(cards_str(bh));  // throws on a duplicate card
      const bool river = board.size() == 5;
      Card hc[7];
      for (size_t i = 0; i < board.size(); ++i) hc[i] = board[i];
      hc[board.size()] = hero[0], hc[board.size() + 1] = hero[1];
      const uint32_t hs = eval_hand(hc, static_cast<int>(board.size()) + 2);
      std::ostringstream cat, vs;
      std::string list = f["combos"];
      size_t n = 0, pos = 0;
      while (pos <= list.size() && !list.empty()) {
        const size_t comma = std::min(list.find(',', pos), list.size());
        const std::vector<Card> v = parse_cards(list.substr(pos, comma - pos));
        if (v.size() != 2) throw std::invalid_argument("each combo needs 2 cards");
        if (++n > 1326) throw std::invalid_argument("too many combos");
        std::vector<Card> chk = bh;
        chk.insert(chk.end(), v.begin(), v.end());
        parse_cards(cards_str(chk));  // throws if the combo shares a card with the board or the hero
        Card vc[7];
        std::copy(board.begin(), board.end(), vc);
        vc[board.size()] = v[0], vc[board.size() + 1] = v[1];
        const uint32_t s = eval_hand(vc, static_cast<int>(board.size()) + 2);
        cat << (n > 1 ? "," : "") << "\"" << category_name(hand_category(s)) << "\"";
        vs << (n > 1 ? "," : "") << (s > hs ? 1 : s < hs ? -1 : 0);
        pos = comma + 1;
      }
      std::ostringstream js;
      js << "{\"hero\":\"" << category_name(hand_category(hs)) << "\",\"cat\":[" << cat.str() << "]";
      if (river) js << ",\"vs\":[" << vs.str() << "]";
      js << "}";
      return {200, "application/json", js.str()};
    } catch (const std::exception& e) {
      return {400, "application/json", "{\"error\":" + json_str(e.what()) + "}"};
    }
  }
  if (method == "GET" && (path == "/api/play/node" || path == "/api/play/cards")) {
    auto f = parse_form(query);
    try {
      Store store(g_db_path);
      const long long solve = std::stoll(f["solve"].empty() ? "0" : f["solve"]);
      if (path == "/api/play/cards") {
        std::string js = "{\"cards\":[";
        auto cards = store.cards_after(solve, f["path"]);
        for (size_t i = 0; i < cards.size(); ++i) js += (i ? ",\"" : "\"") + cards[i] + "\"";
        return {200, "application/json", js + "]}"};
      }
      std::string js = store.spot_json(solve, f["path"]);
      if (js.empty()) return {404, "application/json", "{\"error\":\"this line is not stored\"}"};
      return {200, "application/json", js};
    } catch (const std::exception& e) {
      return {400, "application/json", "{\"error\":" + json_str(e.what()) + "}"};
    }
  }
  if (method == "GET" && (path == "/api/play/spot" || path == "/api/play/stats")) {
    auto f = parse_form(query);
    try {
      Store store(g_db_path);  // one connection per request; SQLite handles the locking
      if (path == "/api/play/stats") return {200, "application/json", store.stats_json()};
      std::vector<long long> exclude;
      std::string tok;
      for (char c : f["exclude"] + ",") {
        if (c == ',') {
          if (!tok.empty() && tok.find_first_not_of("0123456789") == std::string::npos) exclude.push_back(std::stoll(tok));
          tok.clear();
        } else tok += c;
      }
      if (exclude.size() > 200) exclude.resize(200);
      std::string js = store.random_spot_json(f["category"], exclude);
      if (js.empty() && !exclude.empty()) js = store.random_spot_json(f["category"], {});  // all seen: allow repeats
      if (js.empty()) return {404, "application/json", "{\"error\":\"no stored spots for this category yet\"}"};
      return {200, "application/json", js};
    } catch (const std::exception& e) {
      return {500, "application/json", "{\"error\":" + json_str(e.what()) + "}"};
    }
  }
  if (method == "GET" && path == "/api/limits") return {200, "application/json", limits_json()};
  if (method == "POST" && path == "/api/solve") {
    auto f = parse_form(body);
    std::shared_ptr<Solve> s;
    size_t place = 0;
    try {
      GameConfig cfg = config_from_fields(f);
      if (static_cast<int>(cfg.board.size()) < g_lim.min_board)
        throw std::invalid_argument(
            "This server solves turn and river spots only (4 or 5 board cards). "
            "A flop tree needs more memory than this server has.");
      int iters = std::max(1, std::min(static_cast<int>(parse_number(f, "iters", 1000)), g_lim.max_iters));
      double target = parse_number(f, "target", 0.5);
      if (g_lim.threads > 0) cfg.threads = g_lim.threads;
      cfg.mem_limit_gb = std::min(cfg.mem_limit_gb, g_lim.mem_gb);
      prepare_gpu_config(cfg, g_lim.gpu);
      s = std::make_shared<Solve>();
      s->cfg = cfg;
      s->max_iters = iters;
      s->target_pct = target;
      s->pot = cfg.pot;
    } catch (const std::exception& e) {
      return {400, "application/json", "{\"ok\":false,\"error\":" + json_str(e.what()) + "}"};
    }
    {
      std::lock_guard<std::mutex> lk(g_jobs_mu);
      prune_locked();
      if (g_line.size() + g_running >= g_lim.max_line)
        return {503, "application/json",
                "{\"ok\":false,\"error\":\"The solver is busy. Please try again in a minute.\"}"};
      s->id = new_id();
      g_jobs[s->id] = s;
      g_line.push_back(s);
      place = g_line.size();
    }
    g_cv.notify_one();
    return {200, "application/json",
            "{\"ok\":true,\"id\":\"" + s->id + "\",\"place\":" + std::to_string(place) + "}"};
  }
  if (path == "/api/stop" || path == "/api/status" || path == "/api/node") {
    auto f = parse_form(method == "POST" && query.empty() ? body : query);
    auto s = find_solve(f["id"]);
    if (!s) return {404, "application/json", "{\"error\":\"unknown solve id (it may have expired). Solve again.\"}"};
    if (method == "POST" && path == "/api/stop") {
      s->stop = true;
      std::lock_guard<std::mutex> lk(g_jobs_mu);  // still waiting: take it out of the line
      auto it = std::find(g_line.begin(), g_line.end(), s);
      if (it != g_line.end()) {
        g_line.erase(it);
        std::lock_guard<std::mutex> lk2(s->mu);
        s->state = "stopped";
        s->finished = std::chrono::steady_clock::now();
      }
      return {200, "application/json", "{\"ok\":true}"};
    }
    if (method == "GET" && path == "/api/status") {
      size_t place = 0;
      s = find_solve(f["id"], &place);
      if (!s) return {404, "application/json", "{\"error\":\"unknown solve id\"}"};
      return {200, "application/json", status_json(*s, place)};
    }
    if (method == "GET" && path == "/api/node") {
      std::vector<std::string> p;
      std::string tok;
      for (char c : f["path"] + ",") {
        if (c == ',') {
          if (!tok.empty()) p.push_back(tok);
          tok.clear();
        } else tok += c;
      }
      std::lock_guard<std::mutex> lk(s->mu);
      if (!s->game)
        return {409, "application/json",
                s->state == "expired" ? "{\"error\":\"this result expired. Solve again.\"}" : "{\"error\":\"no finished solve\"}"};
      try {
        return {200, "application/json", s->game->node_json(p)};
      } catch (const std::exception& e) {
        return {400, "application/json", "{\"error\":" + json_str(e.what()) + "}"};
      }
    }
  }
  return {404, "text/plain", "not found"};
}

void serve_client(int fd) {
  std::string req;
  char buf[65536];
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t n = recv(fd, buf, sizeof buf, 0);
    if (n <= 0) return;
    req.append(buf, static_cast<size_t>(n));
    header_end = req.find("\r\n\r\n");
    if (req.size() > (1 << 20)) return;
  }
  std::istringstream hs(req.substr(0, header_end));
  std::string method, target, line;
  hs >> method >> target;
  size_t content_length = 0;
  while (std::getline(hs, line)) {
    for (auto& c : line) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (line.rfind("content-length:", 0) == 0) content_length = std::strtoul(line.c_str() + 15, nullptr, 10);
  }
  std::string body = req.substr(header_end + 4);
  while (body.size() < content_length) {
    ssize_t n = recv(fd, buf, sizeof buf, 0);
    if (n <= 0) return;
    body.append(buf, static_cast<size_t>(n));
  }
  Response r = handle(method, target, body);
  std::ostringstream out;
  out << "HTTP/1.1 " << r.code << (r.code == 200 ? " OK" : " Error") << "\r\nContent-Type: " << r.type
      << "\r\nContent-Length: " << r.body.size() << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
      << r.body;
  std::string s = out.str();
  size_t sent = 0;
  while (sent < s.size()) {
    ssize_t n = send(fd, s.data() + sent, s.size() - sent, 0);
    if (n <= 0) break;
    sent += static_cast<size_t>(n);
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Positional: [port] [db]. Flags: --public, then any of --workers N --threads N --mem-gb X
  // --max-iters N --max-seconds S --max-line N --keep-mb X --keep-minutes M (these win over --public).
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--public") {
      g_lim.is_public = true;
      g_lim.min_board = 4;
      g_lim.mem_gb = 0.5;
      g_lim.max_iters = 1000;
      g_lim.max_seconds = 60;
      g_lim.max_line = 10;
      g_lim.keep_mb = 250;
      g_lim.keep_minutes = 15;
    }
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--public") continue;
    if (a.rfind("--", 0) != 0) {
      pos.push_back(a);
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "pokerlab: %s needs a value\n", a.c_str());
      return 2;
    }
    const double v = std::atof(argv[++i]);
    if (a == "--workers") g_lim.workers = std::max(1, static_cast<int>(v));
    else if (a == "--threads") g_lim.threads = static_cast<int>(v);
    else if (a == "--mem-gb") g_lim.mem_gb = v;
    else if (a == "--max-iters") g_lim.max_iters = std::max(1, static_cast<int>(v));
    else if (a == "--max-seconds") g_lim.max_seconds = v;
    else if (a == "--max-line") g_lim.max_line = static_cast<size_t>(std::max(1.0, v));
    else if (a == "--keep-mb") g_lim.keep_mb = v;
    else if (a == "--keep-minutes") g_lim.keep_minutes = v;
    else if (a == "--gpu") g_lim.gpu = v != 0;
    else {
      std::fprintf(stderr, "pokerlab: unknown option %s\n", a.c_str());
      return 2;
    }
  }
  g_lim.gpu = use_gpu(g_lim.gpu, "pokerlab");
  if (g_lim.gpu) std::printf("solves run on the GPU (%s back end)\n", FlatSolver::backend());
  int port = pos.size() > 0 ? std::atoi(pos[0].c_str()) : 8080;
  if (pos.size() > 1) g_db_path = pos[1];
  for (int w = 0; w < g_lim.workers; ++w) std::thread(worker_loop).detach();
  std::signal(SIGPIPE, SIG_IGN);
  int s = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || listen(s, 64) < 0) {
    std::perror("bind/listen");
    return 1;
  }
  std::printf("pokerlab: open http://127.0.0.1:%d  (trainer: /play, database: %s)\n", port, g_db_path.c_str());
  std::printf("pokerlab: solve limits %s\n", limits_json().c_str());
  std::fflush(stdout);
  for (;;) {
    int fd = accept(s, nullptr, nullptr);
    if (fd < 0) continue;
    std::thread([fd] {
      serve_client(fd);
      close(fd);
    }).detach();
  }
}
