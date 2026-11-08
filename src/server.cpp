// pokerlab: a small local HTTP server for the solver UI. No dependencies.
//   GET  /                -> web/index.html
//   POST /api/solve       -> form fields (see spot.hpp) + iters, target. Starts a solve.
//   GET  /api/status      -> progress of the current solve
//   POST /api/stop        -> stop the current solve early
//   GET  /api/node?path=  -> strategy at a node (path: "1,0,Kh,2")
//   GET  /play            -> web/play.html (trainer)
//   GET  /api/play/spot?category=&exclude=  -> one random stored spot (see docs/STORE.md)
//   GET  /api/play/stats  -> counts in the training database
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

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

// ---- solve state shared between the HTTP thread and the solver thread
struct Solve {
  std::mutex mu;  // guards game access and the fields below
  std::unique_ptr<Game> game;
  std::string state = "idle";  // idle, building, running, done, error
  std::string error;
  int iteration = 0, max_iters = 0;
  double expl_pct = -1, target_pct = 0.5, seconds = 0, pot = 0;
  double ev[2] = {0, 0};
  std::string history;  // JSON array of [iteration, exploitability %]
  std::atomic<bool> stop{false};
  std::thread worker;
};
Solve g;

void run_solve(GameConfig cfg, int max_iters, double target_pct) {
  auto t0 = std::chrono::steady_clock::now();
  auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  std::unique_ptr<Game> game;
  try {
    game = std::make_unique<Game>(cfg);
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.state = "error";
    g.error = e.what();
    return;
  }
  {
    std::lock_guard<std::mutex> lk(g.mu);
    g.state = "running";
  }
  Game* gm = game.get();
  int next_check = 5;
  std::string hist;
  for (int it = 1; it <= max_iters && !g.stop; ++it) {
    gm->run_iteration();
    if (it == next_check || it == max_iters) {
      double e = gm->exploitability() / cfg.pot * 100;
      hist += (hist.empty() ? "" : ",") + std::string("[") + std::to_string(it) + "," + std::to_string(e) + "]";
      std::lock_guard<std::mutex> lk(g.mu);
      g.iteration = it;
      g.expl_pct = e;
      g.seconds = secs();
      g.history = hist;
      if (e <= target_pct) break;
      next_check += it < 50 ? 5 : it < 200 ? 25 : 50;
    }
  }
  auto ev = gm->expected_values();
  std::lock_guard<std::mutex> lk(g.mu);
  g.iteration = gm->iterations();
  g.seconds = secs();
  g.ev[0] = ev[0];
  g.ev[1] = ev[1];
  g.game = std::move(game);
  g.state = "done";
}

std::string status_json() {
  std::lock_guard<std::mutex> lk(g.mu);
  std::ostringstream js;
  js << "{\"state\":\"" << g.state << "\",\"error\":" << json_str(g.error) << ",\"iteration\":" << g.iteration
     << ",\"max_iters\":" << g.max_iters << ",\"expl_pct\":" << g.expl_pct << ",\"target_pct\":" << g.target_pct
     << ",\"seconds\":" << g.seconds << ",\"pot\":" << g.pot << ",\"ev\":[" << g.ev[0] << "," << g.ev[1]
     << "],\"history\":[" << g.history << "]";
  if (g.game)
    js << ",\"nodes\":" << g.game->num_nodes() << ",\"memory_mb\":" << g.game->memory_bytes() / 1e6
       << ",\"hands\":[" << g.game->num_hands(0) << "," << g.game->num_hands(1) << "]";
  js << "}";
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
  if (method == "POST" && path == "/api/solve") {
    auto f = parse_form(body);
    try {
      GameConfig cfg = config_from_fields(f);
      int iters = static_cast<int>(parse_number(f, "iters", 1000));
      double target = parse_number(f, "target", 0.5);
      g.stop = true;
      if (g.worker.joinable()) g.worker.join();
      std::lock_guard<std::mutex> lk(g.mu);
      g.stop = false;
      g.game.reset();
      g.state = "building";
      g.error.clear();
      g.iteration = 0;
      g.max_iters = iters;
      g.expl_pct = -1;
      g.target_pct = target;
      g.seconds = 0;
      g.pot = cfg.pot;
      g.history.clear();
      g.worker = std::thread(run_solve, cfg, iters, target);
      return {200, "application/json", "{\"ok\":true}"};
    } catch (const std::exception& e) {
      return {400, "application/json", "{\"ok\":false,\"error\":" + json_str(e.what()) + "}"};
    }
  }
  if (method == "POST" && path == "/api/stop") {
    g.stop = true;
    return {200, "application/json", "{\"ok\":true}"};
  }
  if (method == "GET" && path == "/api/status") return {200, "application/json", status_json()};
  if (method == "GET" && path == "/api/node") {
    auto f = parse_form(query);
    std::vector<std::string> p;
    std::string tok;
    for (char c : f["path"] + ",") {
      if (c == ',') {
        if (!tok.empty()) p.push_back(tok);
        tok.clear();
      } else tok += c;
    }
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.game) return {409, "application/json", "{\"error\":\"no finished solve\"}"};
    try {
      return {200, "application/json", g.game->node_json(p)};
    } catch (const std::exception& e) {
      return {400, "application/json", "{\"error\":" + json_str(e.what()) + "}"};
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
  int port = argc > 1 ? std::atoi(argv[1]) : 8080;
  if (argc > 2) g_db_path = argv[2];
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
