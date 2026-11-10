#include "value_net.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <stdexcept>

#include "thread_pool.hpp"

namespace poker {

// ---------------------------------------------------------------- runouts

FlopRunouts::FlopRunouts(const std::vector<Card>& flop, const std::vector<std::pair<Card, Card>> hs[2]) {
  if (flop.size() != 3) throw std::invalid_argument("FlopRunouts: the board must have 3 cards");
  for (int p = 0; p < 2; ++p) hands[p] = hs[p];
  for (int p = 0; p < 2; ++p) {
    std::map<std::pair<int, int>, int> pos;
    for (size_t j = 0; j < hands[1 - p].size(); ++j) {
      auto [a, b] = hands[1 - p][j];
      pos[{std::min(a, b), std::max(a, b)}] = static_cast<int>(j);
    }
    same[p].assign(hands[p].size(), -1);
    for (size_t h = 0; h < hands[p].size(); ++h) {
      auto [a, b] = hands[p][h];
      auto it = pos.find({std::min(a, b), std::max(a, b)});
      if (it != pos.end()) same[p][h] = it->second;
    }
  }
  uint64_t used = 0;
  for (Card c : flop) used |= 1ULL << c;
  std::vector<Card> deck;
  for (int c = 0; c < kNumCards; ++c)
    if (!(used >> c & 1)) deck.push_back(static_cast<Card>(c));
  for (size_t i = 0; i < deck.size(); ++i)
    for (size_t j = i + 1; j < deck.size(); ++j) {
      Runout ro;
      ro.a = deck[i], ro.b = deck[j];
      Card cards[7] = {flop[0], flop[1], flop[2], ro.a, ro.b, 0, 0};
      for (int p = 0; p < 2; ++p) {
        std::vector<std::pair<uint32_t, uint16_t>> v;
        for (size_t h = 0; h < hands[p].size(); ++h) {
          auto [c1, c2] = hands[p][h];
          if (c1 == ro.a || c1 == ro.b || c2 == ro.a || c2 == ro.b) continue;
          cards[5] = c1, cards[6] = c2;
          v.emplace_back(eval_hand(cards, 7), static_cast<uint16_t>(h));
        }
        std::sort(v.begin(), v.end());
        for (auto& e : v) ro.strength[p].push_back(e.first), ro.order[p].push_back(e.second);
      }
      runouts.push_back(std::move(ro));
    }
  // Dense strengths per runout (0 = the hand uses a runout card), then the pair matrix.
  const size_t n0 = hands[0].size(), n1 = hands[1].size(), K = runouts.size();
  std::vector<uint32_t> s0(K * n0, 0), s1(K * n1, 0);
  for (size_t k = 0; k < K; ++k) {
    for (size_t i = 0; i < runouts[k].order[0].size(); ++i) s0[k * n0 + runouts[k].order[0][i]] = runouts[k].strength[0][i] + 1;
    for (size_t i = 0; i < runouts[k].order[1].size(); ++i) s1[k * n1 + runouts[k].order[1][i]] = runouts[k].strength[1][i] + 1;
  }
  eq.assign(n0 * n1, 0.f);
  const int T = 8;
  ThreadPool pool(T);
  pool.parallel_for(static_cast<int>(n0), [&](int h) {
    std::vector<int32_t> acc(n1, 0);
    for (size_t k = 0; k < K; ++k) {
      const uint32_t a = s0[k * n0 + h];
      if (!a) continue;
      const uint32_t* b = &s1[k * n1];
      for (size_t o = 0; o < n1; ++o) acc[o] += b[o] ? (a > b[o]) - (a < b[o]) : 0;
    }
    const auto [h1, h2] = hands[0][h];
    for (size_t o = 0; o < n1; ++o) {
      const auto [o1, o2] = hands[1][o];
      const bool clash = h1 == o1 || h1 == o2 || h2 == o1 || h2 == o2;
      eq[h * n1 + o] = clash ? 0.f : static_cast<float>(acc[o]) / 990.f;  // C(45, 2) runouts per pair
    }
  });
}

// ---------------------------------------------------------------- features

namespace {

struct Sweep {  // per player: sums over all runouts
  std::vector<double> s1, s2, m, st, mt;  // st, mt: per turn card (52 x n)
  std::vector<double> hist, absd;         // hist: n x kRunoutBins
};

void sweep_player(const FlopRunouts& R, int p, const float* opp, Sweep& S) {
  const int o = 1 - p;
  const size_t n = R.hands[p].size();
  S.s1.assign(n, 0), S.s2.assign(n, 0), S.m.assign(n, 0);
  S.st.assign(kNumCards * n, 0), S.mt.assign(kNumCards * n, 0);
  S.hist.assign(n * kRunoutBins, 0), S.absd.assign(n, 0);
  std::vector<double> win;
  for (const auto& ro : R.runouts) {
    const auto& hs = ro.strength[p];
    const auto& ho = ro.order[p];
    const auto& os = ro.strength[o];
    const auto& oo = ro.order[o];
    double tot = 0, cs[kNumCards] = {0};
    for (size_t j = 0; j < oo.size(); ++j) {
      const double x = opp[oo[j]];
      tot += x;
      cs[R.hands[o][oo[j]].first] += x;
      cs[R.hands[o][oo[j]].second] += x;
    }
    win.assign(ho.size(), 0);
    double sum = 0, card[kNumCards] = {0};
    size_t j = 0;
    for (size_t i = 0; i < ho.size(); ++i) {
      while (j < oo.size() && os[j] < hs[i]) {
        const double x = opp[oo[j]];
        sum += x;
        card[R.hands[o][oo[j]].first] += x;
        card[R.hands[o][oo[j]].second] += x;
        ++j;
      }
      const auto& h = R.hands[p][ho[i]];
      win[i] = sum - card[h.first] - card[h.second];
    }
    sum = 0;
    std::fill(card, card + kNumCards, 0.0);
    j = oo.size();
    for (size_t i = ho.size(); i-- > 0;) {
      while (j > 0 && os[j - 1] > hs[i]) {
        --j;
        const double x = opp[oo[j]];
        sum += x;
        card[R.hands[o][oo[j]].first] += x;
        card[R.hands[o][oo[j]].second] += x;
      }
      const size_t h = ho[i];
      const auto& hc = R.hands[p][h];
      const double lose = sum - card[hc.first] - card[hc.second];
      const double wl = win[i] - lose;
      double m = tot - cs[hc.first] - cs[hc.second];
      if (R.same[p][h] >= 0) m += opp[R.same[p][h]];
      if (m <= 0) continue;
      S.s1[h] += wl;
      S.m[h] += m;
      S.s2[h] += wl * wl / m;
      const double dB = wl / m;
      S.absd[h] += std::fabs(wl);
      S.hist[h * kRunoutBins + std::clamp(static_cast<int>((dB + 1) * 0.5 * kRunoutBins), 0, kRunoutBins - 1)] += m;
      S.st[ro.a * n + h] += wl, S.mt[ro.a * n + h] += m;
      S.st[ro.b * n + h] += wl, S.mt[ro.b * n + h] += m;
    }
  }
}

void histogram(const std::vector<float>& d, const float* r, float* out8) {
  double tot = 0, bins[8] = {0};
  for (size_t h = 0; h < d.size(); ++h) {
    const int k = std::clamp(static_cast<int>((d[h] + 1) * 0.5 * 8), 0, 7);
    bins[k] += r[h];
    tot += r[h];
  }
  for (int k = 0; k < 8; ++k) out8[k] = tot > 0 ? static_cast<float>(bins[k] / tot) : 0.f;
}

}  // namespace

void compute_slow_features(const FlopRunouts& R, const float* const r[2], SlowFeatures& out) {
  thread_local Sweep S;
  for (int p = 0; p < 2; ++p) {
    const size_t n = R.hands[p].size();
    sweep_player(R, p, r[1 - p], S);
    out.e2[p].assign(n, 0.f);
    out.sd[p].assign(n, 0.f);
    out.hist[p].assign(n * kRunoutBins, 0.f);
    out.absd[p].assign(n, 0.f);
    for (size_t h = 0; h < n; ++h) {
      if (S.m[h] <= 0) continue;
      for (int k = 0; k < kRunoutBins; ++k)
        out.hist[p][h * kRunoutBins + k] = static_cast<float>(S.hist[h * kRunoutBins + k] / S.m[h]);
      out.absd[p][h] = static_cast<float>(S.absd[h] / S.m[h]);
      out.e2[p][h] = static_cast<float>(S.s2[h] / S.m[h]);
      double sum = 0, sum2 = 0;
      int k = 0;
      for (int c = 0; c < kNumCards; ++c) {
        const double mt = S.mt[c * n + h];
        if (mt <= 0) continue;
        const double dt = S.st[c * n + h] / mt;
        sum += dt, sum2 += dt * dt, ++k;
      }
      const double mu = k ? sum / k : 0;
      out.sd[p][h] = static_cast<float>(k ? std::sqrt(std::max(0.0, sum2 / k - mu * mu)) : 0);
    }
  }
}

void compute_leaf_features(const FlopRunouts& R, const float* const r[2], double contrib, double stack_left,
                           LeafFeatures& F, const SlowFeatures* slow) {
  thread_local SlowFeatures own;
  if (!slow) {
    compute_slow_features(R, r, own);
    slow = &own;
  }
  const size_t n0 = R.hands[0].size(), n1 = R.hands[1].size();
  // Check-down value: d0 = eq . r1 / M0, d1 = -eq^T . r0 / M1.
  std::vector<double> w0(n0, 0), w1(n1, 0);
  for (size_t h = 0; h < n0; ++h) {
    const float* row = &R.eq[h * n1];
    double s = 0;
    for (size_t o = 0; o < n1; ++o) s += row[o] * r[1][o];
    w0[h] = s;
    const float rh = r[0][h];
    if (rh != 0)
      for (size_t o = 0; o < n1; ++o) w1[o] -= row[o] * rh;
  }
  for (int p = 0; p < 2; ++p) {
    const int o = 1 - p;
    const size_t n = R.hands[p].size();
    // Compatible opponent reach on the flop.
    double tot = 0, cs[kNumCards] = {0};
    for (size_t j = 0; j < R.hands[o].size(); ++j) {
      tot += r[o][j];
      cs[R.hands[o][j].first] += r[o][j];
      cs[R.hands[o][j].second] += r[o][j];
    }
    F.mass[p].resize(n);
    for (size_t h = 0; h < n; ++h) {
      double m = tot - cs[R.hands[p][h].first] - cs[R.hands[p][h].second];
      if (R.same[p][h] >= 0) m += r[o][R.same[p][h]];
      F.mass[p][h] = static_cast<float>(std::max(m, 0.0));
    }
    F.d[p].resize(n);
    const std::vector<double>& w = p == 0 ? w0 : w1;
    for (size_t h = 0; h < n; ++h) F.d[p][h] = F.mass[p][h] > 0 ? static_cast<float>(w[h] / F.mass[p][h]) : 0.f;
  }
  const double spr = contrib > 0 ? stack_left / (2 * contrib) : 0;
  for (int p = 0; p < 2; ++p) {
    const int o = 1 - p;
    const size_t n = R.hands[p].size();
    float hist_own[8], hist_opp[8];
    histogram(F.d[p], r[p], hist_own);
    histogram(F.d[o], r[o], hist_opp);
    // Reach-weighted percentile of each hand inside its own range, and the range mean.
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return F.d[p][a] < F.d[p][b]; });
    double tot = 0, mean = 0;
    for (size_t h = 0; h < n; ++h) tot += r[p][h], mean += r[p][h] * F.d[p][h];
    mean = tot > 0 ? mean / tot : 0;
    std::vector<float> pct(n);
    double cum = 0;
    for (size_t i = 0; i < n; ++i) {
      const size_t h = idx[i];
      pct[h] = tot > 0 ? static_cast<float>((cum + 0.5 * r[p][h]) / tot) : 0.5f;
      cum += r[p][h];
    }
    F.x[p].resize(n * kLeafFeatures);
    for (size_t h = 0; h < n; ++h) {
      float* x = &F.x[p][h * kLeafFeatures];
      const float d = F.d[p][h];
      x[0] = d;
      const float e2 = slow->e2[p][h];
      x[1] = e2;
      x[2] = slow->sd[p][h];
      x[3] = e2 - d * d;
      x[4] = pct[h];
      x[5] = static_cast<float>(mean);
      x[6] = static_cast<float>(p);
      x[7] = static_cast<float>(std::log1p(spr));
      x[8] = static_cast<float>(1 / (1 + spr));
      for (int k = 0; k < 8; ++k) x[9 + k] = hist_own[k], x[17 + k] = hist_opp[k];
      for (int k = 0; k < kRunoutBins; ++k) x[25 + k] = slow->hist[p][h * kRunoutBins + k];
      x[31] = slow->absd[p][h];
    }
  }
}

// ---------------------------------------------------------------- network

void Mlp::init(const std::vector<int>& layer_sizes, unsigned seed) {
  sizes = layer_sizes;
  std::mt19937 rng(seed);
  W.clear(), b.clear();
  for (size_t l = 0; l + 1 < sizes.size(); ++l) {
    const int in = sizes[l], out = sizes[l + 1];
    std::normal_distribution<float> nd(0.f, std::sqrt(2.f / in));  // He init
    std::vector<float> w(static_cast<size_t>(in) * out);
    for (auto& x : w) x = nd(rng);
    if (l + 2 == sizes.size())
      for (auto& x : w) x *= 0.1f;  // small output layer: start near "check-down value"
    W.push_back(std::move(w));
    b.emplace_back(out, 0.f);
  }
  mean.assign(sizes[0], 0.f);
  inv_std.assign(sizes[0], 1.f);
}

size_t Mlp::num_params() const {
  size_t n = 0;
  for (size_t l = 0; l < W.size(); ++l) n += W[l].size() + b[l].size();
  return n;
}

float Mlp::forward(const float* x) const {
  thread_local std::vector<float> a, z;
  a.resize(sizes[0]);
  for (int i = 0; i < sizes[0]; ++i) a[i] = (x[i] - mean[i]) * inv_std[i];
  for (size_t l = 0; l < W.size(); ++l) {
    const int in = sizes[l], out = sizes[l + 1];
    z.assign(out, 0.f);
    for (int j = 0; j < out; ++j) {
      const float* w = &W[l][static_cast<size_t>(j) * in];
      float s = b[l][j];
      for (int i = 0; i < in; ++i) s += w[i] * a[i];
      z[j] = (l + 1 < W.size()) ? std::max(s, 0.f) : s;
    }
    a.swap(z);
  }
  return a[0];
}

void Mlp::save(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  std::fwrite("MLP1", 1, 4, f);
  const uint32_t nl = static_cast<uint32_t>(sizes.size());
  std::fwrite(&nl, 4, 1, f);
  for (int s : sizes) {
    const int32_t v = s;
    std::fwrite(&v, 4, 1, f);
  }
  std::fwrite(mean.data(), 4, mean.size(), f);
  std::fwrite(inv_std.data(), 4, inv_std.size(), f);
  for (size_t l = 0; l < W.size(); ++l) {
    std::fwrite(W[l].data(), 4, W[l].size(), f);
    std::fwrite(b[l].data(), 4, b[l].size(), f);
  }
  std::fclose(f);
}

void Mlp::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot read " + path);
  auto rd = [&](void* p, size_t n) {
    if (std::fread(p, 1, n, f) != n) {
      std::fclose(f);
      throw std::runtime_error(path + ": short read");
    }
  };
  char magic[4];
  rd(magic, 4);
  if (std::string(magic, 4) != "MLP1") throw std::runtime_error(path + ": not a network file");
  uint32_t nl;
  rd(&nl, 4);
  sizes.resize(nl);
  for (auto& s : sizes) {
    int32_t v;
    rd(&v, 4);
    s = v;
  }
  if (sizes.empty() || sizes[0] != kLeafFeatures) throw std::runtime_error(path + ": wrong input size");
  mean.resize(sizes[0]), inv_std.resize(sizes[0]);
  rd(mean.data(), 4 * mean.size());
  rd(inv_std.data(), 4 * inv_std.size());
  W.clear(), b.clear();
  for (size_t l = 0; l + 1 < sizes.size(); ++l) {
    W.emplace_back(static_cast<size_t>(sizes[l]) * sizes[l + 1]);
    b.emplace_back(sizes[l + 1]);
    rd(W.back().data(), 4 * W.back().size());
    rd(b.back().data(), 4 * b.back().size());
  }
  std::fclose(f);
}

namespace {

double mse(const Mlp& net, const std::vector<float>& X, const std::vector<float>& Y, ThreadPool& pool) {
  const size_t n = Y.size();
  if (!n) return 0;
  const int chunks = 64;
  std::vector<double> part(chunks, 0);
  pool.parallel_for(chunks, [&](int c) {
    const size_t lo = n * c / chunks, hi = n * (c + 1) / chunks;
    double s = 0;
    for (size_t i = lo; i < hi; ++i) {
      const double e = net.forward(&X[i * kLeafFeatures]) - Y[i];
      s += e * e;
    }
    part[c] = s;
  });
  return std::accumulate(part.begin(), part.end(), 0.0) / n;
}

}  // namespace

double train_mlp(Mlp& net, const std::vector<float>& X, const std::vector<float>& Y, const std::vector<float>& Xv,
                 const std::vector<float>& Yv, const TrainOptions& opt) {
  const size_t n = Y.size();
  const int F = kLeafFeatures;
  // Input normalization from the training rows.
  for (int i = 0; i < F; ++i) {
    double s = 0, s2 = 0;
    for (size_t r = 0; r < n; ++r) s += X[r * F + i], s2 += static_cast<double>(X[r * F + i]) * X[r * F + i];
    const double mu = s / n, var = std::max(s2 / n - mu * mu, 0.0);
    net.mean[i] = static_cast<float>(mu);
    net.inv_std[i] = static_cast<float>(var > 1e-10 ? 1 / std::sqrt(var) : 1.0);
  }
  const size_t L = net.W.size();
  std::vector<std::vector<float>> mW(L), vW(L), mB(L), vB(L);
  for (size_t l = 0; l < L; ++l) {
    mW[l].assign(net.W[l].size(), 0), vW[l].assign(net.W[l].size(), 0);
    mB[l].assign(net.b[l].size(), 0), vB[l].assign(net.b[l].size(), 0);
  }
  const int T = std::max(1, opt.threads);
  ThreadPool pool(T);
  std::vector<std::vector<std::vector<float>>> gW(T), gB(T);
  for (int t = 0; t < T; ++t) {
    gW[t].resize(L), gB[t].resize(L);
    for (size_t l = 0; l < L; ++l) gW[t][l].resize(net.W[l].size()), gB[t][l].resize(net.b[l].size());
  }
  std::vector<double> loss_part(T);
  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(opt.seed);
  const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
  long step = 0;
  double last = 0;
  for (int ep = 0; ep < opt.epochs; ++ep) {
    std::shuffle(perm.begin(), perm.end(), rng);
    const double lr = opt.lr * (ep >= opt.epochs * 2 / 3 ? 0.2 : 1.0);  // one decay step
    double ep_loss = 0;
    for (size_t start = 0; start < n; start += opt.batch) {
      const size_t end = std::min(n, start + opt.batch), bs = end - start;
      pool.parallel_for(T, [&](int t) {
        for (size_t l = 0; l < L; ++l) {
          std::fill(gW[t][l].begin(), gW[t][l].end(), 0.f);
          std::fill(gB[t][l].begin(), gB[t][l].end(), 0.f);
        }
        std::vector<std::vector<float>> act(L + 1);
        std::vector<float> delta, prev;
        double ls = 0;
        const size_t lo = start + bs * t / T, hi = start + bs * (t + 1) / T;
        for (size_t k = lo; k < hi; ++k) {
          const size_t r = perm[k];
          act[0].resize(F);
          for (int i = 0; i < F; ++i) act[0][i] = (X[r * F + i] - net.mean[i]) * net.inv_std[i];
          for (size_t l = 0; l < L; ++l) {
            const int in = net.sizes[l], out = net.sizes[l + 1];
            act[l + 1].resize(out);
            for (int j = 0; j < out; ++j) {
              const float* w = &net.W[l][static_cast<size_t>(j) * in];
              float s = net.b[l][j];
              for (int i = 0; i < in; ++i) s += w[i] * act[l][i];
              act[l + 1][j] = (l + 1 < L) ? std::max(s, 0.f) : s;
            }
          }
          const float e = act[L][0] - Y[r];
          ls += static_cast<double>(e) * e;
          delta.assign(1, 2 * e / static_cast<float>(bs));
          for (size_t l = L; l-- > 0;) {
            const int in = net.sizes[l], out = net.sizes[l + 1];
            prev.assign(in, 0.f);
            for (int j = 0; j < out; ++j) {
              const float dj = delta[j];
              if (dj == 0) continue;
              const float* w = &net.W[l][static_cast<size_t>(j) * in];
              float* g = &gW[t][l][static_cast<size_t>(j) * in];
              for (int i = 0; i < in; ++i) g[i] += dj * act[l][i], prev[i] += dj * w[i];
              gB[t][l][j] += dj;
            }
            if (l > 0)
              for (int i = 0; i < in; ++i)
                if (act[l][i] <= 0) prev[i] = 0;  // ReLU
            delta.swap(prev);
          }
        }
        loss_part[t] = ls;
      });
      for (int t = 0; t < T; ++t) ep_loss += loss_part[t];
      ++step;
      const double c1 = 1 - std::pow(b1, step), c2 = 1 - std::pow(b2, step);
      for (size_t l = 0; l < L; ++l) {
        auto upd = [&](std::vector<float>& p, std::vector<float>& m, std::vector<float>& v,
                       const std::vector<std::vector<std::vector<float>>>& g) {
          for (size_t i = 0; i < p.size(); ++i) {
            float gi = 0;
            for (int t = 0; t < T; ++t) gi += g[t][l][i];
            m[i] = static_cast<float>(b1 * m[i] + (1 - b1) * gi);
            v[i] = static_cast<float>(b2 * v[i] + (1 - b2) * gi * gi);
            p[i] -= static_cast<float>(lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps));
          }
        };
        upd(net.W[l], mW[l], vW[l], gW);
        upd(net.b[l], mB[l], vB[l], gB);
      }
    }
    last = ep_loss / n;
    if (!Yv.empty()) {
      std::printf("  epoch %2d  train mse %.5f  valid mse %.5f\n", ep + 1, last, mse(net, Xv, Yv, pool));
      std::fflush(stdout);
    }
  }
  return last;
}

// ---------------------------------------------------------------- evaluator

namespace {
FlopRunouts runouts_of(const Game& g) {
  std::vector<std::pair<Card, Card>> hs[2];
  for (int p = 0; p < 2; ++p)
    for (const auto& h : g.hands(p)) hs[p].push_back({h.c1, h.c2});
  return FlopRunouts(g.config().board, hs);
}
}  // namespace

NetLeafEvaluator::NetLeafEvaluator(const Game& g, const Mlp* net) : runouts_(runouts_of(g)), net_(net) {
  leaf_slot_.assign(g.num_nodes(), -1);
  for (size_t k = 0; k < g.leaf_nodes().size(); ++k) leaf_slot_[g.leaf_nodes()[k]] = static_cast<int>(k);
  cache_.resize(g.leaf_nodes().size());
}

void NetLeafEvaluator::evaluate(const Game&, const LeafInfo& leaf, const float* const r[2], float* const v[2]) const {
  thread_local LeafFeatures F;
  const SlowFeatures* slow = nullptr;
  if (net_) {
    // Draw features: refresh on the first calls, then every 16th call of this leaf.
    Cache& c = cache_[leaf_slot_[leaf.node]];
    if (c.calls < 4 || c.calls % 16 == 0) compute_slow_features(runouts_, r, c.slow);
    ++c.calls;
    slow = &c.slow;
  } else {
    static const SlowFeatures none = [] {
      SlowFeatures s;
      for (int p = 0; p < 2; ++p)
        s.e2[p].assign(1400, 0.f), s.sd[p].assign(1400, 0.f), s.hist[p].assign(1400 * kRunoutBins, 0.f),
            s.absd[p].assign(1400, 0.f);
      return s;
    }();
    slow = &none;  // check-down values do not use the slow features
  }
  compute_leaf_features(runouts_, r, leaf.contrib, leaf.stack_left, F, slow);
  const bool all_in = leaf.stack_left <= 1e-9;  // no more betting: the check-down value is exact
  double err = 0, z = 0;
  for (int p = 0; p < 2; ++p) {
    const size_t n = runouts_.hands[p].size();
    for (size_t h = 0; h < n; ++h) {
      double y = F.d[p][h];
      if (net_ && !all_in && F.mass[p][h] > 0) y += net_->forward(&F.x[p][h * kLeafFeatures]);
      v[p][h] = static_cast<float>(leaf.contrib * F.mass[p][h] * y);
      err += static_cast<double>(r[p][h]) * v[p][h];
      if (p == 0) z += static_cast<double>(r[p][h]) * F.mass[p][h];
    }
  }
  // Zero-sum fix (as in DeepStack): the two players' values must cancel. Move both down
  // by half the error, spread in proportion to M(h).
  if (z > 0 && net_ && !all_in) {
    const double shift = err / 2 / (leaf.contrib * z);
    for (int p = 0; p < 2; ++p)
      for (size_t h = 0; h < runouts_.hands[p].size(); ++h)
        v[p][h] -= static_cast<float>(leaf.contrib * F.mass[p][h] * shift);
  }
}

}  // namespace poker
