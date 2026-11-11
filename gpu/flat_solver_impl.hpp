// Implementation of FlatSolver. Included by flat_solver_cpu.cpp (CPU threads) and
// flat_solver.cu (CUDA). All loops are for_each(n, LAMBDA(i)) with raw pointers, so the same
// code runs on both back ends. See flat_solver.hpp for the method.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "backend.hpp"
#include "flat_solver.hpp"

namespace poker {

namespace flat {
#ifdef __CUDACC__
typedef float acc_t;   // T4-class GPUs run FP64 at 1/32 of FP32
#else
typedef double acc_t;  // CPU: same precision as Game
#endif
constexpr int kMaxActions = 8;
enum Mode { kTrain = 0, kAverage = 1, kBest = 2 };
enum Cat { kAct0 = 0, kAct1 = 1, kChance = 2, kFold = 3, kShow = 4, kNumCat = 5 };

HD inline void current_strategy(const float* r, int na, int nh, int h, float* sig) {
  float sum = 0;
  for (int a = 0; a < na; ++a) sum += r[a * nh + h] > 0 ? r[a * nh + h] : 0.f;
  for (int a = 0; a < na; ++a) sig[a] = sum > 0 ? (r[a * nh + h] > 0 ? r[a * nh + h] : 0.f) / sum : 1.f / na;
}
HD inline void average_strategy(const float* s, int na, int nh, int h, float* sig) {
  float sum = 0;
  for (int a = 0; a < na; ++a) sum += s[a * nh + h];
  for (int a = 0; a < na; ++a) sig[a] = sum > 0 ? s[a * nh + h] / sum : 1.f / na;
}
}  // namespace flat

// All members public: CUDA device lambdas may not live in private member functions.
struct FlatSolverImpl {
  int H[2] = {0, 0}, Hmax = 0;
  size_t nodes = 0;
  int levels = 0;
  int iteration = 0;
  double pot = 0, pair_weight_sum = 0;
  std::vector<float> host_weight[2];

  // tree
  Buf<uint8_t> type, player, board_size, dealt, live;
  Buf<uint32_t> num_children, first_child, first_entry, board_id;
  Buf<uint64_t> data;
  Buf<float> contrib0, contrib1;
  Buf<uint8_t> c1[2], c2[2];
  Buf<uint32_t> ch_off[2];  // per card c: hands of player p that hold c are ch_idx[ch_off[c] .. ch_off[c+1])
  Buf<uint16_t> ch_idx[2];
  Buf<int32_t> same[2];
  Buf<uint8_t> e_card, e_perm;
  Buf<uint16_t> e_child;
  Buf<int32_t> perm_inv;
  Buf<uint16_t> hand_perm[2];
  // showdowns, per (showdown s, traverser p): own hand -> [lo, hi) range of equal strength in
  // the opponent's sorted list; opponent sorted order; per card, positions of opponent hands
  // that hold it (for card removal).
  Buf<int32_t> sd_index;                  // per board id
  Buf<int32_t> sd_lo, sd_hi;              // flat, offset sd_own_off[s*2+p]
  Buf<uint64_t> sd_own_off, sd_ord_off;   // per s*2+p
  Buf<int32_t> sd_m;                      // opponent hands in the list, per s*2+p
  Buf<uint16_t> sd_ord;                   // opponent hand index by position
  Buf<uint64_t> sd_card_off;              // per (s*2+p)*53 + c
  Buf<int32_t> sd_card_pos;
  // solver state
  Buf<float> regret, ssum;
  Buf<float> R, V;          // nodes x Hmax: opponent reach (down pass), traverser value (up pass)
  Buf<flat::acc_t> prefix;  // showdown prefix sums of the current level
  Buf<flat::acc_t> foldsum;      // fold: 52 card sums + total, per fold node of the current level
  // level lists
  Buf<uint32_t> list;
  std::vector<int64_t> lbeg, lcnt;  // per level * kNumCat
  int64_t max_show = 0, max_fold = 0;
  static constexpr int kChunk = 32;  // showdown prefix sums: chunk size
  int nch = 1;                       // chunks per hand list

  // Profiling: with prof on, every kernel is followed by a sync and its time is added up.
  bool prof = false;
  double prof_secs[10] = {0};
  long prof_calls[10] = {0};
  template <class F>
  void kernel(int id, int64_t n, F f) {
    if (!prof) {
      for_each(n, f);
      return;
    }
    device_sync();
    const auto t0 = std::chrono::steady_clock::now();
    for_each(n, f);
    device_sync();
    prof_secs[id] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++prof_calls[id];
  }
  std::string profile_report() const {
    static const char* names[10] = {"down_opp", "down_own", "down_chance", "fold_prep", "fold", "show_prep", "show", "up_own", "up_opp", "up_chance"};
    std::string out;
    char buf[160];
    for (int i = 0; i < 10; ++i) {
      std::snprintf(buf, sizeof buf, "  %-12s %9.3f s  %8ld launches\n", names[i], prof_secs[i], prof_calls[i]);
      out += buf;
    }
    return out;
  }

  int64_t beg(int l, int c) const { return lbeg[l * flat::kNumCat + c]; }
  int64_t cnt(int l, int c) const { return lcnt[l * flat::kNumCat + c]; }

  explicit FlatSolverImpl(const FlatTree& t) {
    using namespace flat;
    nodes = t.type.size();
    for (int p = 0; p < 2; ++p) H[p] = t.n_hands[p], host_weight[p] = t.weight[p];
    Hmax = std::max(H[0], H[1]);
    pot = t.pot, pair_weight_sum = t.pair_weight_sum;
    // Levels (breadth first) and the node lists of each level by kind.
    std::vector<int> lev(nodes, -1);
    lev[0] = 0;
    std::vector<uint32_t> order = {0};
    for (size_t i = 0; i < order.size(); ++i) {
      const uint32_t x = order[i];
      if (t.type[x] == 0 || t.type[x] == 3)
        for (uint32_t k = 0; k < t.num_children[x]; ++k) {
          const uint32_t c = t.first_child[x] + k;
          lev[c] = lev[x] + 1;
          order.push_back(c);
        }
    }
    for (size_t x = 0; x < nodes; ++x)
      if (lev[x] < 0) throw std::runtime_error("flat solver: node not reachable");
    levels = 1 + *std::max_element(lev.begin(), lev.end());
    std::vector<std::vector<uint32_t>> bucket(static_cast<size_t>(levels) * kNumCat);
    for (size_t x = 0; x < nodes; ++x) {
      int c = t.type[x] == 0 ? (t.player[x] == 0 ? kAct0 : kAct1) : t.type[x] == 1 ? kFold : t.type[x] == 2 ? kShow : kChance;
      if (t.type[x] == 0 && t.num_children[x] > static_cast<uint32_t>(kMaxActions))
        throw std::runtime_error("flat solver: too many actions at a node");
      bucket[lev[x] * kNumCat + c].push_back(static_cast<uint32_t>(x));
    }
    std::vector<uint32_t> all;
    for (const auto& b : bucket) {
      lbeg.push_back(static_cast<int64_t>(all.size()));
      lcnt.push_back(static_cast<int64_t>(b.size()));
      all.insert(all.end(), b.begin(), b.end());
    }
    for (int l = 0; l < levels; ++l) max_show = std::max(max_show, cnt(l, kShow)), max_fold = std::max(max_fold, cnt(l, kFold));
    list.upload(all);

    type.upload(t.type), player.upload(t.player), board_size.upload(t.board_size), dealt.upload(t.dealt);
    num_children.upload(t.num_children), first_child.upload(t.first_child), first_entry.upload(t.first_entry);
    board_id.upload(t.board_id), data.upload(t.data), contrib0.upload(t.contrib0), contrib1.upload(t.contrib1);
    for (int p = 0; p < 2; ++p) {
      c1[p].upload(t.c1[p]), c2[p].upload(t.c2[p]), same[p].upload(t.same[p]), hand_perm[p].upload(t.hand_perm[p]);
      std::vector<uint32_t> off(53, 0);
      std::vector<uint16_t> idx;
      for (int c = 0; c < 52; ++c) {
        off[c] = static_cast<uint32_t>(idx.size());
        for (int h = 0; h < H[p]; ++h)
          if (t.c1[p][h] == c || t.c2[p][h] == c) idx.push_back(static_cast<uint16_t>(h));
      }
      off[52] = static_cast<uint32_t>(idx.size());
      ch_off[p].upload(off), ch_idx[p].upload(idx);
    }
    e_card.upload(t.e_card), e_perm.upload(t.e_perm), e_child.upload(t.e_child);
    perm_inv.upload(std::vector<int32_t>(t.perm_inv, t.perm_inv + 24));
    live.alloc(nodes);

    // Showdown tables.
    sd_index.upload(t.showdown_index);
    const size_t S = t.sd_order[0].size();
    std::vector<int32_t> lo, hi, m(S * 2), pos;
    std::vector<uint64_t> own_off(S * 2), ord_off(S * 2), card_off;
    std::vector<uint16_t> ord;
    for (size_t s = 0; s < S; ++s)
      for (int p = 0; p < 2; ++p) {
        const int o = 1 - p;
        const auto& hs = t.sd_strength[p][s];
        const auto& ho = t.sd_order[p][s];
        const auto& os = t.sd_strength[o][s];
        const auto& oo = t.sd_order[o][s];
        own_off[s * 2 + p] = lo.size();
        lo.resize(lo.size() + H[p], -1);
        hi.resize(hi.size() + H[p], -1);
        for (size_t i = 0; i < ho.size(); ++i) {
          lo[own_off[s * 2 + p] + ho[i]] = static_cast<int32_t>(std::lower_bound(os.begin(), os.end(), hs[i]) - os.begin());
          hi[own_off[s * 2 + p] + ho[i]] = static_cast<int32_t>(std::upper_bound(os.begin(), os.end(), hs[i]) - os.begin());
        }
        ord_off[s * 2 + p] = ord.size();
        m[s * 2 + p] = static_cast<int32_t>(oo.size());
        ord.insert(ord.end(), oo.begin(), oo.end());
        std::vector<std::vector<int32_t>> per_card(52);
        for (size_t j = 0; j < oo.size(); ++j) {
          per_card[t.c1[o][oo[j]]].push_back(static_cast<int32_t>(j));
          per_card[t.c2[o][oo[j]]].push_back(static_cast<int32_t>(j));
        }
        for (int c = 0; c < 52; ++c) {
          card_off.push_back(pos.size());
          pos.insert(pos.end(), per_card[c].begin(), per_card[c].end());
        }
        card_off.push_back(pos.size());
      }
    sd_lo.upload(lo), sd_hi.upload(hi), sd_m.upload(m), sd_own_off.upload(own_off), sd_ord_off.upload(ord_off);
    sd_ord.upload(ord), sd_card_off.upload(card_off), sd_card_pos.upload(pos);

    regret.alloc(t.arena_size), ssum.alloc(t.arena_size);
    regret.zero(), ssum.zero();
    R.alloc(nodes * Hmax), V.alloc(nodes * Hmax);
    nch = (Hmax + kChunk - 1) / kChunk;
    prefix.alloc(static_cast<size_t>(std::max<int64_t>(max_show, 1)) * (nch + 1));
    foldsum.alloc(static_cast<size_t>(std::max<int64_t>(max_fold, 1)) * 52);
    device_sync();
  }

  double memory_gb() const {
    return (type.bytes() * 5 + num_children.bytes() * 4 + data.bytes() + contrib0.bytes() * 2 + regret.bytes() +
            ssum.bytes() + R.bytes() + V.bytes() + prefix.bytes() + foldsum.bytes() + sd_lo.bytes() * 2 +
            sd_ord.bytes() + sd_card_pos.bytes()) /
           1e9;
  }

  // Down pass: opponent reach R at every node, and the live flags (a node is visited only if
  // some opponent reach gets there, as in Game::walk). Train mode also updates the
  // opponent's strategy sums.
  void down(int p, int mode, float disc_strat) {
    using namespace flat;
    const int o = 1 - p, Ho = H[o], HM = Hmax;
    live.zero();
    const uint8_t one = 1;
    live.upload_at(0, &one, 1);
    std::vector<float> root(Ho);
    for (int i = 0; i < Ho; ++i) root[i] = host_weight[o][i];
    R.upload_at(0, root.data(), Ho);
    float* Rp = R.p;
    uint8_t* lv = live.p;
    const uint32_t* L = list.p;
    const uint32_t* fc = first_child.p;
    const uint32_t* nc = num_children.p;
    const uint64_t* dat = data.p;
    const uint8_t* dl = dealt.p;
    const uint8_t* oc1 = c1[o].p;
    const uint8_t* oc2 = c2[o].p;
    float* reg = regret.p;
    float* ss = ssum.p;
    for (int l = 0; l + 1 < levels; ++l) {
      // Opponent action nodes.
      {
        const uint32_t* lst = L + beg(l, o == 0 ? kAct0 : kAct1);
        kernel(0, cnt(l, o == 0 ? kAct0 : kAct1) * Ho, LAMBDA(int64_t i) {
          const uint32_t n = lst[i / Ho];
          const int h = static_cast<int>(i % Ho);
          if (!lv[n]) return;
          const int na = static_cast<int>(nc[n]);
          float sig[kMaxActions];
          if (mode == kTrain) current_strategy(reg + dat[n], na, Ho, h, sig);
          else average_strategy(ss + dat[n], na, Ho, h, sig);
          const float r = Rp[static_cast<size_t>(n) * HM + h];
          if (mode == kTrain) {
            float* s = ss + dat[n];
            for (int a = 0; a < na; ++a) s[a * Ho + h] = s[a * Ho + h] * disc_strat + r * sig[a];
          }
          for (int a = 0; a < na; ++a) {
            const uint32_t c = fc[n] + a;
            const float x = r * sig[a];
            Rp[static_cast<size_t>(c) * HM + h] = x;
            if (x > 0) lv[c] = 1;
          }
        });
      }
      // Own action nodes: reach passes through unchanged.
      {
        const uint32_t* lst = L + beg(l, p == 0 ? kAct0 : kAct1);
        kernel(1, cnt(l, p == 0 ? kAct0 : kAct1) * Ho, LAMBDA(int64_t i) {
          const uint32_t n = lst[i / Ho];
          const int h = static_cast<int>(i % Ho);
          if (!lv[n]) return;
          const float r = Rp[static_cast<size_t>(n) * HM + h];
          for (uint32_t a = 0; a < nc[n]; ++a) {
            const uint32_t c = fc[n] + a;
            Rp[static_cast<size_t>(c) * HM + h] = r;
            if (h == 0) lv[c] = 1;
          }
        });
      }
      // Chance nodes: hands that hold the dealt card get no reach.
      {
        const uint32_t* lst = L + beg(l, kChance);
        kernel(2, cnt(l, kChance) * Ho, LAMBDA(int64_t i) {
          const uint32_t n = lst[i / Ho];
          const int h = static_cast<int>(i % Ho);
          if (!lv[n]) return;
          const float r = Rp[static_cast<size_t>(n) * HM + h];
          for (uint32_t k = 0; k < nc[n]; ++k) {
            const uint32_t c = fc[n] + k;
            const float x = (oc1[h] == dl[c] || oc2[h] == dl[c]) ? 0.f : r;
            Rp[static_cast<size_t>(c) * HM + h] = x;
            if (x > 0) lv[c] = 1;
          }
        });
      }
    }
    device_sync();
  }

  // Up pass: values of the traverser's hands at every live node, deepest level first.
  void up(int p, int mode, float disc_pos, float disc_neg) {
    using namespace flat;
    const int o = 1 - p, Hp = H[p], HM = Hmax;
    float* Rp = R.p;
    float* Vp = V.p;
    const uint8_t* lv = live.p;
    const uint32_t* L = list.p;
    const uint32_t* fc = first_child.p;
    const uint32_t* nc = num_children.p;
    const uint32_t* fe = first_entry.p;
    const uint32_t* bid = board_id.p;
    const uint64_t* dat = data.p;
    const uint8_t* bs = board_size.p;
    const uint8_t* pl = player.p;
    const float* ct0 = contrib0.p;
    const float* ct1 = contrib1.p;
    const uint8_t* hc1 = c1[p].p;
    const uint8_t* hc2 = c2[p].p;
    const int32_t* sm = same[p].p;
    const uint32_t* choff = ch_off[o].p;
    const uint16_t* chidx = ch_idx[o].p;
    const int NCH = nch, CH = kChunk;
    const uint8_t* ecard = e_card.p;
    const uint8_t* eperm = e_perm.p;
    const uint16_t* echild = e_child.p;
    const int32_t* pinv = perm_inv.p;
    const uint16_t* hperm = hand_perm[p].p;
    const int32_t* sdi = sd_index.p;
    const int32_t* slo = sd_lo.p;
    const int32_t* shi = sd_hi.p;
    const int32_t* smm = sd_m.p;
    const uint64_t* sown = sd_own_off.p;
    const uint64_t* sord_off = sd_ord_off.p;
    const uint16_t* sord = sd_ord.p;
    const uint64_t* scoff = sd_card_off.p;
    const int32_t* scpos = sd_card_pos.p;
    acc_t* pre = prefix.p;
    acc_t* fs = foldsum.p;
    float* reg = regret.p;
    float* ss = ssum.p;
    for (int l = levels - 1; l >= 0; --l) {
      // Fold nodes: card sums of the opponent reach, then compatible reach x payoff.
      if (cnt(l, kFold)) {
        const uint32_t* lst = L + beg(l, kFold);
        // One thread per (node, card): reach of the opponent hands that hold the card.
        kernel(3, cnt(l, kFold) * 52, LAMBDA(int64_t i) {
          const int64_t j = i / 52;
          const int c = static_cast<int>(i % 52);
          const uint32_t n = lst[j];
          if (!lv[n]) return;
          const float* r = Rp + static_cast<size_t>(n) * HM;
          acc_t sum = 0;
          for (uint32_t a = choff[c]; a < choff[c + 1]; ++a) sum += r[chidx[a]];
          fs[j * 52 + c] = sum;
        });
        kernel(4, cnt(l, kFold) * Hp, LAMBDA(int64_t i) {
          const int64_t j = i / Hp;
          const uint32_t n = lst[j];
          const int h = static_cast<int>(i % Hp);
          if (!lv[n]) return;
          const acc_t* f = fs + j * 52;
          acc_t total = 0;  // every hand holds two cards
          for (int c = 0; c < 52; ++c) total += f[c];
          acc_t v = total * 0.5f - f[hc1[h]] - f[hc2[h]];
          if (sm[h] >= 0) v += Rp[static_cast<size_t>(n) * HM + sm[h]];
          const int fp = pl[n];
          const acc_t c = fp == 0 ? ct0[n] : ct1[n];
          Vp[static_cast<size_t>(n) * HM + h] = static_cast<float>(v * (p == fp ? -c : c));
        });
      }
      // Showdown nodes: prefix sums of opponent reach in strength order, then win - lose with
      // card removal from the per-card position lists.
      if (cnt(l, kShow)) {
        const uint32_t* lst = L + beg(l, kShow);
        // Chunked prefix sums: S[c + 1] = reach in chunks 0..c (sorted order). Chunk sums run in
        // parallel; then one short prefix per node over the chunks.
        kernel(5, cnt(l, kShow) * NCH, LAMBDA(int64_t i) {
          const int64_t j = i / NCH;
          const int c = static_cast<int>(i % NCH);
          const uint32_t n = lst[j];
          if (!lv[n]) return;
          const int64_t b = static_cast<int64_t>(sdi[bid[n]]) * 2 + p;
          const int m = smm[b];
          const uint16_t* ord = sord + sord_off[b];
          const float* r = Rp + static_cast<size_t>(n) * HM;
          const int k0 = c * CH, k1 = k0 + CH < m ? k0 + CH : m;
          acc_t sum = 0;
          for (int k = k0; k < k1; ++k) sum += r[ord[k]];
          pre[j * (NCH + 1) + c + 1] = sum;
        });
        kernel(5, cnt(l, kShow), LAMBDA(int64_t j) {
          const uint32_t n = lst[j];
          if (!lv[n]) return;
          acc_t* S = pre + j * (NCH + 1);
          S[0] = 0;
          for (int c = 0; c < NCH; ++c) S[c + 1] += S[c];
        });
        kernel(6, cnt(l, kShow) * Hp, LAMBDA(int64_t i) {
          const int64_t j = i / Hp;
          const uint32_t n = lst[j];
          const int h = static_cast<int>(i % Hp);
          if (!lv[n]) return;
          const int64_t b = static_cast<int64_t>(sdi[bid[n]]) * 2 + p;
          const int lo = slo[sown[b] + h];
          float* out = Vp + static_cast<size_t>(n) * HM + h;
          if (lo < 0) {
            *out = 0;
            return;
          }
          const int hi = shi[sown[b] + h], m = smm[b];
          const uint16_t* ord = sord + sord_off[b];
          const float* r = Rp + static_cast<size_t>(n) * HM;
          const acc_t* S = pre + j * (NCH + 1);
          // P(x) = reach of sorted positions < x = full chunks + the rest of x's chunk.
          acc_t P_lo = S[lo / CH], P_hi = S[hi / CH];
          const acc_t P_m = S[(m + CH - 1) / CH];  // the chunks cover the whole list
          for (int k = (lo / CH) * CH; k < lo; ++k) P_lo += r[ord[k]];
          for (int k = (hi / CH) * CH; k < hi; ++k) P_hi += r[ord[k]];
          acc_t win = P_lo, lose = P_m - P_hi;
          const int cards[2] = {hc1[h], hc2[h]};
          for (int q = 0; q < 2; ++q) {
            const uint64_t a0 = scoff[b * 53 + cards[q]], a1 = scoff[b * 53 + cards[q] + 1];
            for (uint64_t a = a0; a < a1; ++a) {
              const int ps = scpos[a];
              const float x = r[ord[ps]];
              if (ps < lo) win -= x;
              else if (ps >= hi) lose -= x;
            }
          }
          *out = static_cast<float>(ct0[n] * (win - lose));
        });
      }
      // Own action nodes: strategy-weighted value; regret update (train) or best action.
      {
        const uint32_t* lst = L + beg(l, p == 0 ? kAct0 : kAct1);
        kernel(7, cnt(l, p == 0 ? kAct0 : kAct1) * Hp, LAMBDA(int64_t i) {
          const uint32_t n = lst[i / Hp];
          const int h = static_cast<int>(i % Hp);
          if (!lv[n]) return;
          const int na = static_cast<int>(nc[n]);
          float val[kMaxActions] = {0};
          for (int a = 0; a < na; ++a) val[a] = Vp[static_cast<size_t>(fc[n] + a) * HM + h];
          float out = 0;
          if (mode == kBest) {
            out = val[0];
            for (int a = 1; a < na; ++a) out = val[a] > out ? val[a] : out;
          } else {
            float sig[kMaxActions];
            if (mode == kTrain) current_strategy(reg + dat[n], na, Hp, h, sig);
            else average_strategy(ss + dat[n], na, Hp, h, sig);
            for (int a = 0; a < na; ++a) out += sig[a] * val[a];
            if (mode == kTrain) {
              float* r = reg + dat[n];
              for (int a = 0; a < na; ++a) {
                float x = r[a * Hp + h];
                x *= x > 0 ? disc_pos : disc_neg;
                r[a * Hp + h] = x + val[a] - out;
              }
            }
          }
          Vp[static_cast<size_t>(n) * HM + h] = out;
        });
      }
      // Opponent action nodes: sum over the live children.
      {
        const uint32_t* lst = L + beg(l, o == 0 ? kAct0 : kAct1);
        kernel(8, cnt(l, o == 0 ? kAct0 : kAct1) * Hp, LAMBDA(int64_t i) {
          const uint32_t n = lst[i / Hp];
          const int h = static_cast<int>(i % Hp);
          if (!lv[n]) return;
          float out = 0;
          for (uint32_t a = 0; a < nc[n]; ++a) {
            const uint32_t c = fc[n] + a;
            if (lv[c]) out += Vp[static_cast<size_t>(c) * HM + h];
          }
          Vp[static_cast<size_t>(n) * HM + h] = out;
        });
      }
      // Chance nodes: every dealable card adds the value of its representative subtree, read
      // through the suit permutation; then divide by the number of cards.
      {
        const uint32_t* lst = L + beg(l, kChance);
        kernel(9, cnt(l, kChance) * Hp, LAMBDA(int64_t i) {
          const uint32_t n = lst[i / Hp];
          const int h = static_cast<int>(i % Hp);
          if (!lv[n]) return;
          const int ne = 52 - bs[n];
          float out = 0;
          for (int e = 0; e < ne; ++e) {
            const uint32_t idx = fe[n] + e;
            const uint32_t c = fc[n] + echild[idx];
            if (!lv[c]) continue;
            const uint8_t d = ecard[idx];
            if (hc1[h] == d || hc2[h] == d) continue;
            const int pm = eperm[idx];
            const int hh = pm == 0 ? h : hperm[static_cast<size_t>(pinv[pm]) * Hp + h];
            out += Vp[static_cast<size_t>(c) * HM + hh];
          }
          Vp[static_cast<size_t>(n) * HM + h] = out / static_cast<float>(48 - bs[n]);
        });
      }
    }
    device_sync();
  }

  void run_iteration() {
    ++iteration;
    const double t = iteration;
    const double ta = std::pow(t, 1.5), tb = std::pow(t, 0.0);
    const float dpos = static_cast<float>(ta / (ta + 1)), dneg = static_cast<float>(tb / (tb + 1));
    const float dstrat = static_cast<float>(std::pow(t / (t + 1), 2.0));
    for (int p = 0; p < 2; ++p) {
      down(p, flat::kTrain, dstrat);
      up(p, flat::kTrain, dpos, dneg);
    }
  }

  double exploitability() {
    double br = 0;
    for (int p = 0; p < 2; ++p) {
      down(p, flat::kAverage, 1.f);
      up(p, flat::kBest, 1.f, 1.f);
      std::vector<float> v(H[p]);
      V.download_at(0, v.data(), H[p]);
      for (int h = 0; h < H[p]; ++h) br += static_cast<double>(host_weight[p][h]) * v[h];
    }
    return br / pair_weight_sum / 2;
  }
};

FlatSolver::FlatSolver(const FlatTree& t) : im_(new FlatSolverImpl(t)) {}
FlatSolver::~FlatSolver() = default;
void FlatSolver::run_iteration() { im_->run_iteration(); }
double FlatSolver::exploitability() { return im_->exploitability(); }
int FlatSolver::iterations() const { return im_->iteration; }
double FlatSolver::memory_gb() const { return im_->memory_gb(); }
const char* FlatSolver::backend() { return backend_name(); }
void FlatSolver::set_profile(bool on) { im_->prof = on; }
std::string FlatSolver::profile_report() const { return im_->profile_report(); }

}  // namespace poker
