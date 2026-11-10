// Leaf value network for depth-limited flop solving (DeepStack / ReBeL style).
//
// At a leaf (the end of the flop) the value of a hand depends on the board, the two ranges
// and the stacks. The model works per hand: it gets features of the hand against the
// opponent range and of both ranges, and predicts how much the hand wins (in units of the
// pot share at the leaf, per unit of compatible opponent reach) above its check-down value.
//
//   y(h) = v(h) / (contrib * M(h))        M(h) = compatible opponent reach
//   d(h) = check-down value of h: E[win - lose] over all turn and river cards, in [-1, 1]
//   net  predicts y(h) - d(h)
//
// With no network the evaluator returns the check-down value (an "equity only" leaf).
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "game.hpp"

namespace poker {

constexpr int kLeafFeatures = 32;
constexpr int kRunoutBins = 6;  // per-hand histogram of the runout value d_B

// The turn and river runouts of one flop: for each pair of cards, the hands of each player
// that do not use them, sorted by strength.
class FlopRunouts {
 public:
  FlopRunouts(const std::vector<Card>& flop, const std::vector<std::pair<Card, Card>> hands[2]);
  struct Runout {
    Card a, b;
    std::vector<uint32_t> strength[2];
    std::vector<uint16_t> order[2];
  };
  std::vector<Runout> runouts;
  std::vector<std::pair<Card, Card>> hands[2];
  std::vector<int32_t> same[2];  // index of the same combo in the other player's list, or -1
  // eq[h * n1 + o] = E[win - lose] of OOP hand h against IP hand o over all turn and river
  // cards (0 if the hands share a card). The check-down value is then one matrix product.
  std::vector<float> eq;
};

// Features that need a pass over all runouts (draw strength). They change slowly with the
// ranges, so a solver can refresh them less often than the others.
struct SlowFeatures {
  std::vector<float> e2[2], sd[2];
  std::vector<float> hist[2];  // n x kRunoutBins: share of runouts (mass-weighted) per bin of d_B
  std::vector<float> absd[2];  // E[|d_B|]
};
void compute_slow_features(const FlopRunouts& R, const float* const r[2], SlowFeatures& out);

struct LeafFeatures {
  std::vector<float> x[2];     // n_p x kLeafFeatures
  std::vector<float> d[2];     // check-down value
  std::vector<float> mass[2];  // compatible opponent reach M(h)
};

// slow == nullptr: compute the slow features too (from the same ranges).
void compute_leaf_features(const FlopRunouts& R, const float* const r[2], double contrib, double stack_left,
                           LeafFeatures& out, const SlowFeatures* slow = nullptr);

// Small fully connected network: ReLU hidden layers, one linear output.
class Mlp {
 public:
  std::vector<int> sizes;
  std::vector<std::vector<float>> W, b;  // W[l] is sizes[l+1] x sizes[l], row-major
  std::vector<float> mean, inv_std;      // input normalization

  void init(const std::vector<int>& layer_sizes, unsigned seed);
  float forward(const float* x) const;  // one row of raw (not normalized) features
  void save(const std::string& path) const;
  void load(const std::string& path);
  size_t num_params() const;
};

struct TrainOptions {
  int epochs = 30;
  int batch = 1024;
  double lr = 1e-3;
  int threads = 8;
  unsigned seed = 1;
};
// X: rows x kLeafFeatures, Y: targets. Returns the train MSE of the last epoch. Prints the
// validation error after each epoch if Xv is not empty.
double train_mlp(Mlp& net, const std::vector<float>& X, const std::vector<float>& Y, const std::vector<float>& Xv,
                 const std::vector<float>& Yv, const TrainOptions& opt);

// LeafEvaluator for a depth-limited flop tree. net == nullptr: check-down values only.
class NetLeafEvaluator : public LeafEvaluator {
 public:
  NetLeafEvaluator(const Game& g, const Mlp* net);
  void evaluate(const Game& g, const LeafInfo& leaf, const float* const r[2], float* const v[2]) const override;

 private:
  FlopRunouts runouts_;
  const Mlp* net_;
  std::vector<int> leaf_slot_;  // node id -> cache slot
  struct Cache {
    SlowFeatures slow;
    int calls = 0;
  };
  mutable std::vector<Cache> cache_;  // one per leaf; a leaf is evaluated by one thread at a time
};

}  // namespace poker
