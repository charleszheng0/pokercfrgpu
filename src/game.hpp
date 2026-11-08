// Heads-up postflop game tree and the vector-form DCFR solver.
//
// Layout: one flat array of nodes. Each action node owns a slice of two float
// arenas (regrets and strategy sums), sized num_actions x num_hands(actor).
// One traversal handles every hand of both players at once (vector form):
// the traverser gets a value vector, the opponent passes a reach vector.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cards.hpp"
#include "range.hpp"
#include "thread_pool.hpp"

namespace poker {

struct BetSizes {
  std::vector<double> bet;    // fractions of the pot, e.g. 0.33, 0.75
  std::vector<double> raise;  // fractions of the pot after the call
};

struct GameConfig {
  std::vector<Card> board;  // 3, 4 or 5 cards
  double pot = 10;          // pot at the start of the street (both players paid half)
  double stack = 95;        // effective stack behind
  Range range[2];           // 0 = OOP (acts first), 1 = IP
  BetSizes sizes[2][3];     // per player (0 OOP, 1 IP), per street (0 flop, 1 turn, 2 river)
  int max_raises = 3;       // raises allowed per street after the first bet
  bool add_allin = true;
  double allin_threshold = 0.67;  // a bet above this share of the stack becomes all-in
  double mem_limit_gb = 3.0;
  int threads = 0;  // 0 = all hardware threads
  // Storage of regrets and strategy sums: 0 = 32-bit floats, 1 = 16-bit with one scale
  // per node, -1 = auto (16-bit only when 32-bit does not fit in mem_limit_gb).
  int compress = -1;
  // Solve one card of each group of suit-symmetric cards and copy the result to the others.
  bool isomorphism = true;
};

enum class NodeType : uint8_t { Action, Fold, Showdown, Chance };
enum class ActKind : uint8_t { Fold, Check, Call, Bet, Raise, AllIn };

struct ActionInfo {
  ActKind kind;
  double amount;  // chips the actor adds with this action
};

struct Node {
  NodeType type;
  uint8_t player = 0;      // actor (Action) or the player who folded (Fold)
  uint8_t board_size = 0;  // board cards at this node
  uint8_t dealt = 255;     // card dealt to reach this node (child of a chance node)
  uint32_t num_children = 0;
  uint32_t first_child = 0;   // children are nodes_[first_child .. +num_children)
  uint32_t first_action = 0;  // into actions_ (Action nodes) or chance_entries_ (Chance nodes)
  uint32_t slot = 0;          // action node number (index of its 16-bit scales)
  uint32_t board_id = 0;
  uint64_t data = 0;  // offset into the arenas (Action nodes)
  double contrib[2] = {0, 0};  // chips each player has in the pot (incl. starting pot halves)
};

// One dealable card of a chance node: the card, the child (a representative card) that
// holds its subtree, and the suit permutation that maps the representative to the card.
struct ChanceEntry {
  Card card;
  uint8_t perm;
  uint16_t child;
};

struct HandInfo {
  Card c1, c2;
  float weight;
  uint16_t combo;
};

// The solved data at one node, with cards as the user sees them (suit symmetry undone).
struct HandView {
  int combo;                // combo index (see range.hpp)
  double w;                 // reach weight of this hand at the node
  std::vector<double> s;    // average strategy per action
  std::vector<double> ev;   // EV per action: chips won from the pot from this node on
  double e;                 // EV of the mixed strategy
};
struct NodeView {
  std::string type;  // "action", "chance", "fold", "showdown"
  std::vector<Card> board;
  double pot = 0, contrib[2] = {0, 0};
  std::vector<std::string> history;
  int player = -1, winner = -1;
  std::vector<ActionInfo> action_info;
  std::vector<std::string> actions;
  std::vector<double> freq;  // range-wide frequency per action
  double combos = 0;
  std::vector<HandView> hands;
  std::vector<Card> cards;  // chance node: cards that can come
};

class Game {
 public:
  explicit Game(const GameConfig& cfg);  // throws std::invalid_argument / std::runtime_error

  const GameConfig& config() const { return cfg_; }
  size_t num_nodes() const { return nodes_.size(); }
  size_t num_action_nodes() const { return num_action_nodes_; }
  double memory_bytes() const;  // regrets + strategy sums
  bool compressed() const { return compressed_; }
  size_t chance_cards_solved() const { return chance_solved_; }  // dealt cards with their own subtree
  size_t chance_cards_total() const { return chance_total_; }    // all dealt cards
  int num_hands(int p) const { return static_cast<int>(hands_[p].size()); }
  int iterations() const { return iteration_; }

  void run_iteration();      // one DCFR iteration (both players)
  double exploitability();   // chips: (BR value of OOP + BR value of IP) / 2
  std::array<double, 2> expected_values();  // per player, chips won from the starting pot

  // JSON for the node reached by a path. Tokens: action index, or a card such as "Kh"
  // after a chance node. Throws std::invalid_argument on a bad path.
  std::string node_json(const std::vector<std::string>& path);
  NodeView node_view(const std::vector<std::string>& path);
  static std::string to_json(const NodeView& v, bool with_grid = true);  // grid: 13x13 summary

  // Test hook: compares the fast O(n) showdown and fold values with a slow O(n^2)
  // loop over every hand pair, on random reach vectors. Returns the largest error.
  double self_check_terminals(int max_nodes, unsigned seed);

 private:
  enum class Mode { Train, BestResponse, Average };
  struct BuildState {
    uint32_t board_id;
    int board_size;
    double contrib[2];
    int actor;
    int raises;
  };
  struct Showdown {
    std::vector<uint32_t> strength[2];  // sorted ascending
    std::vector<uint16_t> order[2];     // hand index for each sorted entry
  };

  void build_decision(uint32_t idx, const BuildState& st);
  void build_street_end(uint32_t idx, const BuildState& st);
  std::vector<ActionInfo> legal_actions(const BuildState& st) const;
  void prepare_showdowns();

  void walk(uint32_t idx, int p, const float* opp, float* out, Mode mode);
  void current_strategy(const Node& n, float* sigma) const;
  void average_strategy(const Node& n, float* sigma) const;
  // Regrets and strategy sums go through these, so the 32-bit and 16-bit layouts share code.
  void load_regrets(const Node& n, float* dst) const;
  void store_regrets(const Node& n, const float* src);
  void load_strategy_sum(const Node& n, float* dst) const;
  void store_strategy_sum(const Node& n, const float* src);
  void prepare_symmetry();
  std::vector<int> symmetry_group(const std::vector<Card>& board) const;
  void compat_sums(int p, const float* opp, float* out) const;  // sum of compatible opp reach
  void fold_values(const Node& n, int p, const float* opp, float* out) const;
  void showdown_values(const Node& n, int p, const float* opp, float* out) const;
  double pair_weight_sum() const;
  std::string action_label(const ActionInfo& a) const;

  GameConfig cfg_;
  std::vector<Node> nodes_;
  std::vector<ActionInfo> actions_;
  std::vector<std::vector<Card>> boards_;
  std::vector<int32_t> showdown_index_;  // per board id, -1 if not a river board
  std::vector<Showdown> showdowns_;
  std::vector<HandInfo> hands_[2];
  std::vector<int32_t> same_combo_[2];  // index of the same combo in the other list, or -1
  std::vector<float> regrets_, strategy_sum_;     // 32-bit layout
  std::vector<int16_t> regrets16_;                 // 16-bit layout: value = q * scale
  std::vector<uint16_t> strategy_sum16_;
  std::vector<float> regret_scale_, strategy_scale_;  // per action node
  bool compressed_ = false;
  // Suit symmetry: perm id 0..23; usable_[k] = the ranges and the starting board do not
  // change under perm k. hand_perm_[k][p][h] = index of the image of hand h.
  bool usable_[24] = {};
  std::vector<uint16_t> hand_perm_[24][2];
  std::vector<ChanceEntry> chance_entries_;
  size_t chance_solved_ = 0, chance_total_ = 0;
  uint64_t arena_size_ = 0;
  size_t num_action_nodes_ = 0;
  int iteration_ = 0;
  // DCFR discount factors of the current iteration
  float disc_pos_ = 1, disc_neg_ = 1, disc_strat_ = 1;
  std::unique_ptr<ThreadPool> pool_;
};

}  // namespace poker
