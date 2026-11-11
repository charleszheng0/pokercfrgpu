// A flat copy of a solved-game tree for the level-by-level (GPU) solver. Plain arrays only,
// so they can go to GPU memory as they are. Made by Game::export_flat().
#pragma once
#include <cstdint>
#include <vector>

namespace poker {

struct FlatTree {
  int n_hands[2] = {0, 0};
  std::vector<uint8_t> c1[2], c2[2];  // hole cards of each hand
  std::vector<float> weight[2];       // starting range weight of each hand
  std::vector<int32_t> same[2];       // index of the same combo in the other list, or -1

  // Nodes (index = node id). type: 0 action, 1 fold, 2 showdown, 3 chance (NodeType order).
  std::vector<uint8_t> type, player, board_size, dealt;
  std::vector<uint32_t> num_children, first_child, first_entry, board_id;
  std::vector<uint64_t> data;  // arena offset of action nodes
  std::vector<float> contrib0, contrib1;
  uint64_t arena_size = 0;

  // Chance entries: every dealable card, its suit permutation and the child that holds it.
  std::vector<uint8_t> e_card, e_perm;
  std::vector<uint16_t> e_child;
  int perm_inv[24] = {0};
  std::vector<uint16_t> hand_perm[2];  // 24 x n_hands[p]: image of hand h under perm k

  // Showdowns: per board id the showdown index or -1; per showdown, hands sorted by strength.
  std::vector<int32_t> showdown_index;
  std::vector<std::vector<uint32_t>> sd_strength[2];
  std::vector<std::vector<uint16_t>> sd_order[2];

  // Node locking (first street): per node an offset into lock_data, or -1.
  std::vector<int64_t> lock_off;
  std::vector<float> lock_data;

  double pot = 0, pair_weight_sum = 0;
};

}  // namespace poker
