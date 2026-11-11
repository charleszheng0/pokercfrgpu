// Level-by-level DCFR for GPUs. Same algorithm and DCFR parameters as Game (alpha 1.5,
// beta 0, gamma 2), same alternating updates, so the results must match. Only the work order
// changes: Game walks the tree depth-first; this solver does one pass down the tree (reach)
// and one pass up (values, regrets), one tree level at a time, with one thread per
// (node, hand). The back end (CUDA or CPU threads) is chosen when it is compiled.
#pragma once
#include <memory>
#include <string>

#include "flat_tree.hpp"

namespace poker {

struct FlatSolverImpl;

class FlatSolver {
 public:
  explicit FlatSolver(const FlatTree& t);
  ~FlatSolver();
  void run_iteration();
  double exploitability();  // chips, like Game::exploitability()
  int iterations() const;
  double memory_gb() const;  // device memory in use
  static const char* backend();
  void set_profile(bool on);           // sync after every kernel and time it
  std::string profile_report() const;  // seconds and launches per kernel kind

 private:
  std::unique_ptr<FlatSolverImpl> im_;
};

}  // namespace poker
