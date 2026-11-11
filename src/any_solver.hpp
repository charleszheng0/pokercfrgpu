// One solve loop for the CPU solver (Game) and the level-by-level solver (gpu/, CUDA or CPU threads).
//
// The programs pokersolve, pokerbatch and pokerlab take --gpu 1. A build with -DPOKER_CUDA=ON makes
// pokersolve_gpu, pokerbatch_gpu and pokerlab_gpu, which link the CUDA back end; there --gpu 1 runs
// full-tree solves on the GPU. In a normal build there is no CUDA back end, so --gpu 1 prints a
// note and the CPU solver runs as before.
//
// After the last iteration, sync() loads the GPU result back into the Game. Everything that reads
// the Game afterwards (EVs, node views, the spot extraction of pokerbatch) works without changes.
#pragma once
#include <cstdio>
#include <memory>
#include <string>

#include "flat_solver.hpp"
#include "game.hpp"

namespace poker {

// True when this program links the CUDA back end of the level-by-level solver.
inline bool cuda_build() { return std::string(FlatSolver::backend()) == "cuda"; }

// Decide whether a program that got --gpu runs its solves on the GPU. Prints a note once when the
// build has no CUDA back end.
inline bool use_gpu(bool requested, const char* program) {
  if (!requested) return false;
  if (cuda_build()) return true;
  std::fprintf(stderr,
               "%s: --gpu 1 needs a CUDA build (cmake -DPOKER_CUDA=ON, then %s_gpu); "
               "running on the CPU solver\n",
               program, program);
  return false;
}

// The GPU result loads back as 32-bit regrets, so a GPU solve needs 32-bit storage.
inline void prepare_gpu_config(GameConfig& cfg, bool gpu) {
  if (gpu) cfg.compress = 0;
}

class AnySolver {
 public:
  // flat = run on the level-by-level solver. Falls back to Game (with a reason) for depth-limited
  // trees, 16-bit storage, or when the solver cannot be set up (for example not enough GPU memory).
  AnySolver(Game& game, bool flat) : g_(game) {
    if (!flat) return;
    if (g_.num_leaves() > 0) {
      fallback_ = "depth-limited tree";
    } else if (g_.compressed()) {
      fallback_ = "16-bit storage";
    } else {
      try {
        fs_ = std::make_unique<FlatSolver>(g_.export_flat());
      } catch (const std::exception& e) {
        fs_.reset();
        fallback_ = e.what();
      }
    }
  }
  bool on_flat() const { return fs_ != nullptr; }
  const std::string& fallback_reason() const { return fallback_; }  // empty when not asked or no fallback
  const char* backend() const { return fs_ ? FlatSolver::backend() : "cpu-dfs"; }

  void run_iteration() { fs_ ? fs_->run_iteration() : g_.run_iteration(); }
  double exploitability() { return fs_ ? fs_->exploitability() : g_.exploitability(); }  // chips
  int iterations() const { return fs_ ? fs_->iterations() : g_.iterations(); }

  // Load the level-by-level result into the Game (no-op for Game solves).
  void sync() {
    if (!fs_) return;
    std::vector<float> r, s;
    fs_->download(r, s);
    g_.import_arenas(r, s, fs_->iterations());
  }

 private:
  Game& g_;
  std::unique_ptr<FlatSolver> fs_;
  std::string fallback_;
};

}  // namespace poker
