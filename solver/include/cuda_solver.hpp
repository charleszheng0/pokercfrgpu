#pragma once

#include "cfr.hpp"

#include <memory>

namespace poker {

bool cuda_backend_available();

// CUDA is optional. A CUDA build supplies a batched, flat-array backend;
// non-CUDA builds retain this interface and report unavailability cleanly.
class CFRSolverCUDA final : public Solver {
 public:
  CFRSolverCUDA();
  ~CFRSolverCUDA() override;
  CFRSolverCUDA(CFRSolverCUDA&&) noexcept;
  CFRSolverCUDA& operator=(CFRSolverCUDA&&) noexcept;
  CFRSolverCUDA(const CFRSolverCUDA&) = delete;
  CFRSolverCUDA& operator=(const CFRSolverCUDA&) = delete;
  SolverResult solve(int iterations) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace poker
