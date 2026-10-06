#include "cuda_solver.hpp"

#include <stdexcept>

namespace poker {

struct CFRSolverCUDA::Impl {};

bool cuda_backend_available() { return false; }
CFRSolverCUDA::CFRSolverCUDA() : impl_(std::make_unique<Impl>()) {}
CFRSolverCUDA::~CFRSolverCUDA() = default;
CFRSolverCUDA::CFRSolverCUDA(CFRSolverCUDA&&) noexcept = default;
CFRSolverCUDA& CFRSolverCUDA::operator=(CFRSolverCUDA&&) noexcept = default;

SolverResult CFRSolverCUDA::solve(int) {
  throw std::runtime_error(
      "CUDA backend is unavailable: configure with -DPOKER_ENABLE_CUDA=ON on an NVIDIA CUDA system");
}

}  // namespace poker
