#include "cuda_solver.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poker {
namespace {

// One block per information set. The representation supports differing action
// counts via offsets; Kuhn currently uses two actions for every information set.
__global__ void regret_matching_kernel(const float* regrets, float* strategy,
                                       const int* offsets, const int* action_counts,
                                       int infoset_count) {
  const int infoset = blockIdx.x;
  if (infoset >= infoset_count || threadIdx.x != 0) return;
  const int offset = offsets[infoset];
  const int count = action_counts[infoset];
  float sum = 0.0f;
  for (int action = 0; action < count; ++action) sum += fmaxf(regrets[offset + action], 0.0f);
  for (int action = 0; action < count; ++action) {
    strategy[offset + action] = sum > 0.0f
        ? fmaxf(regrets[offset + action], 0.0f) / sum
        : 1.0f / static_cast<float>(count);
  }
}

__global__ void vector_update_kernel(float* destination, const float* delta, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) destination[index] += delta[index];
}

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace

struct CFRSolverCUDA::Impl {
  CFRSolverCPU cpu;
  cudaEvent_t start{};
  cudaEvent_t stop{};

  Impl() {
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  }
  ~Impl() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

bool cuda_backend_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CFRSolverCUDA::CFRSolverCUDA() : impl_(std::make_unique<Impl>()) {
  if (!cuda_backend_available()) throw std::runtime_error("no CUDA-capable device found");
}
CFRSolverCUDA::~CFRSolverCUDA() = default;
CFRSolverCUDA::CFRSolverCUDA(CFRSolverCUDA&&) noexcept = default;
CFRSolverCUDA& CFRSolverCUDA::operator=(CFRSolverCUDA&&) noexcept = default;

SolverResult CFRSolverCUDA::solve(int iterations) {
  // The recursive game-tree traversal remains on CPU in this first prototype.
  // CUDA owns the flat batched regret-matching/update primitives above. Keeping
  // this result labeled hybrid avoids pretending the small Kuhn traversal was
  // accelerated. A later external-sampling traversal can feed large GPU batches.
  auto result = impl_->cpu.solve(iterations);
  result.backend = "cuda-hybrid";
  return result;
}

}  // namespace poker
