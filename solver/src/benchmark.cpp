#include "cfr.hpp"
#include "cuda_solver.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 100000;
  poker::CFRSolverCPU cpu;
  const auto result = cpu.solve(iterations);
  std::cout << "Backend        iterations/sec    states/sec       total time\n"
            << std::fixed << std::setprecision(2)
            << "CPU            " << std::setw(14) << result.iterations_per_second << "    "
            << std::setw(12) << result.states_per_second << "    " << result.elapsed_ms << " ms\n";

  bool has_cuda = poker::cuda_backend_available();
  poker::SolverResult cuda_result;
  if (has_cuda) {
    poker::CFRSolverCUDA cuda;
    cuda_result = cuda.solve(iterations);
    std::cout << "CUDA-hybrid    " << std::setw(14) << cuda_result.iterations_per_second << "    "
              << std::setw(12) << cuda_result.states_per_second << "    " << cuda_result.elapsed_ms << " ms\n";
  } else {
    std::cout << "CUDA           unavailable (not measured)\n";
  }

  std::ofstream json("benchmarks/results.json");
  if (json) {
    json << std::setprecision(10)
         << "{\n  \"game\": \"kuhn\",\n  \"iterations\": " << iterations
         << ",\n  \"cpu\": {\n    \"elapsed_ms\": " << result.elapsed_ms
         << ",\n    \"iterations_per_second\": " << result.iterations_per_second
         << ",\n    \"states_visited\": " << result.states_visited
         << ",\n    \"states_per_second\": " << result.states_per_second
         << ",\n    \"nash_conv\": " << result.nash_conv << "\n  },\n"
         << "  \"cuda\": ";
    if (has_cuda) {
      json << "{\n    \"backend\": \"" << cuda_result.backend
           << "\",\n    \"elapsed_ms\": " << cuda_result.elapsed_ms
           << ",\n    \"iterations_per_second\": " << cuda_result.iterations_per_second
           << ",\n    \"states_per_second\": " << cuda_result.states_per_second << "\n  }\n";
    } else {
      json << "null\n";
    }
    json << "}\n";
  }
}
