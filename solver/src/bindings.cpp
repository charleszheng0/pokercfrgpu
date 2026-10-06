#include "cfr.hpp"
#include "cuda_solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_poker_solver, module) {
  module.def("cuda_available", &poker::cuda_backend_available);
  module.def("solve", [](int iterations, const std::string& backend) {
    std::unique_ptr<poker::Solver> solver;
    if (backend == "cpu") solver = std::make_unique<poker::CFRSolverCPU>();
    else if (backend == "cuda") solver = std::make_unique<poker::CFRSolverCUDA>();
    else throw std::invalid_argument("backend must be cpu or cuda");
    const auto result = solver->solve(iterations);
    py::dict strategies;
    for (const auto& [name, entry] : result.strategy) {
      strategies[py::str(name)] = py::dict("pass"_a=entry.pass, "bet"_a=entry.bet);
    }
    return py::dict(
        "strategy"_a=strategies,
        "iterations"_a=result.iterations,
        "elapsed_ms"_a=result.elapsed_ms,
        "states_visited"_a=result.states_visited,
        "states_per_second"_a=result.states_per_second,
        "iterations_per_second"_a=result.iterations_per_second,
        "game_value_player0"_a=result.game_value_player0,
        "nash_conv"_a=result.nash_conv,
        "exploitability"_a=result.exploitability,
        "backend"_a=result.backend);
  }, py::arg("iterations"), py::arg("backend") = "cpu");
}
