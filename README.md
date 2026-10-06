# Poker GTO Solver

A compact, GPU-oriented imperfect-information game solver prototype using C++17,
CUDA-ready flat kernels, CFR, pybind11, FastAPI, PyTorch interfaces, and React.
The complete working game is **Kuhn Poker**, chosen because its equilibrium value
and exploitability can be checked rather than guessed.

```text
React Solver Lab
      ↓ HTTP
FastAPI
      ↓ pybind11
C++ game engine → CPU CFR traversal
                → CUDA flat-array kernels (optional/hybrid prototype)
```

## Why these pieces

Poker hides private information, so ordinary perfect-information minimax is not
the right model. Counterfactual Regret Minimization learns a behavioral strategy
at each information set by repeatedly comparing each action with the value of the
strategy actually played. Regret matching normalizes positive cumulative regret;
when none is positive, this implementation plays uniformly. The reported policy
is the reach-weighted average strategy across iterations.

CUDA matters at larger scales because regret matching and cumulative table updates
repeat the same arithmetic across very many information sets. The CUDA source here
starts with contiguous regret/strategy arrays, offsets, action counts, and batched
kernels. It deliberately keeps Kuhn's recursive traversal on CPU and labels a CUDA
build `cuda-hybrid`; it does not claim a GPU speedup for a 12-information-set game.

PyTorch comes later: a learned continuation-value function can stop traversal at a
depth limit and enable subgame re-solving. `ml/` defines that interface but ships no
trained model and reports no neural metrics.

## Native build and tests

Requirements: CMake 3.20+ and a C++17 compiler.

```bash
mkdir build
cd build
cmake ..
cmake --build . -j
ctest --output-on-failure
cd ..
./build/poker_solver --iterations 100000
./build/benchmark 100000
```

The test checks game payoffs, all 12 decision information sets, convergence near
Kuhn's known first-player value of `-1/18`, and a measured NashConv threshold.
`benchmark` prints measurements and replaces `benchmarks/results.json` with the
current machine's results. Run it from the repository root so the output path is
correct.

## Python binding and API

Create an environment, install the package (which invokes CMake through
scikit-build-core), and start FastAPI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements.txt
python -m pip install ./python
uvicorn api.main:app --reload
```

Python usage:

```python
from poker_solver import Solver

result = Solver(backend="cpu").solve(iterations=100_000)
print(result["strategy"], result["nash_conv"])
```

API calls:

```bash
curl -X POST http://localhost:8000/solve \
  -H 'content-type: application/json' \
  -d '{"iterations":100000,"backend":"cpu"}'
curl http://localhost:8000/benchmark
```

## Solver Lab

With the API running:

```bash
npm install
npm run dev
```

Open <http://localhost:3000/solver>. Set `NEXT_PUBLIC_SOLVER_API` if the API is
not on `http://localhost:8000`. The page shows every information-set strategy,
solve latency, throughput, EV, and NashConv. CPU/CUDA speedup appears only when
the benchmark JSON contains measurements for both.

## CUDA build

CUDA requires an NVIDIA GPU, a compatible driver, and the CUDA Toolkit (`nvcc`).
It cannot be built or benchmarked on Apple Silicon.

```bash
mkdir build-cuda
cd build-cuda
cmake .. -DPOKER_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . -j
cd ..
./build-cuda/poker_solver --backend cuda --iterations 100000
./build-cuda/benchmark 100000
```

The current CUDA implementation contains real batched regret-matching and vector
update kernels, with CUDA events reserved in the backend, but the game traversal
is still CPU-side. Thus `cuda-hybrid` timing is end-to-end measured timing, not a
claim that Kuhn CFR is GPU-accelerated. The next solver milestone is external or
public-chance sampling that forms large independent traversal batches.

## Layout

```text
solver/include/       game, solver, and CUDA interfaces
solver/src/           Kuhn engine, CPU CFR, CLI, benchmark, pybind11
solver/cuda/          conditional CUDA kernels/backend
solver/tests/         convergence and game tests
python/poker_solver/  JSON-friendly native wrapper
python/api/           FastAPI service
app/solver/           minimal Next.js Solver Lab
ml/                   untrained value-network and dataset contracts
benchmarks/           generated measured results
```

## Correctness and metrics

The six ordered private-card deals are traversed each iteration. Regret updates
are weighted by the opponent's reach probability; average-strategy accumulation
is weighted by the acting player's reach. The evaluator computes expected value
and independently enumerates all 64 deterministic policies for each player's six
information sets. `NashConv = BR₀(σ₁) + BR₁(σ₀)` and exploitability is half of that
for this two-player zero-sum game. No metric is hard-coded.

## Highest-value next improvements

1. Batch external-sampling traversals and apply their regret deltas on the GPU.
2. Add Leduc Poker to stress chance rounds, larger trees, and variable actions.
3. Implement CFR+ with linear averaging and benchmark it against vanilla CFR.
4. Add GPU-vs-CPU numerical-equivalence tests and Nsight profiling on NVIDIA CI.
5. Generate depth-limit value targets, train/evaluate the PyTorch network, then
   connect it through the leaf-evaluator seam.
