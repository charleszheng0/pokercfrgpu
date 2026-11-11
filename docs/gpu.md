# CUDA GPU solver

Measured 2026-11-11 on Google Colab (Tesla T4, 16 GB) against an Apple M2 laptop (8 GB, 8 threads).
One run per spot.

## What it is

The same Discounted CFR algorithm as `src/game.cpp` (same discount parameters, same alternating
updates), rewritten for GPUs:

- One pass down the game tree for reach probabilities, and one pass up for values and regret
  updates. The solver does one tree level at a time.
- One CUDA thread per (node, hand).
- Showdowns use chunked prefix sums over hands sorted by strength, with per-card removal lists.
  Folds use per-card reach sums.
- One source (`gpu/flat_solver_impl.hpp`) builds for CUDA (`flat_solver.cu`) or for CPU threads
  (`flat_solver_cpu.cpp`). The laptop runs the CPU back end, so the tests run without a GPU.

## Correctness

- Test [9]: the GPU-algorithm solver matches the original solver on a turn spot (chance cards,
  suit symmetry) within 0.1% relative exploitability. The first 20 river iterations match to
  3 decimals.
- Test [8]: node locking and loading GPU results back into the solver give the same values.
- All tests pass on Colab and on the laptop.

## Speed

Same spot, same exploitability target. Times are solver iterations only.

| Spot | Game tree | M2 CPU (8 threads) | T4 GPU (CUDA) | Speedup | Exploitability (CPU / GPU) |
|---|---|---|---|---|---|
| River | 87 nodes | 0.217 s | 0.173 s | 1.3x | 0.278% / 0.286% |
| Turn | 36,327 nodes | 26.0 s | 4.79 s | 5.4x | 0.494% / 0.452% |
| Flop, 3-bet pot | 574,770 nodes, 0.5 GB | 39.6 s | 5.01 s | 7.9x | 0.378% / 0.383% |
| Flop, single-raised pot | 969,606 nodes, 3.4 GB | 1,588 s | 138.4 s | 11.5x | 0.452% / 0.431% |

The single-raised flop used 6.4 GB of swap on the 8 GB laptop, so part of that 11.5x is memory,
not compute.

## Profiling

A per-kernel timer showed 89% of GPU time in two kernels that used one thread per node (fold card
sums and showdown prefix sums). Spreading that work over (node, card) and (node, 32-hand chunk)
threads cut a 20-iteration 3-bet flop run from 4.79 s to 1.13 s (4.2x). FP64 sums became FP32
(the T4 runs FP64 at 1/32 rate), and the per-level syncs were removed (one sync per pass).

## Memory

The GPU solver keeps reach and value vectors for every node: 1.8 GB for the 3-bet flop and
10.5 GB for the single-raised flop. Both fit a 16 GB T4.

## Training and solving with CUDA (`--gpu 1`)

On a machine with an NVIDIA GPU and the CUDA toolkit, build the GPU versions:

```bash
cmake -S . -B build -DPOKER_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/pokerbatch_gpu --db data/spots.db --street flop --flops data/flop_list.txt --gpu 1   # training
./build/pokersolve_gpu --board "Ks 9d 5c 2h" --pot 10 --stack 95 --oop "..." --ip "..." --gpu 1
./build/pokerlab_gpu 8080 --gpu 1                                                            # web solver
```

- Each full-tree solve runs on the GPU solver. After the last iteration the result loads back into
  the CPU `Game`, so the quality gate, spot scoring, storage and node views are the same code as
  on the CPU. Stored rows get `backend=cuda` in their `config` column.
- GPU solves use 32-bit storage (16-bit regrets do not load back). If the GPU solver cannot be set
  up for one solve (for example not enough GPU memory), that solve runs on the CPU and the
  program says why.
- In a normal build (no CUDA), `--gpu 1` prints a note and the CPU solver runs as before.
- Shared code: `src/any_solver.hpp`. Test [10] runs this path with the CPU back end of the same
  solver code and checks that the loaded result reads like a CPU solve (exploitability, EVs,
  node views), and that 16-bit storage falls back to the CPU.
- The web solver shows "GPU" next to the iteration count when a solve ran on the GPU.

### Colab check (2026-11-11, Tesla T4, CUDA 13.0, 2 CPU cores)

`notebooks/colab_gpu_option.ipynb` (script `scripts/colab_gpu_option.sh`) built the three `_gpu`
programs and ran each one next to its CPU program. All tests passed.

| Run | CPU program (Colab, 2 cores) | `_gpu` program, `--gpu 1` (T4) |
|---|---|---|
| `pokersolve`, turn Ks9d5c2h, 200 iterations | 35.00 s, 0.482%, EV OOP 39.890 | 1.88 s, 0.473%, EV OOP 39.885 |
| `pokerbatch`, 4 river + 2 turn + 1 flop (3-bet Ac9d8c) | 319 s | 27 s |
| of which: turn 9c2sQs4d / 5d9dTc8h | 90.69 s / 64.29 s | 5.41 s / 3.32 s |
| of which: flop Ac9d8c | 157.94 s, 0.378%, 414 spots | 10.16 s, 0.382%, 414 spots |
| `pokerlab`, one turn solve via the web API | | 1.69 s, 0.407%, `backend: cuda`, node view works |

- Same boards (same seed), all 7 solves passed the gate in both runs, and every solve stored the
  same number of spots (10, 15, 15, 15, 125, 125, 414). GPU rows have `backend=cuda` in `config`.
- Category counts differ a little (for example mixed 211 vs 215): the two runs reach slightly
  different near-equilibrium strategies, so a few spots change category. This is expected.
- The first GPU river solve took 0.35 s instead of about 0.02 s: CUDA starts up on the first solve.
- The CPU side here is Colab's 2-core CPU, not the 8-thread M2 used in the speed table above. Use
  the speed table for speedup claims; these rows show that the `--gpu` path works end to end.

## Use for the value network

`pokervnet --gpu` labels value-network training data with this solver. See
[value-net.md](value-net.md), section 6.

## Reproduce

`notebooks/colab_benchmark.ipynb` (speed table) and `scripts/colab_vnet.sh` (value-network rounds).
