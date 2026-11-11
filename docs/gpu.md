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

## Use for the value network

`pokervnet --gpu` labels value-network training data with this solver. See
[value-net.md](value-net.md), section 6.

## Reproduce

`notebooks/colab_benchmark.ipynb` (speed table) and `scripts/colab_vnet.sh` (value-network rounds).
