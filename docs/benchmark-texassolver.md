# Speed benchmark: pokercfrgpu vs TexasSolver (CPU)

Date: 2026-11-11. Machine: Apple M2, 8 GB, macOS 27.0.1, plugged in, no other solver running.
Both solvers: 8 threads, same spot, same tree, 3 runs each, runs of the two solvers interleaved.

## What is compared

- TexasSolver (bupticybee), branch `console`, commit `02c8021`, built with CMake Release and
  OpenMP (Homebrew `libomp`). It was built in a scratch folder as an outside tool; no TexasSolver
  code is in this repository.
- pokercfrgpu `pokersolve` (CPU solver, `src/game.cpp`), branch `value-net-depth-limit`.
- Trees and ranges are the same as in [reference_check.md](reference_check.md): pot 100,
  stack 950, bets 33% and 75% of the pot, raise 60%, all-in, 4 raises max, all-in threshold
  0.67, OOP 52 hand classes, IP 58 hand classes. Boards: river `Ks 9d 5c 2h 2d`, turn
  `Ks 9d 5c 2h`. Earlier check: the strategies of both solvers agree within 0.4 percentage
  points at 0.05% exploitability.
- Each solver stops when its own exploitability is at or below the target. "Solve time" is the
  time each solver reports for its iterations and exploitability checks (tree build not included).
  "Wall time" is the full process time.

## Results (median of 3 runs; range in brackets)

| Spot, target | Ours: iterations, exploitability, solve time | TexasSolver: iterations, exploitability, solve time | Ours faster by |
|---|---|---|---|
| River, 0.3% | 200, 0.233%, 0.080 s [0.080-0.080] | 201, 0.280%, 0.134 s [0.133-0.134] | 1.7x |
| River, 0.05% | 600, 0.045%, 0.240 s [0.240-0.240] | 711, 0.050%, 0.477 s [0.476-0.479] | 2.0x |
| Turn, 0.5% | 200, 0.466%, 8.96 s [8.58-9.91] | 181, 0.499%, 15.99 s [15.97-17.52] | 1.8x |

Wall time (whole process): river 0.08 s vs 4.9-5.2 s (TexasSolver loads a hand-rank table at
start, about 5 s), turn 9.0 s vs 21.2 s (2.3x).

### Check-frequency control

TexasSolver above checks exploitability every 10 iterations; `pokersolve` checks every 10 up to
100, then every 50. To make sure the extra checks do not explain the gap, TexasSolver was run
again with a check every 50 iterations only (fewer checks than ours), for a fixed 200 iterations:

| Spot, 200 iterations | Ours | TexasSolver, check every 50 | Ours faster by |
|---|---|---|---|
| River | 0.080 s, 0.233% | 0.118 s [0.117-0.120], 0.280% | 1.5x |
| Turn | 8.96 s, 0.466% | 16.46 s [13.66-19.25], 0.422% | 1.8x |

## What this shows

- On these river and turn spots, the CPU solver reaches the same quality target 1.7-2.0x faster
  than TexasSolver on the same laptop (1.5-1.8x per 200 iterations with fewer checks for
  TexasSolver).
- Per iteration, TexasSolver converges a little faster on the turn (0.422% vs 0.466% after
  200 iterations). On the river ours converges a little faster (0.233% vs 0.280%).

## What this does not show

- It is two boards and one tree shape. Flop spots were not compared.
- Each tool measures its own exploitability; the formulas are the same idea (best response,
  % of the pot) but are not the same code.
- It compares against TexasSolver only. TexasSolver is an open-source solver. It says nothing
  about commercial solvers (PioSOLVER, GTO Wizard, and others); no head-to-head run against
  them exists.
- The GPU solver ([gpu.md](gpu.md)) is not part of this comparison. TexasSolver runs on the CPU.

## Reproduce

Input files for TexasSolver: `set_pot 100`, `set_effective_stack 950`, the board, the
two ranges as explicit classes, `set_bet_sizes <player>,<street>,bet,33,75`,
`...,raise,60`, `...,allin`, `oop,<turn|river>,donk,33,75`, `set_allin_threshold 0.67`,
`set_thread_num 8`, `set_accuracy <target>`, `set_print_interval 10` (or 50),
`set_use_isomorphism 1`. Ours:

```bash
./build/pokersolve --board Ks9d5c2h --pot 100 --stack 950 \
  --oop "TT-22,AJs-A2s,KTs+,K9s-K6s,QTs+,Q9s,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo+,QTo+,JTo" \
  --ip "AA-22,A2s+,K7s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,A8o+,KTo+,QTo+,JTo" \
  --bets 33,75 --raises 60 --max_raises 4 --threads 8 --target 0.5 --iters 3000
```
