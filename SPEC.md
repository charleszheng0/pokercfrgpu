# Spec: heads-up postflop solver and trainer (C++)

## Goal
A heads-up No-Limit Hold'em postflop solver that is fast enough to fill a strategy
database, plus an ASCII trainer that reads the database with no compute.

## Scope (v1)
| In | Out (not v1) |
|---|---|
| Heads-up, postflop only (flop, turn or river start) | Preflop solving |
| User input: board, pot, stack, two ranges, bet sizes | Neural networks, NFSP, DeepStack |
| Vector-form DCFR, all hands in one tree pass | Multiway pots |
| Exact exploitability (best response) every N iterations | Sampling (MCCFR) |
| Flat arrays: nodes, regrets, strategy sums | Suit isomorphism, 16-bit compression (v2) |
| CPU threads; CUDA kernel later, benchmarked on Colab | Claims of GPU speed without a benchmark |

## Datapath
spot -> tree builder -> DCFR solver -> exploitability gate -> stored solution -> UI / game.
The gate accepts a solve only when exploitability <= target (default 0.5% of pot).

## Interfaces
- `pokersolve` CLI: flags in, JSON out (strategy, EV, exploitability per checkpoint).
- `pokerlab` local server: `POST /api/solve`, `GET /api/status`, `GET /api/node?path=`.
- `web/`: ASCII, Undertale-like page (black, white mono text, heart cursor, block bars).

## Acceptance tests (done = all pass)
1. Evaluator: all 2,598,960 five-card hands give the known category counts.
2. Range parser: `AA`=6, `AKs`=4, `AKo`=12, `22+`=78 combos; board cards removed.
3. Toy river game (nuts + air vs bluff-catchers, pot-size bet): call freq 50% +/- 5%,
   bluff share of bets 33% +/- 5%, exploitability < 0.1% of pot.
4. River solve, ~1,000 combos per player, 2 sizes: exploitability < 0.5% pot in < 2 s (M2).
5. Turn solve, ~500 combos per player, 2 sizes: exploitability < 0.5% pot in < 60 s (M2).
6. UI: enter a spot, watch progress, click through the tree, see grid + frequencies + EV.

## Limits
- M2, 8 GB RAM: solver refuses trees above a memory cap (default 3 GB) and says why.
- Full-range flop solves need isomorphism + compression (v2).

## Later milestones
v2: suit isomorphism, compression, solution files + SQLite index, spot queue, play mode.
v3: CUDA kernels (regret matching, showdown sweep), Colab benchmark vs CPU, honest chart.
v4: check against an open reference solver on 20 spots.
