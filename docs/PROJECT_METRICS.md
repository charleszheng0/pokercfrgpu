# Project metrics that matter

This file defines the small set of measurements that can support credible
engineering or research claims about this project. It deliberately separates
game-playing strength, solver quality, approximation quality, and systems speed.
No metric in this file by itself establishes that a Hold'em strategy is GTO.

## Current measured results

| Area | Metric and conditions | Observed result | What it supports |
|---|---|---:|---|
| Engine | 5,000 seeded evaluator comparisons against Treys 0.1.8 | 0 disagreements | Cross-implementation evidence for hand ranking; not exhaustive rule verification |
| Small-game CFR | Kuhn exact NashConv at 10,000 iterations | 0.0046 | The reference CFR implementation converges on Kuhn under the tested setup |
| Restricted river CFR | Mean NashConv over the saved 20-position/200-iteration labels | 0.3981 | The average strategy is still exploitable in the tiny abstract game |
| River labels | Recomputed labels on 4 saved positions; maximum absolute difference | <9e-15 chips | Deterministic regeneration for this sample, not an independent solver validation |
| River payoff accounting | CFR payoff vs engine stack delta under the documented dead-pot baseline, 72 cases | 0 chips maximum difference | Tested zero-sum payoff mapping for those representative states |
| River value model | Three initializations, same 180 training and 20 held-out boards | MAE 2.067–2.353 chips; RMSE 2.737–3.043 | Model-seed sensitivity on one small held-out set; each board yields six correlated hand examples |
| Shallow river CFR | 20-iteration conditional EVs vs 200-iteration label on the original held-out set | 0.961 chips MAE | A direct shallow solve approximates these labels better than the neural model; does not show a better strategy at equilibrium |
| River deeper reference | First three held-out positions, 200 vs 500 iterations | EV MAE 0.080 chips; NashConv mean 0.246 at 500 vs 0.428 at 200 | Evidence that 200-iteration values stabilize on these three positions; small sample only |
| Data augmentation | Four suit mappings per training board, same seed/architecture/optimizer/test boards | MAE 2.134 raw vs 2.201 augmented; RMSE 2.915 vs 2.881 | Suit augmentation did not reduce MAE in this controlled run |
| Hold'em NFSP | 1,000 fixed hands/seed vs random, three seeds | +0.86 to +1.48 chips/hand | Task-specific performance against random in that restricted setup |
| Hold'em NFSP | Same evaluation vs the 32-sample heuristic | −1.04 to −2.08 chips/hand | The tested NFSP policies lose chips against this fixed heuristic |
| Playable engine | Complete Python test suite, including 150 seeded and 75 arbitrary-wager randomized games | 43/43 tests passed | Tested game rules and API functionality; does not establish exhaustive rule correctness |
| Browser game loop | Heuristic hand, custom raise, showdown, then next hand | 8 actions, 5-card board, button alternated | End-to-end playability on one real local HTTP session |

Raw solver and model measurements are in `experiments/river_value_diagnosis.json`,
`experiments/river_value_equity_180_seed5*.json`, and the original experiment
records. NFSP evaluation conditions and confidence intervals are in
`experiments/holdem_nfsp_fixed_eval.json`.

## Metrics to report for future work

For a solver strategy, report exact best-response NashConv (or a clearly named
approximation), exploitability units, iteration count, action abstraction,
range definition, public-state sample, and CPU time. Compare multiple seeds or
positions and include a spread or confidence interval. CFR iteration count by
itself is not a quality result.

For a value approximation, keep separate held-out public states from training,
then report chip MAE/RMSE, signed bias, error percentiles, and errors grouped by
board texture, range overlap, pot, and stack. Also report strategic regret:
the EV lost when actions chosen from predicted values are evaluated against a
deeper reference. Prediction loss alone is insufficient. Record the reference
solver's residual NashConv so label uncertainty is visible.

The first continuation-value release gate is set relative to the tested 2-chip
big blind: held-out value MAE at most 0.25 BB (0.5 chips), RMSE at most 0.5 BB
(1 chip), mean action regret at most 0.125 BB (0.25 chips), and 95th-percentile
action regret at most 0.5 BB (1 chip), all compared with a deeper CPU reference
on disjoint public states. Action regret is the primary strategic criterion;
the value-error limits cap the scale of approximation error at a fraction of
one blind. Also require end-to-end latency below the shallow CPU solve on the
same state distribution. These are prospective engineering gates, not
achieved results, and should be rescaled in BB if the abstraction stakes change.

For game-playing strength, report net chips per hand with standard error or a
confidence interval against named fixed opponents, on fixed evaluation deals,
with seat/button rotation, stack/blind structure, and action abstraction. Keep
win rate separate from chip return. Never translate a win-rate change into
equilibrium proximity.

For systems performance, report end-to-end decision latency (median and p95),
solver iterations or evaluated states per second, memory use, and workload
dimensions. Separate model-only inference from feature construction and search.
For CPU/GPU comparisons, include transfer and launch costs, numerical error,
and the same strategy-quality metrics.

For reproduction, preserve the command, seed, software versions, hardware,
revision, dirty-tree state, raw output, and the exact evaluation dataset. Treat
multiple labels from one public state as clustered observations rather than
independent samples.

## Resume-ready claims currently supported

These are technical facts, not polished impact claims:

- Built and tested a CPU heads-up Hold'em engine and independently cross-checked
  its hand evaluator on 5,000 seeded hands with zero disagreements.
- Implemented a playable local heads-up table with arbitrary legal integer-chip
  wager targets and verified a full heuristic hand plus alternating next hand
  through the HTTP API.
- Implemented synchronous CFR references that reached Kuhn NashConv 0.0046 at
  10,000 iterations and a distinct tiny-range Hold'em river CFR prototype.
- Profiled a restricted river value-network prototype against direct shallow
  solving; the network was faster in the measured micro-workload but had
  materially higher held-out EV error and failed the accuracy gate.
- Trained restricted Hold'em NFSP policies that gained against random opponents
  in the fixed suite but lost chips against the tested heuristic.

Do not claim full-game DeepStack, safe re-solving, GTO advice, turn/flop search,
or CUDA acceleration from the current evidence.
