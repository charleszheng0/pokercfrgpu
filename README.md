# Research-driven heads-up Hold'em AI

**Status: CPU milestone 1.** This repository contains a new restricted heads-up no-limit Hold'em reference engine, transparent seven-card evaluator, random and heuristic baselines, terminal play, and rule tests. The original Kuhn CFR and Leduc NFSP work now lives in `small_games/`; early CUDA experiments are in `cuda_experiments/`, original weights are in `artifacts/legacy/`, and a standalone PyTorch example is in `examples/`. The original learning results remain unverified. No CUDA work was done in this milestone.

## Run

Python 3.10+ and the standard library suffice for the new engine.

```sh
python3 -m unittest discover -s tests -v
python3 -m poker_cpu.play --opponent heuristic --seed 7
python3 -m experiments.baseline --hands 100 --seed 7 --samples 32
python3 -m small_games.compare_cfr_nfsp --iterations 1500
```

The small-game NFSP code requires PyTorch. Its preserved `.pt` files have unknown training provenance, so they are not counted as validated results. See the [small-game guide](small_games/README.md) for all original entry points.

## Rules and abstraction

The engine uses a 52-card deck, alternate two-card private deals, burn cards, flop/turn/river, a 1/2 small/big blind, button acting first preflop and big blind acting first thereafter, and configurable starting stacks (100 each by default). The legal wager targets per street are minimum raise, approximately half pot, pot, and effective all-in, deduplicated and bounded by the opposing stack. A short all-in raise below the minimum is legal. `Action.amount` is the **target contribution on this street**, rather than an incremental chip amount. All legal actions are explicit objects returned by `legal_actions()`.

Heads-up play has no contested side pot: an unmatched excess wager is returned to its owner. The reference engine runs out the board when no further betting is possible. Rewards are net chips relative to starting stacks. Each observation includes only the acting player's private cards and public information.

The heuristic baseline samples unknown opponent cards and future board cards to estimate showdown equity, then applies simple pot-odds thresholds. It ignores future betting, range conditioning, and fold equity. The random bot samples legal actions uniformly. Neither is a claim of strong play.

## Architecture and research direction

`poker_cpu/cards.py` is the exact five-to-seven-card evaluator. `poker_cpu/engine.py` owns game state and legal transitions. `poker_cpu/agents.py` contains the two CPU baselines. `poker_cpu/play.py` is the initial playable interface. `experiments/baseline.py` records raw baseline conditions and timing. `small_games/` contains the preserved Kuhn/Leduc work; `cuda_experiments/` holds the original CUDA snippets for later audit.

The central experimental question is whether neural continuation values and hardware acceleration can reduce solving cost while preserving strategic quality. The evaluator passed a 5,000-deal independent Treys comparison. The next gates are validated small-game CFR exploitability, reproducible NFSP learning, restricted Hold'em solving, continuation-value error measurement, and CPU profiling. CUDA is deferred until those checks pass; this Apple M2 host has no NVIDIA CUDA device.

See [the roadmap](docs/ROADMAP.md), [architecture](docs/ARCHITECTURE.md), [research notes](docs/RESEARCH_NOTES.md), [experiment log](docs/EXPERIMENT_LOG.md), [failures](docs/FAILURE_ANALYSIS.md), [benchmarks](docs/BENCHMARKS.md), and [interview notes](docs/INTERVIEW_NOTES.md).

## Limits

The first milestone has no trained Hold'em policy, CFR Hold'em search, neural continuation model, browser table, or measured CUDA acceleration. Rule tests and an independent evaluator comparison do not yet amount to exhaustive game-tree verification. No equilibrium claim is made for Hold'em.
