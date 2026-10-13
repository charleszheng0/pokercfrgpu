# Research-driven heads-up Hold'em AI

**Status: CPU research prototype.** This repository contains a restricted heads-up no-limit Hold'em reference engine, a seven-card evaluator, random and heuristic baselines, a self-play-trained 20-chip Hold'em NFSP agent, and a local browser table. Kuhn and Leduc have exact small-game best-response measurements. The Leduc NFSP policy remains much more vulnerable than the CFR reference, and the Hold'em NFSP policy has not passed the strength gate. Early CUDA experiments are archived in `cuda_experiments/`; no CUDA optimization has begun.

## Run

Python 3.10+ and the standard library suffice for the Hold'em engine and browser table. Install PyTorch for NFSP, neural-value experiments, and the complete test suite:

```sh
python3 -m pip install -r requirements-research.txt
```

```sh
python3 -m unittest discover -s tests -v
python3 -m poker_cpu.play --opponent heuristic --seed 7
python3 -m trainer.server --port 8765
python3 -m experiments.baseline --hands 100 --seed 7 --samples 32
python3 -m experiments.kuhn_cfr
python3 -m experiments.leduc_cfr
python3 -m experiments.leduc_nfsp --seeds 11 23 37 --episodes 3000 --checkpoints 0 1000 3000
python3 -m experiments.river_cfr
python3 -m experiments.river_value
python3 -m experiments.holdem_nfsp --seeds 11 23 37 --hands 10000 --eval-hands 200
python3 -m experiments.evaluate_holdem_nfsp
python3 -m experiments.evaluate_initial_holdem_nfsp
```

Open `http://127.0.0.1:8765` after starting the server. The playable table supports full hands through showdown against the heuristic, random, or experimental Hold'em NFSP opponent. It shows each player's cards only when public, the board, pot, stacks, blinds, dealer/button position, legal actions, and complete betting history. Preset sizes are available, and the target input accepts any legal integer-chip bet or raise through the effective stack. The Next hand control alternates the button. NFSP requires PyTorch and is an experimental 20-chip policy trained on this engine's restricted preset action set; custom human wagers can put its observation outside that training distribution. When NFSP frequencies are shown, they are explicitly the saved policy's probabilities, not GTO frequencies. Per-action EV, EV loss, and DeepStack-style search remain unsupported. Preserved legacy `.pt` files have unknown training provenance and are not counted as validated results. See the [small-game guide](small_games/README.md) for original entry points.

## Rules and abstraction

The engine uses a 52-card deck, alternate two-card private deals, burn cards, flop/turn/river, a 1/2 small/big blind, button acting first preflop and big blind acting first thereafter, and configurable starting stacks (100 each by default). Baseline agents use a small set of minimum, approximately half-pot, pot, and effective all-in targets. `legal_actions()` provides these agent presets; `is_legal_action()` accepts any legal integer-chip bet/raise target, and `betting_limits()` exposes the exact user bet range. A short all-in raise below the minimum is legal but does not reopen raising. `Action.amount` is the **target contribution on this street**, rather than an incremental chip amount.

Heads-up play has no contested side pot: wager targets are capped at the opponent's effective stack, and any unmatched blind/call excess is returned. Short blind stacks are posted correctly; when a blind puts a player all-in, unmatched chips are returned and the board runs out. Ties split the pot, with an odd chip awarded to the first seat left of the button. The engine records blinds, actions, board reveals, refunds, and awards in a public event history; private cards remain hidden from observations and are revealed in the trainer history only after showdown. Rewards are net chips relative to starting stacks. Each hand uses configured starting stacks, and the trainer alternates button positions across hands.

The heuristic baseline samples unknown opponent cards and future board cards to estimate showdown equity, then applies simple pot-odds thresholds. It ignores future betting, range conditioning, and fold equity. The random bot samples legal actions uniformly. Neither is a claim of strong play.

## Architecture and research direction

`poker_cpu/cards.py` is the exact five-to-seven-card evaluator. `poker_cpu/engine.py` owns game state and legal transitions. `poker_cpu/agents.py` contains the two CPU baselines. `poker_cpu/nfsp.py` contains a separate full-hand restricted Hold'em NFSP learner. `poker_cpu/play.py` and `trainer/` are terminal and browser interfaces. `experiments/` records raw measurements. `small_games/` contains preserved Kuhn/Leduc code plus a synchronous CFR reference; `cuda_experiments/` holds original CUDA snippets for later audit.

The central experimental question is whether neural continuation values and hardware acceleration can reduce solving cost while preserving strategic quality. The evaluator passed a 5,000-deal independent Treys comparison. Kuhn CFR reached 0.0046 exact NashConv after 10,000 iterations, near the known −1/18 game value. Synchronous Leduc CFR reached 0.14 NashConv after 500 iterations. Leduc NFSP improved reproducibly but remains at 2.54–2.96 NashConv after 20,000 episodes with η=0.3. Three 20-chip Hold'em NFSP policies trained for 10,000 self-play hands each returned +0.86 to +1.48 chips/hand against random on a 1,000-hand fixed suite, but lost −1.04 to −2.08 against the heuristic. On one tiny river position, the neural path was about 40.5× faster than shallow CFR end to end, but its held-out value error remained more than twice as large. CPU profiling shows recursive Leduc tree traversal is the current search bottleneck. CUDA is deferred until the remaining quality gates pass; this Apple M2 host has no NVIDIA CUDA device.

See [the roadmap](docs/ROADMAP.md), [CPU quality gates](docs/QUALITY_GATES.md), [architecture](docs/ARCHITECTURE.md), [research notes](docs/RESEARCH_NOTES.md), [experiment log](docs/EXPERIMENT_LOG.md), [failures](docs/FAILURE_ANALYSIS.md), [benchmarks](docs/BENCHMARKS.md), and [interview notes](docs/INTERVIEW_NOTES.md).

## Limits

The project has a tiny river-only Hold'em CFR research reference and a small river value-model experiment. These use three private hands per seat, a fixed effective stack, and at most one raise; they are not a full-game strategy or a safe continual re-solver. The trained full-hand NFSP agent is playable, but its fixed-opponent results do not establish equilibrium or strong poker play. There is no measured CUDA acceleration. Rule tests and an independent evaluator comparison do not amount to exhaustive game-tree verification. No equilibrium claim is made for Hold'em.

The neural river experiment currently predicts root hand-conditional EV, not depth-cutoff continuation values. Its 2.07–2.35 chip MAE on 20 held-out boards is worse than direct 20-iteration CFR against the 200-iteration target (0.96 chips), and the tested suit augmentation did not improve MAE. It is not used for live advice. See [the metric definitions and supported claims](docs/PROJECT_METRICS.md), [the benchmark record](docs/BENCHMARKS.md), and [the experiment log](docs/EXPERIMENT_LOG.md).
