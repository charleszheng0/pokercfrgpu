# Research-driven heads-up Hold'em AI

**Status: CPU research prototype.** This repository contains a restricted heads-up no-limit Hold'em reference engine, a seven-card evaluator, random and heuristic baselines, a self-play-trained 20-chip Hold'em NFSP agent, and a local browser table. Kuhn and Leduc have exact small-game best-response measurements. The Leduc NFSP policy remains much more vulnerable than the CFR reference, and the Hold'em NFSP policy has not passed the strength gate. Early CUDA experiments are archived in `cuda_experiments/`; no CUDA optimization has begun.

## Run

Python 3.10+ and the standard library suffice for the Hold'em engine and browser table. Install PyTorch for NFSP, neural-value experiments, and the complete test suite:

```sh
python3 -m pip install -r requirements-research.txt
```

```sh
python3 -m unittest discover -s tests -v
node tests/range_editor_smoke.js
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

For the real Chrome browser check, install the small JavaScript test dependency and run:

```sh
npm ci
npm run test:browser
```

Open `http://127.0.0.1:8765` after starting the server. **Play & Learn** opens by default and supports full hands against heuristic or experimental Hold'em NFSP opponents. It shows cards only when public, the board, pot, stacks, blinds, button, legal actions, and hand history. Users can enter any legal integer-chip bet or raise target. NFSP frequencies, when available, are labeled as experimental policy output. The game explains why solver EV is unavailable at each street. “Study this decision with manual ranges” copies the current public state and your hand into Solver Lab; you must choose ranges that reflect the prior actions, and unsupported action sizes remain explicit.

**ML Training** shows the separate scheduled CPU self-play worker, editable NFSP settings, checkpoint loss curves, fixed-opponent evaluation, model-version comparisons, and the restricted river cache job. The server starts the worker by default; use `--no-training` to run only the table, or Stop/Start in the dashboard. Training runs in another low-priority process with one Torch thread, and new promoted weights load only for a new NFSP hand. `artifacts/offline_training/` holds local checkpoints, candidate history, active pointer, and raw worker log. The average-policy learner reserves a configurable share of best-response observations for held-out imitation loss. Strategy strength is evaluated separately on seeded games. Online NFSP runs use self-play hands rather than supervised epochs. The promotion filter requires at least 100 evaluation hands and a positive lower 95% candidate-minus-incumbent bound against the fixed heuristic. This is a narrow regression gate, not a GTO or general-strength claim.

The Solver Demo can **Sample cached strategy** after its real CFR+ solve. This samples the saved mixed strategy for that exact river board, hand, weighted ranges, and abstraction. A separate scheduled CFR+ job can improve that fixed-position cache if exact restricted NashConv falls below 0.1 chip and improves on its incumbent. No cached solver policy covers arbitrary preflop/flop/turn decisions yet; the Random legal-action baseline remains a distinct research opponent rather than being mislabeled as solved play.

Click **Try Solver Demo** in Play & Learn for a one-decision river practice hand with no setup. Wait for the local CPU solve on first use (later exact-request cache hits are faster), choose any displayed action, and see its computed frequency, conditional EV, EV loss, exact restricted-game NashConv, and the range strategy matrix. Choose another action to compare it immediately without running CFR again; **Back to game** resumes normal play. The demo assumes hero `Jc Tc` or `8h 8s` and opponent `9h 9s` or `Ac Ad`, each combo at equal weight, on `As Kh Qd 7c 2s` with an 8-chip pot and 10-chip effective stacks. These are illustrative supplied ranges, not inferred from a played hand. [Browser screenshot](artifacts/solver_demo.png).

During an ordinary hand, after a supported flop, turn, or river action you can optionally click **Analyze with illustrative ranges** in the post-action review. This replays the saved public decision through the shared CFR+ API and shows frequencies, conditional chip/BB EVs, selected-action loss, NashConv, and runtime. The range is a three-combination example after card removal; it is not inferred from the opponent's behavior. Flop and turn reviews use an explicit current-street cutoff with exact showdown runouts and **no later betting**, so their values are not full-hand EV. River reviews use the restricted river betting tree. Preflop and played wagers outside the abstraction give an explicit unsupported reason. This review is an experimental wiring demonstration, not validated default live advice. For exact function-by-function failures, see [the live solver diagnosis](docs/LIVE_SOLVER_DIAGNOSIS.md).

**Solver Lab** contains manual CPU postflop analysis through the shared `poker_cpu.solve.solve()` API. Its 13×13 hand matrix expands to all 1,326 suit combinations; users can edit class and individual combo frequencies, choose illustrative presets, and see card blockers. The broad preflop presets are unvalidated assumptions and exceed the current eight-combo-per-player lab limit. River decisions, including selected later abstract actions, use the validated finite-game reference. Tiny-range turn decisions traverse river card chance and river betting in the same CFR tree; the interface limits them to four compatible deals and 100 iterations, and every result reports residual NashConv. Flop traversal is disabled in the interface: after CPU caching improvements, one one-deal 20-iteration exact fixture reached 0.102 chips restricted NashConv in 6.26 seconds, which is still too narrow for ordinary advice. Preflop solving is unsupported. Users supply a pot **before the selected street's betting** and ranges that already reflect earlier actions. The lab displays per-hand and range-aggregate strategy, conditional action EVs in chips and BB, selected-action EV loss, exact restricted-game NashConv, iterations, and latency. No automatic live range inference or full-game GTO claim is made. The backend still accepts explicit JSON ranges for developer workflows. Preserved legacy `.pt` files have unknown training provenance and are not counted as validated results. See the [small-game guide](small_games/README.md) for original entry points.

## Rules and abstraction

The engine uses a 52-card deck, alternate two-card private deals, burn cards, flop/turn/river, a 1/2 small/big blind, button acting first preflop and big blind acting first thereafter, and configurable starting stacks (100 each by default). Baseline agents use a small set of minimum, approximately half-pot, pot, and effective all-in targets. `legal_actions()` provides these agent presets; `is_legal_action()` accepts any legal integer-chip bet/raise target, and `betting_limits()` exposes the exact user bet range. A short all-in raise below the minimum is legal but does not reopen raising. `Action.amount` is the **target contribution on this street**, rather than an incremental chip amount.

Heads-up play has no contested side pot: wager targets are capped at the opponent's effective stack, and any unmatched blind/call excess is returned. Short blind stacks are posted correctly; when a blind puts a player all-in, unmatched chips are returned and the board runs out. Ties split the pot, with an odd chip awarded to the first seat left of the button. The engine records blinds, actions, board reveals, refunds, and awards in a public event history; private cards remain hidden from observations and are revealed in the trainer history only after showdown. Rewards are net chips relative to starting stacks. Each hand uses configured starting stacks, and the trainer alternates button positions across hands.

The heuristic baseline samples unknown opponent cards and future board cards to estimate showdown equity, then applies simple pot-odds thresholds. It ignores future betting, range conditioning, and fold equity. The random bot samples legal actions uniformly. Neither is a claim of strong play.

## Architecture and research direction

`poker_cpu/cards.py` is the exact five-to-seven-card evaluator. `poker_cpu/engine.py` owns game state and legal transitions. `poker_cpu/agents.py` contains the two CPU baselines. `poker_cpu/nfsp.py` contains a separate full-hand restricted Hold'em NFSP learner. `poker_cpu/solve.py` defines the typed shared request/result and converts a live game into a public request without reading the opponent's cards. `poker_cpu/multi_street.py` carries postflop betting, card-removal chance, pot/stack commitments, and public action history through remaining streets. It uses the chance-aware CFR/CFR+/DCFR and exact finite-game best-response routines in `poker_cpu/river_solver.py`; the original river game remains a numerical reference. `poker_cpu/one_street_solver.py` remains a separate no-later-betting comparison. `poker_cpu/ranges.py` defines illustrative range presets; `poker_cpu/range_inference.py` applies supplied action likelihoods to public betting histories; `poker_cpu/public_chance.py` computes flop/turn/river reveal probabilities with card removal. The old `poker_cpu/solver_interface.py` research adapter now delegates solving to the shared API. `poker_cpu/play.py` and `trainer/` are terminal and browser interfaces. `experiments/` records raw measurements. `small_games/` contains preserved Kuhn/Leduc code plus a synchronous CFR reference; `cuda_experiments/` holds original CUDA snippets for later audit.

The central experimental question is whether neural continuation values and hardware acceleration can reduce solving cost while preserving strategic quality. The evaluator passed a 5,000-deal independent Treys comparison. Kuhn CFR reached 0.0046 exact NashConv after 10,000 iterations, near the known −1/18 game value. Synchronous Leduc CFR reached 0.14 NashConv after 500 iterations. Leduc NFSP improved reproducibly but remains at 2.54–2.96 NashConv after 20,000 episodes with η=0.3. Three 20-chip Hold'em NFSP policies trained for 10,000 self-play hands each returned +0.86 to +1.48 chips/hand against random on a 1,000-hand fixed suite, but lost −1.04 to −2.08 against the heuristic. On one tiny river position, the neural path was about 40.5× faster than shallow CFR end to end, but its held-out value error remained more than twice as large. CPU profiling shows recursive Leduc tree traversal is the current search bottleneck. CUDA is deferred until the remaining quality gates pass; this Apple M2 host has no NVIDIA CUDA device.

See [the roadmap](docs/ROADMAP.md), [CPU quality gates](docs/QUALITY_GATES.md), [architecture](docs/ARCHITECTURE.md), [research notes](docs/RESEARCH_NOTES.md), [experiment log](docs/EXPERIMENT_LOG.md), [failures](docs/FAILURE_ANALYSIS.md), [benchmarks](docs/BENCHMARKS.md), and [interview notes](docs/INTERVIEW_NOTES.md).

## Limits

The project has a validated tiny-range river reference, a bounded exact-chance turn-to-river CFR tree, and a small river value-model experiment. They use explicit ranges and restricted bets; they are not a full-game strategy or a safe continual re-solver. Exact rank caching and all-in runout integration improved CPU search; chance-sampled CFR remains research-only after worse measured NashConv on fixed flop/turn comparisons. Flop still lacks broad-range quality and interactive latency, and preflop has no solver blueprint. Automatic live strategy feedback remains unavailable because betting-history ranges have not been validated. The trained full-hand NFSP agent is playable, but its fixed-opponent results do not establish equilibrium or strong poker play. There is no measured CUDA acceleration. Rule tests and an independent evaluator comparison do not amount to exhaustive game-tree verification. No equilibrium claim is made for Hold'em.

The neural river experiment currently predicts root hand-conditional EV, not depth-cutoff continuation values. Its 2.07–2.35 chip MAE on 20 held-out boards is worse than direct 20-iteration CFR against the 200-iteration target (0.96 chips), and the tested suit augmentation did not improve MAE. It is not used for live advice. See [the metric definitions and supported claims](docs/PROJECT_METRICS.md), [the benchmark record](docs/BENCHMARKS.md), and [the experiment log](docs/EXPERIMENT_LOG.md).
