# Experiment log

## E001 — 2026-10-13 — CPU rule reference

**Problem and hypothesis.** The repository had Kuhn/Leduc experiments but no heads-up Hold'em rules reference. A small, deterministic Python engine could expose betting mistakes before training.

**Change.** Added `poker_cpu` evaluator, state machine, restricted wager targets, observation boundary, and seven unit/property tests. Preserved all pre-existing modified/untracked files. Revision at start: `f0f9fc7feeb704d51bb77bf8763e00ea30d5c0fe`, working tree dirty.

**Method.** `python3 -m unittest discover -s tests -v`, 150 seeded random games with legal-action sampling plus targeted blind, all-in, fold, showdown, hand-rank, and hidden-card checks.

**Observed.** First run: 5/7 tests passed. Two failing assertions incorrectly counted an already awarded terminal pot and expected a bet above an effective stack. After correcting those test expectations: 7/7 passed. The engine implementation did not change for those two failures. A subsequent rules change added burn cards; 7/7 still passed.

**Interpretation.** The tested paths conserve chips and hide opposing cards. This is not exhaustive rules validation. Next: enumerate additional edge cases.

## E002 — 2026-10-13 — baseline agent comparison

**Problem and hypothesis.** A simple equity/pot-odds bot might provide a more meaningful fixed opponent than uniform random action selection.

**Change.** Added Monte Carlo showdown-equity heuristic and `experiments/baseline.py`. Used 32 equity samples per heuristic decision; no model training.

**Method.** 100 hands, seeds 7–106, alternating the heuristic's seat, default 100-chip stacks and 1/2 blinds, random opponent. Raw JSON: `experiments/baseline_seed7.json`. Apple M2, Python 3.10.6, dirty tree at the revision above.

**Observed.** Heuristic mean net reward −2.68 chips/hand, standard error 5.91, 27 wins and 0 ties; wall time 4.75 seconds. The approximate normal 95% interval for mean reward spans roughly −14.3 to +8.9 chips/hand.

**Interpretation.** This small run does not establish an advantage over random. The heuristic is a useful distinct behavior baseline, but not yet a strong bot. Next: larger paired evaluations and diagnosis of future-betting and fold-equity errors before tuning thresholds.

## E003 — 2026-10-13 — independent evaluator cross-check

**Problem and hypothesis.** Five-card fixtures might miss seven-card ranking errors. Comparing independent evaluators on random complete boards and two distinct hole-card pairs should detect many such errors.

**Change.** Added `experiments/crosscheck_evaluator.py`; installed Treys 0.1.8 only into `/tmp/pokercfrgpu-oracle` for validation.

**Method.** 5,000 seeded nine-card samples (five public, two per player), seed 19; compare the sign of our rank difference to Treys. Command: `PYTHONPATH=/tmp/pokercfrgpu-oracle:. python3 -m experiments.crosscheck_evaluator --trials 5000 --seed 19`.

**Observed.** First invocation failed because the new script had an unclosed parenthesis. After the syntax fix: 5,000 comparisons, zero disagreements. Raw result: `experiments/evaluator_crosscheck_seed19.json`.

**Interpretation.** The evaluator agrees with an independent implementation on this sample. This does not prove every possible seven-card rank or the betting engine correct.

## E004 — 2026-10-13 — organize original research files

**Problem and hypothesis.** Original Kuhn/Leduc code, saved models, CUDA snippets, and a tensor exercise were mixed at the repository root. Moving each track to a named directory and updating imports/model paths should improve navigation without changing algorithm behavior.

**Change.** Moved the original Python research files to `small_games/`, CUDA sources to `cuda_experiments/`, four original weights to `artifacts/legacy/`, the tensor exercise to `examples/`, and NFSP checks to `tests/`. Applied Black 24.10.0 to the original Python files. Made Kuhn training and the tensor exercise run only as entry points; added module commands and an NFSP output directory so new weights do not replace legacy weights. No CUDA kernel was changed.

**Method and observation.** `python3 -m unittest discover -s tests -v` passed 11/11 checks, combining seven Hold'em and four historical Leduc/NFSP checks. Importing all small-game modules passed without triggering Kuhn training. A one-iteration `small_games.compare_cfr_nfsp` smoke run loaded both preserved average-policy checkpoints and completed. `small_games.train_nfsp --episodes 4 --seed 42 --output-dir /tmp/pokercfrgpu-nfsp-smoke` wrote four files outside the repository. `python3 -m small_games.kuhn` ran and printed a strategy. Black check and `git diff --check` passed.

**Interpretation.** The original scripts remain runnable under package paths. The smoke runs verify wiring and file locations, not NFSP convergence, CFR quality at one iteration, or CUDA correctness. `cuda_experiments/regretmatch.cu` visibly remains an unfinished draft and was deliberately left unchanged.

## E005 — 2026-10-13 — exact Kuhn CFR reference

**Problem and hypothesis.** The original Kuhn script printed a plausible strategy but had no exact best-response test. A separate synchronous CFR reference should approach the known first-player value −1/18 and reduce exact NashConv.

**Change.** Added `small_games/kuhn_reference.py`, exhaustive pure-policy best responses over information sets, `experiments/kuhn_cfr.py`, and three tests. The original `small_games/kuhn.py` remains intact.

**Method.** `python3 -m experiments.kuhn_cfr`, full six-deal sweep per iteration, checkpoints 100/1,000/10,000, Apple M2, Python 3.10.6, HEAD `23308d7946b52a247b480e15aa7ec0a93723c4f1`, dirty tree. Raw: `experiments/kuhn_cfr.json`.

**Observed.** NashConv 0.05135→0.01454→0.00464; seat-0 EV at 10,000 iterations −0.055546 versus −0.055556 known. Separately, adapting the original script's 12 information-set probabilities to the exact evaluator after 12,342 iterations yielded NashConv 0.00471; this was a one-off diagnostic, not the saved reference experiment.

**Interpretation.** The new CFR reference converges toward the known Kuhn result and validates the exact best-response machinery on this tiny game. It does not validate Hold'em abstraction.

## E006 — 2026-10-13 — synchronous Leduc CFR

**Problem and hypothesis.** The original Leduc CFR changes regrets as it sweeps physical deals. Freezing strategy over each complete sweep should give a clear synchronous reference and decreasing exact vulnerability.

**Change.** Added `SynchronousCFR` while preserving the original `CFR`, plus a convergence script and tests.

**Method.** `python3 -m experiments.leduc_cfr`, 120 physical deals, exact information-set best responses, 20/100/500 iterations. Same host and revision as E005; raw: `experiments/leduc_cfr.json`.

**Observed.** NashConv 1.58485→0.43216→0.13954. The 500-iteration checkpoint and its evaluations took 30.79 seconds cumulatively.

**Interpretation.** This is evidence of convergence in the repository's custom Leduc rules, not a proof of equilibrium or direct comparison to published Leduc configurations.

## E007 — 2026-10-13 — NFSP learning and anticipatory-rate comparison

**Problem and hypothesis.** Training reward and pre-existing checkpoints could not establish NFSP learning. Exact best responses should reveal whether the average policies become less vulnerable. A larger anticipatory rate might provide more best-response examples at a fixed episode budget.

**Method.** `experiments/leduc_nfsp.py` evaluated both average-policy networks at 0, 1,000, 3,000, 10,000, and 20,000 episodes for seeds 11/23/37, with 64-sample minibatches, Adam learning rate 0.001, ε=0.1, and target update every 200 Q updates. First used η=0.1, then changed only η to 0.3. One η=0.1 run continued to 100,000 episodes. Commands and raw snapshots are in `experiments/leduc_nfsp*.json`; newly trained weights are in `artifacts/nfsp_long/`, `artifacts/nfsp_eta03/`, and `artifacts/nfsp_100k/`.

**Observed.** At 3,000 episodes, η=0.1 NashConv rose relative to untrained networks in all three seeds. At 20,000 episodes η=0.1 ended at 3.8052/3.2425/2.9777; η=0.3 ended at 2.5385/2.9604/2.5559. The paired-seed means were 3.3418 and 2.6849. Seed 11 at 100,000 η=0.1 episodes reached 2.6698. A direct greedy-Q check at this checkpoint gave 5.3667 NashConv.

**Interpretation.** NFSP learning is reproducible on this custom small game, and η=0.3 improved this three-seed, 20,000-episode comparison. The policy is still much more exploitable than the CFR reference, and the η result is not enough to establish broad hyperparameter superiority. Next: diagnose exploration, replay scale, optimizer choice, and Q targets with controlled experiments.

## E008 — 2026-10-13 — local browser trainer

**Problem and hypothesis.** Terminal play was functional but hard to use. A local browser table could expose the tested CPU engine and actual experiment results without inventing solver estimates.

**Change.** Added `trainer/server.py`, a responsive table, per-browser sessions, random/heuristic opponent selection, legal controls, hand histories, and JSON-backed results table. The opponent's hole cards appear only after terminal state.

**Method and observation.** Unit tests exercise hidden information, legal play, and result-file provenance. A real HTTP smoke test fetched the page, started a random-opponent hand, submitted a legal action, and received terminal reward `[-1, 1]` with a completed history. The combined suite now has 18 passing tests.

**Interpretation.** This is a playable baseline trainer, not yet a trained Hold'em AI or solver-feedback interface.

## E009 — 2026-10-13 — CPU bottleneck profile

**Problem and hypothesis.** CUDA work requires evidence of where CPU time goes. Recursive traversal might dominate CFR more than network inference.

**Method.** `python3 -m experiments.profile_cpu`, cProfile on 10 Leduc CFR iterations, plus repeated hand evaluation, 32-sample heuristic decisions, and untrained NFSP policy inference. Apple M2, PyTorch 2.10.0, Python 3.10.6, dirty tree. Raw: `experiments/cpu_profile.json`.

**Observed.** Seven-card evaluation 0.189 ms; heuristic action 14.09 ms; network policy inference 0.034 ms; profiled Leduc CFR iteration 110 ms. CFR spent 1.09 of 1.10 profiled seconds in recursive `visit` including children, terminal payoffs, and state hashing/creation.

**Interpretation.** Python tree traversal is the current search bottleneck. Moving only regret matching to a GPU would not address most of this measured CPU time; this is an inference from the profile, not a CUDA benchmark.

## E010 — 2026-10-13 — tiny-range Hold'em river CFR

**Problem and hypothesis.** Full no-limit Hold'em is too large for a first local search reference. A fixed river board, three explicit hands per seat, and one raise should permit exact best responses and expose range/reach mistakes.

**Change.** Added `poker_cpu/river_solver.py` with weighted joint ranges, card-collision removal, a public action tree, synchronous CFR, exact information-set-constrained best responses, and per-hand root values. The local opening action set was checked against `HoldemGame` at the same river pot/stack state. One test initially expected the bettor to keep an uncalled four-chip bet after an opponent fold; correcting the test to the proper net reward made the suite pass.

**Method.** `python3 -m experiments.river_cfr`; board `As Kh Qd 7c 2s`, three hands per player, nine compatible deals, pot 8, effective stack 10, min/half-pot/pot/all-in targets, one raise, 20/100/500 iterations. Raw: `experiments/river_cfr.json`.

**Observed.** Exact local NashConv 2.0366→0.7513→0.3018; 500 iterations plus checkpoint evaluations took 1.83 seconds. The combined suite reached 23 passing tests after the test correction.

**Interpretation.** This is a measured river-only solving prototype. It has no public-belief updates from earlier streets or opponent counterfactual-value safety constraint and does not establish full-game Hold'em quality.

## E011 — 2026-10-13 — first learned river continuation values

**Problem and hypothesis.** A small neural model might approximate deeper river root values faster than a shallower solve. Raw card/range indicators were the first input choice.

**Method.** `python3 -m experiments.river_value`: 60 training and 20 held-out public boards, six hand labels each, weighted three-hand ranges, 200-iteration CPU river CFR labels, 20-iteration shallow comparison, 300 epochs Adam at 0.001, seed 53. Boards do not overlap across train/test. Raw labels and configuration: `experiments/river_value.json`.

**Observed.** Held-out MAE 4.2795 chips and RMSE 5.3516. A zero predictor gave 3.9503 MAE; shallow CFR versus the same reference gave 0.9611. The 200-iteration label strategies averaged 0.3981 NashConv; shallow strategies on test boards averaged 1.9568.

**Interpretation.** The raw-card model failed even the zero baseline. It is not fit for live solver advice. Label strategies are approximate, and value error alone does not measure downstream policy quality.

## E012 — 2026-10-13 — equity feature and dataset-size follow-up

**Hypothesis.** With only 60 training boards, a raw-card network has too little signal about relative river hand strength. Add one exact showdown-equity-against-range feature while preserving the saved boards, labels, network width, optimizer, and epoch count.

**Observed after feature change.** `python3 -m experiments.river_value_equity` reduced held-out MAE to 2.5949 chips on the same 20 boards. This supports the feature hypothesis but remains worse than shallow CFR.

**Second controlled change.** Generated 120 extra training positions with the same 200-iteration solver (`experiments/river_value_extended_dataset.json`), retaining the identical held-out 20 boards. Trained the equity model with seeds 53/54/55 and the same architecture and schedule. MAE was 2.3526/2.2255/2.0667 chips. The seed-53 paired comparison moved 2.5949→2.3526. Raw metrics: `experiments/river_value_equity_180_seed*.json`.

**Latency method and observation.** `python3 -m experiments.river_value_latency` timed model feature construction plus batched prediction for six hand values on one held-out board: 1.734 ms per position. A 20-iteration shallow solve with the same range took 70.295 ms (about 40.5× longer). This was on Apple M2 CPU, without model-loading time.

**Interpretation.** The network reduces cost but does not preserve value accuracy at the shallow solver's level. It is not inserted at a CFR cutoff or exposed as an EV recommendation. More representative range/public-state data and a downstream strategy test are required before integration.

## E013 — 2026-10-13 — NFSP DQN reward scaling

**Observation and hypothesis.** At 20,000 episodes the η=0.3 policies remained 2.54–2.96 NashConv. Terminal Leduc rewards can be much larger than a randomly initialized Q output. Dividing DQN rewards by 20 might make bootstrapping easier without changing action ordering or supervised targets.

**Change and method.** Added optional `reward_scale` to `NFSPAgent` (default 1), plus a terminal-target fit test. Ran `experiments/leduc_nfsp.py` with seeds 11/23/37, η=0.3, 20,000 episodes, reward scale 20; all other settings were unchanged. Raw: `experiments/leduc_nfsp_eta03_scaled20.json`. The complete suite had 24 passing tests.

**Observed.** Final NashConv 2.6916/2.6157/2.1579, mean 2.4884. The matching unscaled run was 2.5385/2.9604/2.5559, mean 2.6849. Seed 11 worsened, while 23 and 37 improved.

**Interpretation.** The small mean reduction is mixed across seeds and does not justify changing the default. The scaled option and raw results remain for future controlled runs.

## E014 — 2026-10-13 — river range-size scaling

**Problem and hypothesis.** A solver that works on nine joint deals may still be too slow when private ranges expand. Measure time, traced memory, and exact best-response cost as the range grows.

**Method.** `python3 -m experiments.river_scale`, seed 71, fixed board, 3/6/10 disjoint hands per seat, 100 CFR iterations. Python `tracemalloc` was active during training. Raw: `experiments/river_scale.json`.

**Observed.** Joint deals 9/36/100; information sets 66/132/220; training 1.47/5.37/13.80 seconds; traced Python peak 0.89/1.52/2.54 MB; exact best response 0.011/0.038/0.112 seconds. The sampled games differ by range size, so their NashConv values are not direct quality comparisons.

**Interpretation.** CPU traversal cost grows rapidly with joint deals even before realistic range sizes. `tracemalloc` slows these runs and excludes non-Python allocations. The measurements strengthen the case for tree/range batching after algorithmic correctness, but do not identify a CUDA kernel yet.

## E015 — 2026-10-13 — first restricted full-hand Hold'em NFSP agent

**Problem and hypothesis.** The small-game NFSP learner could not play actual Hold'em. A separate legal-mask-aware agent might learn a usable average policy through 20-chip self-play, but performance against fixed opponents had to be measured independently of self-play reward.

**Change.** Added `poker_cpu/nfsp.py`: 525 own/private/public/history/action features, 11 action slots, DQN and target network, supervised average-policy network, reservoir sampling, episodic anticipatory mode η=0.3, and reward scaling by the starting stack. `experiments/holdem_nfsp.py` saves both seat policies and Q networks. Tests check hidden-card isolation, legal action mapping, and replay transitions.

**Method.** Trained seeds 11/23/37 for 10,000 self-play hands each, 20-chip stacks and 1/2 blinds, 64-sample updates every four hands, Adam learning rate 0.001, ε=0.1, target copy every 200 Q updates. Each saved seat-0 average policy played 200 seeded hands against random and heuristic agents with alternating seats and buttons. Commands, configuration, model hashes, raw scores, and per-seed training times are in `experiments/holdem_nfsp.json` and `artifacts/holdem_nfsp/`.

**Observed.** Mean chips per hand against random were +1.970/+1.170/+2.135 (standard errors 0.955/0.972/0.982). Against heuristic they were −1.710/−1.765/−2.850 (standard errors 0.997/0.922/0.851). Training times were 38.47/24.30/22.89 seconds. The agent won more than half of heuristic hands in this small sample but lost chips overall, consistent with costly losses on some hands.

**Interpretation.** A self-play-trained policy now runs in the real engine, but 200 hands per opponent and no untrained comparison are inadequate to claim learning or strength. The next experiment uses a larger fixed deal suite and initialized policy controls. The negative heuristic chip returns are a measured weakness, not evidence of closeness to equilibrium.

## E016 — 2026-10-13 — browser NFSP opponent and policy feedback

**Change.** Added a selectable saved NFSP opponent to the local browser table. Its hands use the same 20-chip game as training. A separately saved human-seat average policy reports legal-action probabilities and measured inference latency. The display leaves action EV and EV loss empty because the river-only CFR model is not a validated full-hand adviser.

**Method and observation.** A unit test loaded temporary policy weights, started an NFSP hand, verified private-card isolation and normalized legal-action probabilities, and submitted an action. A direct session smoke test loaded the trained weights and completed a hand. A local HTTP smoke started the server, created an NFSP hand, checked card isolation and policy output, and fetched real dashboard rows. The complete suite passed 27 tests after a probability-sum assertion was relaxed to a 1e-6 tolerance for float32 softmax output.

**Interpretation.** The interface is playable against a trained policy and does not label its probabilities as equilibrium recommendations. It does not yet provide solver EV or NFSP-versus-CFR action comparison.

## E022 — 2026-10-13 — table wager visibility and decision review

**Problem.** The original browser page offered legal action buttons and a completed-hand text history, but did not show current street contributions next to players. It also discarded the NFSP frequency vector once the user acted.

**Change.** Reworked the page into a responsive table and strategy-study layout. Player seats now show stack and current-street bet, the pot and board have a dedicated center, and action buttons distinguish the target contribution from chips added. A live action log records bet/raise targets and actual call amounts. For 20-chip NFSP games, the page shows the saved average policy's legal action frequencies and stores each human decision's frequency snapshot for review. The existing tiny river solver does not provide valid live full-hand EV, so the panel states that EV and EV loss are unavailable.

**Verification.** Added focused session tests for visible street contributions, raise target/amount added, and persisted legal-action frequencies. `python3 -m unittest tests.test_trainer -v` passed all five trainer checks after fixing an index/action mapping error caught by the new frequency review. The frequency model remains labeled experimental; no EV numbers are generated.

**Interpretation.** Wager amounts and model frequencies are now available during play and after decisions. These frequencies report the learned NFSP model's current mix, not what an equilibrium solver says a player should do. A validated full-hand solver is still required before showing per-action EV or EV loss.

## E023 — 2026-10-13 — continuation-value label and generalization audit

**Question.** The earlier value-model report showed 2.07–2.35 chips MAE, while a 20-iteration river CFR solve showed 0.96 chips MAE against the 200-iteration targets. Determine exactly what was compared, verify the labels and payoff mapping, and test a controlled data augmentation change.

**Measurement definition.** Each target is the player's expected chip utility conditional on one of that player's explicit private hands, averaged over compatible opponent hands using normalized products of the supplied range weights, and then averaged over the two players' average CFR policies. It is a *root hand-conditional EV*, not a depth-cutoff continuation value and not an action EV. Labels use the 200-iteration solver's average strategy. For the prior network runs, the MAE/RMSE used 120 hand examples from 20 held-out public boards (six labels per position); these examples are correlated within each board. The three MAEs 2.353, 2.225, and 2.067 came from different network initialization seeds on the same data split, not three independent test sets. Target values were divided by 14 for MSE training and predictions multiplied by 14 for evaluation; 14 is the configured root-pot half (4) plus maximum effective contribution (10). The zero predictor's held-out MAE was 3.950 chips.

The 0.961-chip shallow-solver result is the mean absolute difference between each 20-iteration CFR root hand value and the corresponding 200-iteration label on the original 20 held-out positions. It is a direct computation on the same public state/ranges, not a learned model and not a downstream policy regret comparison. The label reference itself had mean NashConv 0.398 across its original 20 positions, so it is an approximate strategy target.

**Audit and controlled experiment.** Added `experiments/diagnose_river_value.py`. It recomputed the original 200-iteration labels on four saved positions and checked representative terminal payoffs against engine stack deltas under the documented root dead-pot baseline. Maximum errors were 0 chips for labels and 0 chips for 72 payoff cases. The script records the exact range and feature schema, including blocker-filtered joint deals and the added exact showdown-equity feature. Train/test split is by generated public position; no board is shared across the saved split. Each board contributes multiple correlated hand labels. Encoding has board/private-card indicators, the three explicit weighted hands for each range, acting-player ID, pot, stack, and the added opponent-range equity feature; the root history is always empty. The saved target is deterministic under the same finite CFR implementation.

With seed 53, 300 epochs, identical 128-128 MLP, Adam learning rate 0.001, and the same held-out boards, the unaugmented model trained on 1,080 labels in 1.65 seconds and scored 2.134 chips MAE / 2.915 RMSE. Four suit mappings per training board trained on 4,320 labels in 5.02 seconds and scored 2.201 MAE / 2.881 RMSE. This does not support suit augmentation as a fix: MAE worsened slightly while RMSE improved slightly. It also does not establish that the change is statistically meaningful on only 20 public test positions.

For the first three held-out positions, 20/200/500 CFR iteration NashConv was 1.911/0.570/0.344, 2.346/0.597/0.347, and 1.165/0.116/0.047. Their 20-vs-200 EV MAEs were 0.725/0.838/0.806 chips, while 200-vs-500 MAEs were 0.069/0.112/0.059. This shows much better value stability by 200 iterations for these three states, while the 500-iteration strategies still have nonzero NashConv. It is not a full 20-position deeper-reference validation.

**Conclusion and next experiment.** The old values are reproducible for the audited sample, and no payoff or range-weight defect was found. The model's core mismatch is now explicit: it was trained to regress root values, while the planned use is depth-limited continuation values and action comparison. We will not call the model a DeepStack-style continuation network or attach it to live action advice. Next, define a cutoff value target from deeper solves, carry public betting state and coherent ranges in the input, and evaluate action regret against deeper references before revisiting model architectures. A river-only interface remains unsupported until the solver can construct ranges from live history and validate the full decision abstraction.

## E024 — 2026-10-13 — explicit range belief conditioning groundwork

**Motivation.** Extending a fixed-board solver to turn/flop states requires ranges that can be filtered by newly public cards and updated by observed betting actions. The existing `RiverGame` accepted weighted hand lists but did not expose a reusable belief-update operation.

**Change.** Added `WeightedRange` to `poker_cpu/river_solver.py`. It validates unique two-card hands and finite nonnegative weights, normalizes the prior, removes hands blocked by known cards, and multiplies surviving weights by supplied per-hand action likelihoods before renormalizing. An observed action with zero posterior mass raises an error rather than silently reverting to a uniform range. `RiverGame` now uses this representation to validate each input range; its compatible joint-deal normalization remains in place.

**Verification.** Added tests for posterior odds after one blocker and action likelihood, impossible observations, matched-bet showdown payoff, and fold payoff. The complete suite passed 31 tests after this change. Existing river datasets continue to use the same joint probabilities because independently normalizing each seat's marginal weights only changes a common multiplicative factor before joint normalization.

**Limit.** This is validated range-state groundwork, not a turn/flop solver. There is no policy-derived range likelihood adapter in the live game, no chance-card CFR traversal, and no betting after the river chance event. The next solver milestone remains a four-card turn state with explicit river chance and comparisons against a deeper enumerated reference.

## E025 — 2026-10-13 — rerun diagnosis after normalized range refactor

**Reason.** `RiverGame` now accepts validated `WeightedRange` objects and normalizes each player's marginal weights before filtering mutually blocked deals. Although this only applies a common scale factor to each marginal and should leave normalized joint probabilities invariant, rerun the saved diagnostic against the current code.

**Observed.** On the same 180/20 position split, the dataset has 180 unique training boards, 20 unique test boards, zero overlap, 1,080 training hand labels, and 120 test hand labels. Targets range from −4.684 to +13.470 chips. Recomputed saved labels on four positions differed by at most 8.9e-15 chips; 72 payoff checks still had zero error. The first three convergence rows changed only at floating-point noise scale. The raw and augmented model MAEs/RMSEs repeated exactly (2.134/2.915 and 2.201/2.881). Fit times in this rerun were 4.63 and 8.19 seconds, versus 1.65 and 5.02 seconds in the initial run; this variation makes those short fit times unsuitable as a stable performance claim. The rerun is stored at `experiments/river_value_diagnosis.json`.

**Verification and interpretation.** `python3 -m unittest discover -s tests -q` passed 31 tests, and `py_compile` plus `git diff --check` passed. This supports that the range normalization preserved the measured river values and labels. It does not validate turn/flop chance handling or a model-driven action. The current implementation remains a river-only solver with no live range inference.

## E026 — 2026-10-13 — complete playable heads-up Hold'em hand loop

**Problem.** The existing CPU engine and table already supported ordinary blinds, streets, a preset action set, and play against baseline agents, but the engine rejected otherwise legal integer raise sizes, required stacks at least as large as the big blind, asserted on tied odd-chip pots, and the browser offered no arbitrary wager target. Folded cards were also revealed after a hand because the table treated every terminal state as a showdown.

**Change.** Extended the existing engine and reused its evaluator, observations, agents, `TableSession`, and browser table. Preset `legal_actions()` remains the action vocabulary for existing baselines and NFSP; `is_legal_action()` now accepts every legal integer-chip wager target up to effective stack, and `betting_limits()` supplies exact browser bounds. Added short blind posting/refunds, all-in-from-blinds runout, tied odd-chip allocation left of button, burn-card retention for uniqueness checks, and public deal/blind/action/board/refund/award events. The browser now shows button/SB/BB roles, legal preset actions plus a custom target field, full board and pot/stack state, showdown-only opponent-card reveal, action history, and a button that becomes Next hand after completion. Button position alternates between hands. Added a typed solver request/result boundary whose current result explicitly reports unsupported and has no advice values.

**Agent compatibility.** The saved Hold'em NFSP was trained in `poker_cpu/nfsp.py` by calling this same `HoldemGame` with 20-chip stacks, 1/2 blinds, the existing 525-float encoder, the preset `indexed_actions()` mapping, and terminal net-chip rewards scaled by 20. The new normal-stack rules do not change those ordinary transitions. The browser keeps NFSP at 20 chips and marks custom wager choices outside its learned action set. The policy remains experimental. Kuhn/Leduc agents were not integrated.

**Verification.** Added coverage for both button action orders, arbitrary and below-minimum wager targets, short all-in raise non-reopening, short blind refund/runout, odd-chip tie distribution, 52-card uniqueness including burns, public event history, fold-card privacy, custom HTTP actions, alternating next hands, and randomized arbitrary wagers across 75 seeded games in addition to the existing 150 seeded games. `python3 -m unittest discover -s tests -q` passed 43 tests. A local HTTP smoke served the page, started a heuristic hand, accepted a custom raise to 7 chips (outside presets), completed an eight-action five-card-board showdown, and started hand 2 with the button switched to seat 1. The embedded JavaScript passed `node --check`; Python compilation passed.

**Failures and fixes.** Initial assertions incorrectly added the still-displayed terminal pot to stacks after awards and used a 100-chip reward baseline for a manually constructed 10-chip tie; corrected the test accounting/setup. A browser history test initially depended on random AI behavior and was made deterministic with a passive test agent. The first showdown-card privacy check confused the `winner is None` tie marker with a fold; added an explicit `showdown` flag and reran the complete suite. No failure was hidden.

**Limits.** Starting stacks are configured per hand (the trainer resets them for the next practice hand) rather than carried as a cash-game bankroll. Wagers use integer chip units. In heads-up play, effective-stack caps and refunds eliminate contested side pots; a multiway side-pot ledger is not implemented. NFSP may encounter out-of-distribution states after a custom human wager. There is still no full-hand CFR/DeepStack solver, turn/flop search, GTO frequency, action EV, or EV-loss feedback. CUDA remains deferred.

**Final reruns.** After fixing showdown-only card disclosure, `python3 -m unittest discover -s tests -q` passed 43/43 in 4.090 s, 3.852 s, and 3.950 s; an earlier pass took 3.832 s. Python compilation, embedded-JavaScript `node --check`, and `git diff --check` passed. A second local HTTP smoke again accepted custom raise-to-7, completed an eight-action showdown with both opponent cards revealed, and started hand 2 with button 1. Runtime variation is retained in `docs/BENCHMARKS.md` rather than treated as an optimization result.

## E017 — 2026-10-13 — fixed Hold'em NFSP evaluation and untrained controls

**Observation and hypothesis.** The 200-hand evaluation showed positive returns versus random but negative returns versus the heuristic. A larger identical deal suite with each seed's initialized policy as a control could separate some effects of learning from sampling noise. Policy-action RNG was reset per opponent evaluation so saved and initialized policies use the same seeded draw stream; actual action histories still diverge.

**Method.** Ran `python3 -m experiments.evaluate_holdem_nfsp` and `python3 -m experiments.evaluate_initial_holdem_nfsp`: seeds 11/23/37, 1,000 hands per policy and opponent, evaluation seed 900000, alternating seats/buttons, 20-chip stacks, random and 32-sample heuristic opponents. Each initialized network reproduces the first agent's seeded weight initialization before any training. Raw records: `experiments/holdem_nfsp_fixed_eval.json` and `experiments/holdem_nfsp_untrained_eval.json`. Both record Python/PyTorch versions, machine, Git revision, and dirty-tree status.

**Observed.** Trained chip means against random were +1.078/+0.857/+1.477 (standard errors 0.453/0.450/0.449); initialized controls were −0.379/−0.146/+0.073 (standard errors 0.376/0.375/0.381). Against heuristic, trained means were −1.037/−1.750/−2.077 (standard errors 0.435/0.414/0.401); initialized controls were −0.912/−0.924/−0.968 (standard errors 0.402/0.395/0.406). Trained policies won 58.6–58.9% of all heuristic hands but lost chips on average.

**Interpretation.** The consistent random-opponent gain over initialized policies is evidence of task-specific learning in this suite. It did not transfer to the heuristic: all trained policies lost chips, and two were materially worse than their initialized controls by point estimate. Large losses despite frequent wins suggest risk sizing or calling mistakes; that is a hypothesis requiring hand-level diagnostics. Neither win rate nor weak-opponent profit establishes equilibrium proximity.

## E018 — 2026-10-13 — full-hand NFSP CPU inference profile

**Method.** Loaded the seed-11 seat-0 saved average policy, warmed it for 100 actions, then timed 1,000 repetitions each of feature encoding, forward pass, and complete stochastic action selection on one real 20-chip preflop observation. Used one PyTorch CPU thread on Apple M2. Script and raw output: `experiments/profile_holdem_nfsp.py`, `experiments/holdem_nfsp_profile.json`.

**Observed.** Median encoding 0.050 ms, network forward 0.038 ms, complete action 0.129 ms; complete-action p95 0.135 ms. These are isolated inference timings, not server response time or game-search latency.

**Interpretation.** Network inference is cheap compared with the measured recursive CFR traversal and Monte Carlo heuristic decision. This profile does not justify CUDA for NFSP inference on this hardware or establish training throughput bottlenecks.

## E019 — 2026-10-13 — hand-level NFSP loss diagnosis

**Hypothesis.** The trained policy's negative heuristic return might be concentrated in large wager or call decisions despite winning many hands.

**Method.** `python3 -m experiments.diagnose_holdem_nfsp` compared the seed-11 trained average policy with its weight-initialized control on the same 500 new seeded hands against the 32-sample heuristic, alternating seats and buttons. It saved each hand's net chips, own action type/street/target/risk, and all-in flag. Raw: `experiments/holdem_nfsp_diagnostic.json`. This suite uses evaluation seed 901000, separate from E017.

**Observed.** Trained/control chip means were −2.186 ± 0.637 and −1.184 ± 0.590 SE. Both won about 280/500 hands. The trained policy had 191 losses of at least 10 chips versus 148 for control. It folded 14 times versus 80, called 163 times versus 80, and raised 342 times versus 244. Trained hands containing a call averaged −9.21 chips, compared with −7.59 for control. It entered all-in 215 hands versus 286 for control, but its all-in hands averaged −4.14 chips versus −0.66.

**Interpretation.** The loss is associated with more large losses and a much lower fold rate; it is **not** explained by simply taking all-in more often. Conditional call and all-in averages are selection-biased and do not prove those actions caused the losses. The next controlled test should alter faced-bet decisions while keeping deals and the saved NFSP policy fixed, then compare net chips and latency.

## E020 — 2026-10-13 — faced-bet heuristic guard ablation

**Hypothesis.** If the NFSP policy's aggressive faced-bet choices drive its heuristic losses, substituting the existing equity heuristic only when `to_call > 0` might improve chip return.

**Change and method.** `experiments/nfsp_faced_bet_ablation.py` kept the seed-11 saved policy unchanged and compared it with a wrapper that delegated faced-bet decisions to the 32-sample heuristic. Both played the same 500 seeded deals versus the fixed heuristic with alternating seats/buttons; evaluation seed 902000. The script recorded net chips, action counts, and decision wall time. Raw: `experiments/nfsp_faced_bet_ablation.json`.

**Observed.** Saved NFSP returned −0.708 ± 0.636 SE chips/hand; guarded NFSP returned −0.758 ± 0.553. The guard changed 514 decisions, reduced raises from 300 to 77 and increased folds from 9 to 35, but the point-estimate return worsened slightly. Measured mean policy-decision time increased from 1.05 to 5.51 ms, including Python and the heuristic's equity samples.

**Interpretation.** This specific guard did not improve strength on the test suite and imposed a large latency cost; it was not added to the playable agent. The 500-hand difference is much smaller than sampling uncertainty. The result does not rule out better call/fold training or range-aware decisions, but it rejects this simple substitution as an evidenced fix.

## E021 — 2026-10-13 — longer Hold'em NFSP self-play, one seed

**Hypothesis.** The seed-11 10,000-hand NFSP policy might be undertrained; extending the same configuration to 30,000 hands could improve return against a stronger fixed opponent.

**Change and method.** Re-ran `experiments/holdem_nfsp.py` from the same seed 11 with only the training horizon changed to 30,000 hands, saving separately to `artifacts/holdem_nfsp_30k/` and `experiments/holdem_nfsp_30k.json`. The seed's 10,000-hand model remained intact. Evaluated the new seat-0 average policy on the same 1,000-hand seed-900000 random/heuristic suite using `experiments/evaluate_holdem_nfsp.py`; raw fixed result is `experiments/holdem_nfsp_30k_fixed_eval.json`. The model hashes, software, revision, and dirty-tree status are recorded in the training file.

**Observed.** Training took 104.14 seconds. On the fixed suite, the 10,000-hand checkpoint returned +1.078 ± 0.453 SE versus random and −1.037 ± 0.435 versus heuristic. The 30,000-hand checkpoint returned +0.817 ± 0.432 versus random and −1.505 ± 0.426 versus heuristic. Its own 200-hand evaluation returned +2.08 and −1.94, but uses different deals from the 10,000-hand model's original 200-hand check and is not the controlled comparison.

**Interpretation.** Extra self-play did not produce an observed improvement on either fixed opponent; both point estimates moved downward. One training seed and hand-level variance cannot establish that longer training is generally harmful, and the comparison lacks a paired-difference interval. The 10,000-hand checkpoint remains the browser default. The next experiments should target policy quality and range calibration, not simply extend training time.
