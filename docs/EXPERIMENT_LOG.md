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

## E027 — 2026-10-13 — explicit-range river CFR analysis and update comparison

**Question.** Can the existing tiny river prototype return reproducible mixed strategies and ordinary conditional action EVs, distinguish those EVs from counterfactual regret values, and identify a practical CPU update rule on its fixed restricted game?

**Changes.** Added vanilla CFR, a synchronous CFR+ variant (positive regret clipping and linear strategy averaging), and DCFR with α=1.5, β=0, γ=2. Added root-seat selection and analysis for either seat. `analyze_root()` now returns per-hand and range-averaged root frequencies, per-action conditional chip EVs, exact finite-game NashConv, and a chosen-action EV gap only when the acting hand is supplied. Conditional EVs condition on a player's hand and compatible opponent deals, force the root action, and follow the average continuation strategy. Added `TinyRiverCFRBackend`: it accepts explicit ranges and the exact supported river root action set, while unsupported boards, missing ranges, prior river histories, and mismatched action sets return no advice values. It reports end-to-end latency separately from search-only latency; the requested budget limits search only. The browser remains without analysis because it has no defensible live range inference.

**Verification.** Added tests for the action-value definition against an independently written brute-force traversal over all compatible deals and terminal histories for both root seats; repeatability, strategy normalization, nonnegative CFR+ regrets, stack below the blind, and adapter refusal/acceptance paths are also covered. The full suite passed 50 tests. `py_compile` and `git diff --check` passed. An initial test run caught a missing joint-deal weight name in the report; the next caught a test fixture passing `WeightedRange` where `RiverGame` expects hands plus weights. Both were corrected before the successful full run.

**Controlled result.** `python3 -m experiments.river_variants --checkpoints 20 100 500 1000 2000` solved the same board, ranges, 8-chip pot, 10-chip effective stack, and one-raise min/half-pot/pot/all-in tree. The nine compatible joint deals and action set were fixed across variants. Exact best-response NashConv (chips) at 100/500/1,000/2,000 iterations was vanilla CFR 0.7513/0.3018/0.2032/0.1385, CFR+ 0.3336/0.0827/0.0459/0.0184, and corrected DCFR 0.4505/0.1036/0.0509/0.0200. At 2,000, measured training wall time was 6.44/6.61/7.37 seconds respectively; exact NashConv/action report evaluation added about 0.016–0.017 seconds at that checkpoint. These are single-run Apple M2 measurements with a dirty working tree; timing varied across runs and does not support a general speed claim. CFR+ and DCFR substantially reduced exploitability in this one game; neither is certified for broader Hold'em positions. Raw data and revision are in `experiments/river_variants.json`.

**Interpretation and limits.** The first milestone now returns deterministic legal mixed strategies and conditional action EVs on the supplied finite river game, and reports exact best-response NashConv within that same abstraction. At 2,000 iterations residual NashConv is nonzero, so the strategy is still exploitable. The best-response calculation is an exact diagnostic for the abstract game, not an independent equilibrium oracle. There is no broad position suite, sampled-range generalization study, live range inference, prior-action river node reconstruction, turn/flop chance transition, neural continuation evaluation, or downstream UI feedback. `chosen_action_ev_loss` is a local one-step action gap under the average continuation policy, not the full best-response exploitability of the human's overall strategy.

## E028 — 2026-10-13 — five-position weighted-range river validation

**Hypothesis.** The 2,000-iteration variant result could be particular to one hand list. A deterministic set of distinct boards, ranges, weights, pots, and stacks should expose failures in card handling or regret convergence beyond that fixture.

**Method.** `python3 -m experiments.river_range_suite` uses seed 729 to generate five positions. Each position has a five-card board, three explicit hands per seat, positive nonuniform range weights, nine compatible joint deals, and one of 6/8/12-chip pots and 6/10/14-chip effective stacks. CFR+ trains for 500 iterations, then exact best response computes abstract-game NashConv. Raw positions, ranges, weights, timing, software, revision, and dirty-tree state are stored in `experiments/river_range_suite.json`.

**Observed.** NashConv was 0.00559, 0.10842, 0.10353, 0.20381, and 0.04657 chips. At 2 chips per big blind, these correspond to 0.0028–0.1019 BB. Training took 0.78–2.11 seconds and exact reports took 4.9–12.6 ms in this run. All games completed with nine compatible deals and no invalid range or terminal payoff errors.

**Interpretation.** The backend handles these five weighted, blocker-free explicit range pairs and produces exact best-response diagnostics for each restricted game. Convergence quality varies materially by position at the fixed 500-iteration budget, supporting the use of per-position convergence reporting rather than a universal iteration claim. This small generated suite does not cover range collisions in every case, arbitrary betting histories, broader hand distributions, or population-level accuracy; collision filtering and posterior conditioning have separate unit tests. It remains a local river solver, not live GTO feedback.

## E029 — 2026-10-13 — DCFR update-order correction and rerun

**Review finding.** A code review after the initial CFR-variant comparison found that the prototype discounted DCFR regrets after constructing the current strategy. That means the policy update used the undiscounted regret vector and did not match the intended discounted-regret recurrence. The initial DCFR figures are superseded.

**Fix.** Apply the positive/negative regret discounts and prior average-strategy discount at the start of the iteration, before regret matching and current strategy accumulation. New counterfactual regret increments are added after traversal. CFR and CFR+ behavior are unchanged. The DCFR parameters remain α=1.5, β=0, γ=2.

**Verification.** Re-ran the full suite (50/50 passed), then reran all three variants at 20/100/500/1,000/2,000 checkpoints. On the fixed river game, corrected DCFR NashConv was 1.7983/0.4505/0.1036/0.0509/0.0200 chips. At 2,000 iterations it took 7.37 seconds of training in the recorded run, plus 16.8 ms for exact analysis. Vanilla CFR and CFR+ results were unchanged to the saved precision. The updated raw file overwrites the preliminary DCFR record; this entry preserves why the prior measurement was not accepted.

**Adapter timing.** The API now reports `search_latency_ms` for CFR training and `latency_ms` for end-to-end analysis including exact NashConv. The computation budget applies only to search; exact diagnostic calculation and the final atomic iteration may run beyond it. The unsupported/full-table path remains unchanged.

## E030 — 2026-10-13 — action-EV stability against deeper river solves

**Change.** Extended `experiments/river_variants.py` to compare each checkpoint's range-averaged root action EVs with the 2,000-iteration values for the same variant and game. These values use each checkpoint's own average continuation policy, so the comparison measures the stability of policy-dependent action values as solving deepens; it is not a fixed-policy value-estimation error.

**Observed.** At 500 iterations, mean/max absolute root action EV differences from the 2,000-iteration checkpoint were 0.0245/0.0951 chips for CFR, 0.0078/0.0146 for CFR+, and 0.0084/0.0187 for DCFR. At 1,000 iterations, the corresponding mean/max errors were 0.0156/0.0324, 0.0041/0.0065, and 0.0046/0.0103 chips. The final 2,000-iteration rerun measured 7.52/7.81/8.88 seconds of CFR/CFR+/DCFR training and 11.3–14.0 ms for exact analysis. NashConv is unchanged from E029 to reported precision. The command, revisions, raw root action EVs and timings are in the updated `experiments/river_variants.json`.

**Interpretation.** Conditional action values on this one game become close to the 2,000-iteration profile by 500–1,000 iterations, especially for CFR+ and DCFR. This does not establish action accuracy across other boards, nor does it validate neural cutoffs or live range inference. The five-board suite's per-position NashConv remains the broader current check.

## E031 — 2026-10-13 — short-budget river adapter latency and convergence

**Method.** Ran `python3 -m experiments.river_adapter_budget` with a 10 ms search-only budget on the fixed 8-chip-pot/10-chip-stack river board. Both seats had three weighted explicit hands, the acting hand was JcTc, and the requested action was check. The raw API response and configuration are in `experiments/river_adapter_budget.json`.

**Observed.** Three CFR+ iterations took 10.23 ms of search; the end-to-end response took 20.11 ms because exact NashConv and conditional action values were computed afterward. NashConv was 5.432 chips, and the acting-hand check EV gap was 1.255 chips. This is far from a useful convergence result, despite the successful API response.

**Interpretation and action.** The latency budget is currently soft and does not guarantee solver quality. Search alone slightly exceeded 10 ms due to an atomic iteration, and the exact convergence report doubled total runtime. The API includes this residual exploitability value so consumers can see the failure. The browser remains disconnected and displays no solver metrics. A future live integration needs an explicit convergence/quality gate, validated on more positions, and a strategy for amortizing or reducing exact best-response computation.

## E032 — 2026-10-13 — explicit-range River Analysis mode

**Observation.** The table's study pane could show experimental NFSP action probabilities, but the validated river solver was not reachable from the app. Automatic range inference was still unsupported, so wiring it into ordinary gameplay would have implied unsupported advice.

**Change.** Added a separate River analysis mode beside Play Hold'em. The form accepts a complete board, hero hand, pot, both stacks, two editable weighted-hand JSON ranges, and a configurable 20–5,000 iteration count (default 500). The API validates the request and calls the existing CFR+ river game. It returns hero-hand root frequencies and conditional action EVs, exact restricted-tree NashConv, iteration count, latency, and half-budget convergence comparison. The selected-action EV loss is computed from the displayed hero-hand action EVs and documented as holding the final average continuation profile fixed. Inputs outside the supported first-to-act river state return an explicit error. No betting-history range inference or GTO label was added. The existing table and NFSP remain under the Play mode. Range normalization now rejects a sum that overflows even if each supplied weight is individually finite.

**Measured result.** On the saved sample with board As Kh Qd 7c 2s, hero Jc Tc, 20-chip pot, 100-chip stacks, and three weighted combos per seat, 500 CFR+ iterations took 2,722.8 ms end to end on Apple arm64. Exact restricted-game NashConv decreased from 0.82848 chips at 250 iterations to 0.48939 at 500. The final hero-hand strategy was check 0.14%, bet 2 0.10%, bet 10 1.55%, bet 20 10.84%, and bet 100 87.38%. Conditional action EVs were 13.680, 14.265, 19.048, 24.636, and 26.802 chips respectively; EV loss versus the best root action was 13.121, 12.536, 7.754, 2.166, and 0 chips. These outputs demonstrate the UI/API path and solver calculations on this fixture; the residual NashConv remains material, and one case does not establish general convergence or GTO quality. The raw payload and response are saved in `experiments/river_analysis_ui_baseline.json`.

**Verification.** Added report and unsupported-input tests. Test-only assertion mismatches occurred for capitalization, supported-state wording, and then the more specific zero-compatible-hand message after validation was moved ahead of training; each expectation was corrected. A code review also found that finite range weights could sum to infinity; the solver now rejects non-finite normalization totals, with a regression test. The final full suite passed 53 tests, and the HTTP request, JavaScript syntax, Python compilation, and diff checks passed. Details are in `docs/BENCHMARKS.md`. The UI limit of eight combinations per seat bounds work. Three invocations of the same 500-iteration fixture ranged from 2,723 ms to 3,249 ms. This is a usability measurement, not a latency guarantee.

## E033 — 2026-10-13 — Solver Lab hand matrix, public ranges, and later river states

**Observation.** Raw JSON range entry made the standalone river analyzer unsuitable for ordinary users. The existing CFR+ solver could start only at its first river action in the UI, and the game had no explicit per-street explanation for missing live feedback. A range posterior could be mathematically updated from public history, but there was no source of validated action likelihoods.

**Hypothesis and change.** A 13×13 class matrix with per-suit combo editing could preserve exact backend range inputs while removing JSON from the interface. The river game already stores public history and paid chips; replaying supported abstract actions into a `RiverState` should allow CFR+ to solve later nonterminal decisions without changing payoff semantics. Added Play & Learn as the default and Solver Lab as the advanced mode, illustrative preset catalog, matrix editor with card-blocker display, a public-event Bayesian range-update module requiring supplied action likelihoods, and a public-card chance-transition reference for flop/turn/river. The lab now offers hero-first, opponent-check-to-hero, facing-bet, and facing-raise river cases. The live game remains gated because no action-likelihood provider has been fitted or validated and there is no earlier-street CFR traversal.

**Measured results.** `python3 -m experiments.solver_lab_milestone` saved the exact request, response, hardware, revision, and dirty-tree status in `experiments/solver_lab_milestone_v2.json`. On the same three-combo-per-seat illustrative ranges, 500-iteration first-action CFR+ had NashConv 0.489393 chips and 2,689.5 ms end-to-end latency. After hero check/opponent bet-to-10, 500 iterations had NashConv 0.850238 and 416.9 ms; 2,000 iterations reduced that position's NashConv to 0.111001 in 1,626.1 ms. After an opponent check with the opponent first to act, 500 iterations had NashConv 0.452920 in 1,421.9 ms. These post-action runs deliberately reused the unconditioned manual ranges, so they validate root-state computation and convergence diagnostics, not realistic advice after those actions. Exact weighted turn-card probabilities across 49 possible card codes summed to 1.0000000000000004; that complete enumeration took 3.049 ms on the recorded Apple arm64 run. Card-removal tests also cover uniform 1/C(48,3) flop, 1/45 turn, and 1/44 river probabilities for fixed private hands.

**Verification and limits.** The Python suite passed 64 tests. The dependency-free Node smoke checked 169 matrix cells, all 1,326 distinct two-card combos, default range serialization, public-card removal, and preservation of opponent combos that only conflict with the observed hero hand for other hypothetical hero hands. A local HTTP smoke loaded both modes and the matrix script/preset catalog, started a random-opponent hand, and received CFR+ reports for first and later river action states. Its first version failed a test-only assertion that expected the JavaScript file to contain its own filename; the corrected assertion checked the exported editor symbol and the smoke passed. Python compilation, both JavaScript syntax checks, and diff whitespace checks passed. No browser visual automation or calibrated human-range evaluation was performed.

**Next experiment.** Fit an action-likelihood model on separately collected public histories with known training hands, then evaluate held-out log loss/calibration and posterior sensitivity to action size, seat, and street. Validate turn CFR traversal against enumerated river continuations before exposing turn/flop frequencies. Automatic river advice should require both a range-model gate and a per-position solver-quality gate; the observed NashConv spread shows that an iteration count alone is insufficient.

## E034 — 2026-10-13 — exact-runout one-street flop/turn CFR reference

**Problem and hypothesis.** The chance-transition arithmetic in E033 did not yield flop or turn betting values. A small reference game can reuse the river betting tree and CFR+ update if each compatible private deal receives an exactly enumerated showdown distribution. This tests card removal and chip payoffs before implementing future-street betting.

**Change and method.** Added `OneStreetGame` for a three- or four-card public board. Betting uses the existing restricted action tree on the current street; after that street, all remaining public cards are dealt uniformly and there is no further betting. For each private deal, the implementation enumerates all 990 flop or 44 turn runouts and scores the resulting seven-card hands. `python3 -m experiments.one_street_baseline` ran one deterministic position on each street, with two private combos per player, four compatible deals, an 8-chip pot, 10-chip effective stack, and 200 synchronous CFR+ iterations. The raw configuration, frequencies, conditional EVs, exact finite-game NashConv, timing, revision, and dirty-tree status are in `experiments/one_street_baseline.json`.

**Observed.** The final saved rerun took 68.66 ms for turn equity precomputation, 337.36 ms for search, and 5.12 ms for exact root analysis; NashConv was 0.08849 chips. The flop run took 1,535.41 ms for equity precomputation, 339.60 ms for search, and 4.73 ms for analysis; NashConv was 0.06864 chips. An earlier same-configuration run measured 68.42/340.64/5.12 ms on turn and 1,531.79/337.57/4.76 ms on flop; the script's output file was overwritten by the final rerun, so those earlier timings are anecdotal rather than preserved raw benchmark rows. Tests verified exact turn payoff against the average of 44 independently scored river showdowns, flop/turn probability normalization, legal betting actions, and a noninitial turn decision after an opponent bet. Another test replayed an actual Hold'em engine event history through the range-inference arithmetic; it did not validate the assumed action likelihoods. The complete Python suite passed 69 tests.

**Interpretation and next step.** These measurements are for a one-street game, so its frequencies are not full flop/turn Hold'em recommendations. The roughly 22× higher flop equity-precomputation time on this fixture reflects the 990-versus-44 runout count; one run is insufficient to characterize latency more generally. The next controlled reference should add turn decisions following a flop action and compare root action values with deeper exact continuation solving on tiny ranges. Range priors must also be conditioned on public actions before the resulting advice could be attached to gameplay. Preflop traversal and neural continuation values remain unsupported.

## E035 — 2026-10-13 — shared exact-chance postflop CFR architecture

**Observation and hypothesis.** The playable table, manual river API, and one-street flop/turn reference had separate interfaces. The one-street evaluator excluded future betting, so its turn frequencies could differ from a turn-to-river game. The existing CFR recursion already handled private-deal chance reach; adding public chance nodes to that same recursion should preserve river results while permitting a tiny exact turn continuation.

**Change.** Added immutable `PublicState` and `MultiStreetGame` with street/board, current actor, per-street and cumulative chip payments, action/reveal history, raise size, folds, showdown, exact public-card chance, and postflop action order. Chance nodes now enter the preserved synchronous CFR/CFR+/DCFR traversal and exact finite-game best-response evaluator. Added typed `SolveRequest`/`SolveResult` with every compatible acting-hand root strategy, chip/BB conditional EVs, selected hand and action gap, range aggregate, iteration count, NashConv, latency, assumptions, and explicit unsupported reasons. Solver Lab's `/api/solve` and the legacy river route delegate to this shared API. `request_from_game()` constructs a live request from public engine state and the hero's own hand; Play & Learn calls the shared API but remains gated on validated ranges. The old `TinyRiverCFRBackend` now delegates rather than solving separately. The UI displays per-hand strategy as a 13×13 class grid and allows tiny turn studies; preflop and experimental flop advice remain unavailable.

**Controlled validation.** On the saved four-compatible-deal river fixture, 50 CFR+ iterations produced **zero reported difference** in hero-hand root frequencies, conditional action EVs, and NashConv versus the preserved `RiverGame`/`RiverCFR` reference; shared end-to-end latency was 139.68 ms and NashConv remained 1.22916 chips. A turn all-in branch matched the independently scored 44-river one-street payoff. A flop-to-river all-check profile matched the independently enumerated one-street showdown value. Separate tests checked flop→turn→river chance counts and action order, a later turn root after an opponent bet, real-engine request construction, unsupported live requests, public study-position serialization without opponent cards, full per-hand/aggregate serialization, and the legacy adapter. A local HTTP smoke returned 200 for both river and tiny-turn `/api/solve` requests. The Python suite passed 81 tests; the Node matrix smoke, compilation, and whitespace checks passed.

**Measured continuation and update variants.** `python3 -m experiments.shared_solver_milestone` saved raw inputs, output, hardware, revision, and dirty-tree state in `experiments/shared_solver_milestone.json`. On one fixed private deal with an 8-chip pot, 4-chip effective stacks, and 10 CFR+ iterations, turn-to-river betting gave NashConv 1.40774 chips in 629.71 ms; the older no-river-betting one-street game had NashConv 0.72719 chips, a different game's residual. Its check/bet-to-2 conditional EVs were −4.6656/−6.1977 chips under future river betting versus −4.4045/−5.8060 without it. These values compare different average continuation profiles and show model sensitivity, not prediction error against equilibrium. At 100 iterations on that same one-deal turn game, exact restricted NashConv was 0.18753 for CFR, 0.01696 for CFR+, and 0.00160 for DCFR; end-to-end times were 2.394/2.404/2.420 seconds. One deal is too narrow to select DCFR as a general default; CFR+ remains the lab default pending a position suite.

**Failed performance/quality gate.** A one-deal flop-to-river exact solve with 1-chip effective stacks, no raises, and just one CFR+ iteration took 8.77 seconds and left NashConv 2.42689 chips in its finite game. This is neither interactive nor converged. Flop advice remains disabled in the UI; the research API requires an explicit experimental flag and caps it at three iterations/one deal. Preflop remains unsupported. Live advice remains unavailable because no initial prior/action-likelihood model has passed held-out calibration and action-value sensitivity tests. Future experiments should validate turn root EVs across distinct boards and ranges, then address flop chance-tree cost with a statistically checked sampling or continuation approximation before enabling automatic feedback.

## E036 — 2026-10-14 — measured NashConv target and Solver Lab wiring

**Change.** Added an optional finite-game NashConv target to the typed shared solver configuration, exposed it only in Solver Lab's advanced controls, and returned the measured checkpoint trace and target status through `/api/solve`. The target checks exact best-response NashConv on the current supplied-range abstract game; it does not certify full Hold'em quality. A target of zero is allowed and means “continue until the configured iteration/time budget unless the measured exploitability is zero.”

**Validation.** The complete Python suite passed 84 tests in 16.851 seconds. New tests check invalid targets, early stop when a deliberately loose target is already met at the halfway checkpoint, failure to meet a strict target, and serialization of target/checkpoint fields through the API handler.

**Measured fixture.** On the Solver Lab three-combo-per-seat river fixture (board `As Kh Qd 7c 2s`, pot 20, stacks 100/100, 40-iteration cap), target 0.25 chips ended at the cap with NashConv 4.314712 chips and was correctly reported unmet; measured checkpoints were 10.003315 at iteration 20, 9.193891 at 24, 5.729797 at 28, 5.797374 at 32, 6.341446 at 36, and 4.314712 at 40. It took 525.89 ms end to end. Target 0.05 produced the same strategy and residual in 499.52 ms. A deliberately loose 1,000,000-chip target stopped at iteration 20 with NashConv 10.003315 chips in 282.52 ms. These are single local CPU runs; the loose-target case validates stopping and API plumbing, not strategy quality. NashConv rose between some checkpoints, so this is a measured stop condition and no monotone-convergence claim is made. The [raw fixture, outputs, and environment](../experiments/nashconv_target_followup.json) are saved.

**Next issue.** Best-response diagnostics dominate target checks and add noticeable cost. Solver Lab keeps showing actual iterations, target status, and end-to-end time. Turn search remains capped and automatic live range inference remains unvalidated.

## E037 — 2026-10-14 — profile-led multi-street CPU scaling

**Observation.** `cProfile` on the prior one-deal, one-iteration exact flop fixture counted 27,720 seven-card evaluations and 582,120 five-card subset evaluations. Search took 12.77 seconds under the profiler; 11.89 seconds of cumulative time passed through terminal payoff. The unprofiled pre-change run on the same input took 7.466 seconds end to end, including 5.844 seconds of search. These timings are separate single runs, so the profiler timing is used for attribution only.

**Hypothesis and change 1.** Multiple betting histories and turn/river reveal orders reach the same seven-card holding. Cache rank by canonical public board and private hand. On an unprofiled cold run, one exact flop iteration then took 3.381 seconds end to end / 1.458 seconds search, with 1,980 rank misses and 25,740 hits. Root action EVs and NashConv were numerically unchanged: check 1.692156 chips, bet-to-1 4.335859 chips, NashConv 2.426894 chips. The comparable turn fixture went from 669.94 to 429.59 ms end to end, again with unchanged EVs/NashConv.

**Hypothesis and change 2.** Once both players are all-in, subsequent public cards contain no decisions. Collapse those chance branches to one exact expected payoff, enumerating unordered remaining-card combinations and caching the result. Explicit ordered-runout tests on flop and turn agreed with the collapsed value. A cold one-iteration flop run took 2.261 seconds end to end / 0.784 seconds search, again with the same action EVs and NashConv to floating-point precision. The turn fixture took 414.54 ms. `MultiStreetGame` state caches were moved from unbounded class-method caches to bounded per-game caches; a weak-reference regression test shows a finished game can be collected.

**Hypothesis and change 3.** Chance-sampled vanilla CFR might trade per-iteration cost for more iterations. Following Lanctot, Waugh, and Bowling's [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://webdocs.cs.ualberta.ca/~bowling/papers/09colt-ws.pdf), especially Section 3's unbiased sampled counterfactual-value idea, the research solver now samples public-card outcomes using a fixed seed while still enumerating private deals. It integrates all-in branches exactly and evaluates final NashConv and action EVs with full public chance. A 500-seed one-iteration turn test found mean sampled root regret within 0.01 chips of the exact update on its fixed fixture. This implementation is a chance-sampling variant for research; no claim is made that it reproduces the paper's full algorithm or bound. CFR+ and DCFR are rejected in sampled mode.

**Controlled comparison.** `python3 -m experiments.multistreet_scale` saved full configuration, strategies, action EVs, cache counts, seeds, hardware, and [raw results](../experiments/multistreet_scale.json). In one tiny one-deal flop game (8-chip pot, 1-chip stack, no raises), 20 exact CFR+ iterations reached 0.102053 chips restricted NashConv in 6.257 seconds. Three 1,000-iteration sampled vanilla CFR runs took 1.671–1.739 seconds but had 0.529049–0.534386 chips NashConv; their largest conditional action-EV deltas from the exact 20-iteration profile were 0.032338–0.080622 chips. On one four-deal turn game (8-chip pot, 4-chip stack, no raises), the deeper 300-iteration exact CFR+ reference had 0.106631 chips NashConv in 16.507 seconds; 100 exact iterations gave 0.210419 chips in 5.611 seconds. Three 1,000-iteration sampled runs took 2.098–2.316 seconds but had 0.658756–0.754197 chips NashConv and maximum action-EV deltas of 0.226397–0.517927 chips from the deeper reference. The flop 20-iteration profile and turn 300-iteration profile are themselves approximate finite-game references; the deltas are descriptive, not equilibrium error.

**Interpretation.** Exact rank and all-in caching improve search cost without changing the modeled game. Chance sampling reduces time but fails the measured NashConv quality comparison on these positions, so it remains opt-in research code, unavailable in the app. Even the exact flop benchmark has only one private deal, tiny stacks, and a narrow action abstraction; it does not validate ordinary gameplay advice. The complete Python suite passed 88 tests in 9.199 seconds; the Node range-editor smoke, Python compilation, and diff checks passed. The next bottlenecks are broad-range tree growth, posterior range calibration, and a validated chance or continuation approximation on held-out positions.

## E038 — 2026-10-14 — two additional fixed turn boards

**Question.** Was the 100-iteration exact turn profile close to a deeper solve only on the first board? Before solving, selected boards `2c 7h Jd Qs` and `9c 6d 3s Th`, retaining the same two private combos per seat, 8-chip pot, 4-chip effective stack, no raises, and synchronous CFR+. The [script](../experiments/turn_board_suite.py) and [raw output](../experiments/turn_board_suite.json) save board, configuration, hardware, per-hand strategies and action EVs, exact restricted NashConv, and time. A 300-iteration profile is a deeper approximate reference. The 100-iteration profile's most frequent action for each hero hand was scored using the deeper profile's conditional action EVs.

**Observed.** On `2c 7h Jd Qs`, 100 iterations had 0.145911 chips NashConv in 5.518 seconds; 300 iterations had 0.064895 in 16.646 seconds. The modal-action losses under the deeper profile were 0.013981 chips for `Jc Tc` and 0 for `8h 8s`; maximum conditional action-EV deltas were 0.068661 and 0.234463 chips. On `9c 6d 3s Th`, 100 iterations had 0.006712 chips NashConv in 5.785 seconds; 300 iterations had 0.000751 in 19.918 seconds. Both modal-action losses were zero and maximum action-EV deltas were 0.002749 and 0.001749 chips. The earlier `As Kh Qd 7c` fixture had 100/300-iteration NashConv 0.210419/0.106631; its corresponding modal losses were 0.012036 and 0.040842 chips, with maximum action-EV deltas 0.103355 and 0.157285 chips.

**Interpretation.** In these six tiny-range hand decisions, choosing the 100-iteration profile's modal action lost at most 0.040842 chips under the 300-iteration continuation profile. This is a useful fixed-position decision check, but it is not true equilibrium regret: the deeper profile still has nonzero NashConv and supplies the continuation policy. One action EV differed by 0.234463 chips, and even 100 iterations took about 5–6 seconds per board. More boards, distinct range structures, stack/pot ratios, and post-action states are required before turn analysis could be a trusted ordinary-game feature. Automatic range inference remains unvalidated.

## E039 — 2026-10-14 — one-click Play & Learn solver demo

**Objective and implementation.** Add a real calculated strategy demonstration before further solver research. The new **Try Solver Demo** button opens a single fixed river decision inside Play & Learn. `trainer/demo.py` specifies board `As Kh Qd 7c 2s`, hero `8h 8s`, an 8-chip pot, 10-chip effective stacks, hero first to act, and two equal-weight combos per seat (`Jc Tc`/`8h 8s` versus `9h 9s`/`Ac Ad`). These are explicit illustrative priors, not reconstructed betting-history beliefs. The new `/api/demo-analysis` handler calls the same `analyze_river_position()` and typed `solve()` path as Solver Lab with 2,000 CFR+ iterations. The browser renders the five legal actions, hero-hand frequencies, conditional action EVs in chips and BB, selected-action EV/loss, restricted NashConv, iteration count, latency, and a 13×13 range strategy grid. Changing the selected action uses the saved response and makes no second solve request. The normal game resumes from its existing state.

**Measured solver result.** `python3 -m experiments.demo_analysis_baseline` saved [raw request, result, revision, and hardware](../experiments/demo_analysis_baseline.json). One CPU run took 3,689.11 ms end to end and returned exact finite-game NashConv 0.080251 chips (0.040126 BB). For the hero hand, check EV was −4.000022 chips, bet-to-2 −6.001050, bet-to-4 −6.130421, bet-to-8 −4.315231, and bet-to-10 −3.955995. These are conditional chip EVs under the solver's average continuation policy, not game-wide GTO values. The example's four compatible joint deals and restricted bet tree are deliberately small.

**Browser validation.** A real headless Chrome test (`npm run test:browser`) launched the local server, clicked **Try Solver Demo**, captured the actual HTTP response, compared every displayed action frequency and chip/BB EV with that response, checked the board, hidden opponent cards, range assumptions, 169-cell grid, NashConv, iterations, and latency, chose two different actions, verified their EV/loss feedback, confirmed only one solver HTTP request, returned to the game, and started a hand. It passed with a 3,728.87 ms browser-request solve. A screenshot is saved at [artifacts/solver_demo.png](../artifacts/solver_demo.png). The first browser run failed only because the test expected the grid heading's capitalization literally; the assertion was corrected to be case-insensitive and the full run passed. This was a test assertion error, not a solver or UI defect.

**Scope.** This is a working one-decision training demo, not a replay of a real hand or an automatic live range estimate. The displayed 0.080251-chip NashConv measures only the explicitly supplied finite abstraction and remains nonzero. Gameplay and NFSP are preserved.

## E040 — 2026-10-14 — opt-in post-action gameplay review

**Problem and hypothesis.** The one-click demo proved the browser could display real solver output, but actual hands still lacked a saved public decision state. Capturing the typed solve request before a human action should allow a later analysis request without reading opponent cards or changing game play. A tiny illustrative range can exercise the wiring while clearly exposing the unvalidated range assumption.

**Change.** `TableSession` now saves each human decision's public board, hero hand, pot and stacks at street start, prior street actions, and chosen action. `POST /api/review-decision` runs the same `solve()` backend on a saved river decision only when the player clicks **Analyze with illustrative ranges**. The review uses the observed hero hand and the first three unblocked combinations from a fixed candidate list with equal weights. These are disclosed examples, not posterior ranges inferred from earlier actions or the opponent. An arbitrary played wager outside the restricted action tree returns an unsupported-state reason. Results are stored in that hand's review record, including after the hand ends; pre-river decisions show an unsupported message. The interface labels the mode experimental and displays measured restricted NashConv, iterations, latency, action frequencies, conditional chip/BB EVs, and selected-action EV loss.

**Controlled measurement.** `python3 -m experiments.live_review_baseline` played a seeded passive check/call hand with 100-chip stacks and 1/2 blinds. At the river `5s 4h Js Qh 2s`, hero `8s Qc` faced a check with a four-chip pot and checked. The exact finite abstract-game NashConv after 1,000 CFR+ iterations was 0.00007175 chips; the review took 838.75 ms. The calculated check frequency was 0.9999949 and its conditional EV was −0.666667 chips; the selected-action loss was zero under this abstract continuation profile. The raw decision, opponent combinations, per-action values, host, seed, and revision are in [live_review_baseline.json](../experiments/live_review_baseline.json). This single scenario does **not** validate the illustrative opponent range or general live advice.

**Verification and failure.** The new regression test completed a hand, confirmed saved decisions exclude opponent hole cards, received an explicit unsupported result for preflop, and checked a real river report, frequency normalization, and persistence in completed history. The final full Python suite passed 90 tests in 13.79 seconds. A simultaneous browser test initially failed in Node's `undici` startup poll with a parser assertion; replacing only the test's startup poll with `node:http` fixed it. The subsequent isolated headless Chrome run passed, displaying the actual demo response with 0.080251-chip NashConv and 3,821.81 ms solve time. A later browser fixture substituted a seeded passive agent in the test server process, then played a complete hand through the real UI, clicked the opt-in review, and compared displayed loss, action EV, and NashConv with the real `/api/review-decision` response. That run measured 4,567.55 ms for the demo solve and 849.26 ms for the gameplay review. The test fixture changes only the spawned test process, not production opponent behavior.

**Next gate.** Measure range-model calibration and posterior sensitivity on held-out public hand histories before default automatic advice. Compare post-action root EVs against deeper restricted solves over multiple boards, ranges, stacks, and prior-action histories. A nonterminal cutoff value model and depth-limited search have not been implemented or validated in this milestone.

## E041 — 2026-10-14 — compact solver analysis interface

**Goal and change.** The one-click demo and manual Solver Lab showed real values but placed long explanations beside the decision data. Moved the demo's styles into `trainer/static/demo.css` and reorganized it into a compact title, table/strategy row, action-summary/range-grid row, and collapsed assumptions section. Frequency bars now use a controlled action-size palette; conditional EV is shown first in BB with chips beneath; the selected action and highest-EV row have separate highlights. The 13×13 strategy grid uses the selected action's color, has blocker-muted cells, hover details, and a 0–100% legend. Manual Solver Lab results received matching frequency/EV rows, selection behavior, and collapsed technical details. The long introductory lab paragraph moved behind a disclosure control. No solver math, request body, or response format changed.

**Verification.** The headless Chrome test fetched real `/api/demo-analysis` and `/api/solve` responses, checked each displayed action frequency and chip/BB EV against those responses, confirmed the visible frequency bar had nonzero width, compared selected-action loss, verified both 169-cell strategy grids, switched demo actions without another solve, selected a lab action row, and checked a 390-pixel viewport for horizontal overflow in the demo. It also retained the full playable-hand and post-action review journey. The browser run passed with 0.080251-chip demo NashConv, 0.489393-chip Solver Lab NashConv, and 0.00007175-chip saved-hand review NashConv; these are unchanged finite-game calculations on the same fixtures. The Node range-editor smoke and whitespace check passed. This was a presentation change; solve latency samples vary by local run and do not establish a performance improvement.

## E042 — 2026-10-14 — live-feedback pipeline trace and bounded flop/turn review

**Observed.** A seeded, complete Play & Learn hand was traced before any solver change. The [raw four-street trace](../experiments/live_pipeline_diagnosis.json) shows that default live `SolveRequest` objects have no ranges and therefore return unsupported at flop, turn, and river. Preflop returns even earlier because there is no preflop solver. With a three-combo illustrative opponent range, exact full flop search rejected the three compatible deals; exact turn search ran only one iteration in 13,454.3 ms and left 43.5924-chip restricted NashConv. River CFR+ completed 1,000 iterations in 850.3 ms with 0.00007175-chip restricted NashConv. The [code-level diagnosis](LIVE_SOLVER_DIAGNOSIS.md) records the exact state, function path, status, and smallest fix per street. These times are from the final saved rerun; earlier local runs varied by tens of milliseconds.

**Hypothesis and change.** The preserved `OneStreetGame` already exactly enumerates remaining public runouts for flop/turn and models current-street betting, but was disconnected from the shared `SolveRequest`. Added `SolverConfig.continuation_mode="showdown_cutoff"` to route only explicit flop/turn requests through that reference and the same `RiverCFR` machinery. It assumes **no betting on later streets** and retains exact restricted-game best-response diagnostics. A preflight compatible-deal cap prevents costly equity precomputation on broad ranges. The post-action review now uses this mode at 500 CFR+ iterations for flop and turn; river keeps the full restricted tree at 1,000 iterations. Preflop remains unsupported. `POKER_SOLVER_DEBUG=1` emits structured decision, range, solve-result, and error events to stderr without opponent hole cards; the actual [JSONL trace](../experiments/live_pipeline_debug.jsonl) is saved.

**Measured after change.** On the same hero `8s Qc`, 100-chip-stack hand with a passive opponent, the cutoff returned flop NashConv **0.0003649 chips in 1,532.0 ms** and turn NashConv **0.0002181 chips in 466.7 ms**, each after 500 iterations. These residuals are **only inside a no-future-betting game**; no action-EV accuracy against a deeper full game was measured. An adapter test matched the preserved `OneStreetGame`/`RiverCFR` frequencies, conditional EVs, and NashConv to nine decimal places on fixed flop and turn positions. An initial diagnostic run failed with `AttributeError` because `RiverState` has no `chance` attribute; `solve()` now uses `getattr(..., False)` for the common state boundary, and the rerun passed. The browser test played the full hand, requested river, turn, and flop analyses via the real API, and checked displayed cutoff EV/frequencies against each response. It also verified preflop shows unsupported.

**Interpretation and next step.** This delivers real, opt-in CFR-derived flop/turn/river feedback from disclosed assumed ranges, not default live GTO advice. The largest unresolved gap is preflop state/chance modeling. The largest quality gap is how far no-future-betting cutoff values deviate from deeper continuation solves; run paired held-out root-action comparisons before treating early-street advice as strategically useful. A fitted betting-history range model is still needed for automatic default feedback.

## E043 — 2026-10-14 — offline NFSP worker, dashboard, and restricted cache smoke

**Question.** Can self-play run in a separate scheduled process, checkpoint optimizer/replay state, and expose measured runs without blocking the table? Can action sampling refuse absent solver cache entries?

**Change.** Added `trainer/offline_training.py`, configurable NFSP network/replay/target cadence, status/config/start/stop API, ML Training dashboard, and an exact-request-keyed restricted river demo cache. The worker saves candidates at configured hand counts, evaluates the displayed seat-1 policy against fixed random and heuristic opponents, and promotes against the incumbent only if at least 100 evaluation hands and an unpaired normal-approximation 95% difference interval has lower bound above zero. A separate fixed-demo river CFR+ job records exact restricted NashConv and promotes only when it improves below 0.1 chip. Revision `32d3fb3`, dirty working tree, Apple M2, Python 3.10.6. Both jobs are CPU-only.

**Observed.** A temporary 24-hand run (seed 11, 16×16 network, batch 4, checkpoints at 8/16/24, four evaluation hands/opponent) wrote two policy files and three actual curve points; policy losses were 1.3391, 1.2409, 1.2874. The candidate was **not promoted** because four hands are below the 100-hand gate. A second temporary 32-hand run with a 20-iteration CFR+ river cache job reported restricted NashConv 2.09864 chips in 0.0505 s; that solution was also **not promoted**. An HTTP worker smoke started the separate process, saved a 24-hand candidate, reported a waiting state and the same 2.09864-chip failed river cache gate, then accepted a stop request. The full Python suite passed 95 tests and headless Chrome rendered the dashboard and cached-solution action selection alongside real demo EVs/frequencies. Browser demo response: NashConv 0.080251 chips, 2,000 iterations, 5,433.7 ms recorded solve time on that run.

**Follow-up.** Added a true held-out split of best-response observations for the supervised average-policy learner; those examples are excluded from reservoir training updates. A disposable 64-hand run (seed 11, 16×16 network, batch 4, 20% held-out, checkpoints every 16) recorded held-out counts 1/4/5/6 and cross-entropies 1.2305/1.2851/1.3199/1.2219. These tiny sample counts cannot establish generalization. Online self-play still has no meaningful supervised epoch count; each scheduled run is a cycle of hands.

**Resume regression.** A new deterministic stop/resume test first failed because the worker counted the next unplayed hand as completed (reported nine after eight). Tracking completion after each played hand fixed it. The corrected test passed a bit-for-bit tensor comparison of the resumed and uninterrupted 16-hand seat-1 policies under the same seed and configuration.

**Interpretation.** Scheduling, persistence, API polling, browser rendering, and conservative promotion gates work in these smoke conditions. No new NFSP candidate was shown to be stronger; no new blueprint was promoted. The cache covers one exact explicit-range river position, not a full-game solved blueprint. Next: longer multi-seed training, paired evaluation on more baselines, and a broader validated position library before cache-driven full-hand play.
