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
