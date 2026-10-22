# Project metrics that matter

## Offline training promotion and cache coverage

For every candidate checkpoint, retain seed, settings, trained hand count, per-checkpoint Q/policy loss, evaluation opponent and rules, hands, seat, mean chips/hand, standard error, incumbent comparison, interval, SHA-256, and promotion decision. The active NFSP seat-1 policy is promoted only after at least 100 fixed-opponent evaluation hands and a positive lower bound on the unpaired normal-approximation 95% candidate-minus-incumbent interval. This gate is a regression filter against one heuristic, not a proof of robust strength or equilibrium. A stronger future gate needs paired seat-balanced direct matches and multiple opponent classes. For a cached CFR strategy, record exact request key, board/ranges/action abstraction, iterations, exact restricted NashConv, solve latency, and the number of real gameplay states that match the cache key. A low NashConv for one key must never be reported as broad cache coverage.

The 2026-10-14 smoke created a 24-hand NFSP candidate and a 20-iteration fixed-river cache candidate; neither passed promotion. The browser-confirmed 2,000-iteration demo cache is one solved position only. See E043 in `EXPERIMENT_LOG.md` and `BENCHMARKS.md` for actual values.

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
| River CFR update comparison | One fixed 9-deal river game, 2,000 iterations, exact best-response NashConv | CFR 0.1385; CFR+ 0.0184; DCFR 0.0200 chips | Update-rule convergence evidence for this one abstraction only; all remain nonzero |
| River conditional action EV | Root action EVs checked by brute-force enumeration across both acting seats and all compatible deals | Maximum test difference <1e-10 chips | The reported ordinary action EV computation is separated from CFR's reach-weighted counterfactual regret update |
| River action-EV stability | 500-iteration estimates versus 2,000-iteration profile on the fixed tiny game | Mean/max abs difference: CFR 0.0245/0.0951; CFR+ 0.0078/0.0146; DCFR 0.0084/0.0187 chips | Policy-dependent stability on one game; not a held-out value model or population accuracy result |
| River solver latency | Same 9-deal case, 2,000 iterations, one recorded Apple M2 run | CFR 7.52 s; CFR+ 7.81 s; DCFR 8.88 s training; 11.3–14.0 ms exact analysis | Single-run CPU reference evidence, not a general performance claim |
| River position generalization check | Five seeded boards/range pairs, weighted three-hand ranges, 500 CFR+ iterations each | NashConv 0.0056–0.2038 chips; 0.78–2.11 s train | Reproducible support across varied tiny games; convergence varies by position and sample is too small for broad claims |
| Live-adapter latency probe | 10 ms search budget on one explicit-range river state | 3 iterations; 10.23 ms search; 20.11 ms end-to-end; NashConv 5.432 chips | Current budgeted solver is far from convergence and unsuitable for live advice |
| River analysis UI/API | Sample 3-hand ranges/seat, 500 CFR+ iterations, board As Kh Qd 7c 2s | 2,689.5 ms in latest run; NashConv 0.48939 chips; five hero-hand actions with frequencies, conditional EVs, and EV-loss values | Visible analysis for an explicit finite river game; residual exploitability remains and ranges are not inferred from history |
| Later river decision | Hero check, opponent bet-to-10, same manual ranges | NashConv 0.85024 at 500 iterations / 0.11100 at 2,000; latency 416.9 / 1,626.1 ms | CPU root-state replay and convergence within the stated abstract game; ranges were not action-conditioned |
| Public-card chance | Three weighted combos per seat on one flop, all 49 candidate turn codes | Probability sum 1.0000000000000004 in 3.049 ms | Independent card-removal chance arithmetic for that sample; turn CFR is measured separately |
| Shared river numerical parity | Four compatible private deals, 50 CFR+ iterations, one board | Max acting-hand frequency and conditional EV differences versus preserved river solver: 0.0; identical 1.229164-chip NashConv | The new shared game/solver path reproduces the reference on this fixture; broader parity suite pending |
| Exact turn-to-river update variants | One private deal, 8-chip pot, 4-chip stack, 100 iterations | NashConv: CFR 0.187533; CFR+ 0.016962; DCFR 0.001604 chips; end-to-end 2.39–2.42 s | Future betting and exact public chance are implemented on a tiny game; one board cannot choose a global update default |
| Exact flop-to-river stress | One private deal, 1-chip stack, no raises, one iteration | 8.77 s; 2.426894-chip NashConv | Fails interactive latency and convergence gates; flop advice remains disabled |
| Range inference gate | Public-history Bayesian updates with supplied per-hand action likelihoods | Unit tests pass; no fitted likelihood provider or held-out accuracy measured | Correct update plumbing in tested cases; automatic live ranges remain unvalidated |
| River labels | Recomputed labels on 4 saved positions; maximum absolute difference | <9e-15 chips | Deterministic regeneration for this sample, not an independent solver validation |
| River payoff accounting | CFR payoff vs engine stack delta under the documented dead-pot baseline, 72 cases | 0 chips maximum difference | Tested zero-sum payoff mapping for those representative states |
| River value model | Three initializations, same 180 training and 20 held-out boards | MAE 2.067–2.353 chips; RMSE 2.737–3.043 | Model-seed sensitivity on one small held-out set; each board yields six correlated hand examples |
| Shallow river CFR | 20-iteration conditional EVs vs 200-iteration label on the original held-out set | 0.961 chips MAE | A direct shallow solve approximates these labels better than the neural model; does not show a better strategy at equilibrium |
| River deeper reference | First three held-out positions, 200 vs 500 iterations | EV MAE 0.080 chips; NashConv mean 0.246 at 500 vs 0.428 at 200 | Evidence that 200-iteration values stabilize on these three positions; small sample only |
| Data augmentation | Four suit mappings per training board, same seed/architecture/optimizer/test boards | MAE 2.134 raw vs 2.201 augmented; RMSE 2.915 vs 2.881 | Suit augmentation did not reduce MAE in this controlled run |
| Hold'em NFSP | 1,000 fixed hands/seed vs random, three seeds | +0.86 to +1.48 chips/hand | Task-specific performance against random in that restricted setup |
| Hold'em NFSP | Same evaluation vs the 32-sample heuristic | −1.04 to −2.08 chips/hand | The tested NFSP policies lose chips against this fixed heuristic |
| Playable engine and solver | Complete Python test suite, including randomized engine play, preserved river parity, turn/flop chance transitions, range updates, and shared adapter gates | 81/81 tests passed | Tested engine and tiny postflop semantics; does not establish exhaustive rule correctness or full Hold'em solver validity |
| Browser modes | Default Play & Learn and advanced Solver Lab using one solver API | HTTP smoke loaded page/script/presets and solved restricted river/turn positions through `/api/solve`; Node smoke covered 169 cells, 1,326 combos, and the per-hand strategy grid | Both entry points call the shared contract; live requests still return unsupported pending validated ranges; visual browser automation remains unrun |
| Browser game loop | Heuristic hand, custom raise, showdown, then next hand | 8 actions, 5-card board, button alternated | End-to-end playability on one real local HTTP session |

Raw solver and model measurements are in `experiments/river_value_diagnosis.json`,
`experiments/river_value_equity_180_seed5*.json`, and the original experiment
records. River UI analysis raw request/response is in
`experiments/river_analysis_ui_baseline.json`; the newer Solver Lab root-state
and chance record is `experiments/solver_lab_milestone_v2.json`. The shared postflop benchmark is `experiments/shared_solver_milestone.json`. NFSP evaluation conditions and confidence intervals are in
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

For automatic range inference, report the source of each initial prior and
action-likelihood dataset, count of independent players/sessions, held-out
action log loss against a context-only baseline, calibration by street and
action size, and posterior mass assigned to the actual hidden hand in held-out
games. Then measure how inferred ranges change root action EVs and chosen-action
regret versus reference ranges on the same public decisions. The current
Bayesian update tests establish arithmetic only; none of these model-quality
measurements exists yet, so automatic solver advice remains gated.

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
- Implemented a CPU tiny-range river analyzer with CFR/CFR+/DCFR, per-hand
  action frequencies, conditional action EVs, and exact best-response NashConv;
  2,000-iteration CFR+ reached 0.0184 chips NashConv on one 9-deal abstraction.
- Built a 13×13 hand editor covering 1,326 suit combos and a separate Solver
  Lab that reports conditional CFR+ frequencies, EVs, and exact restricted-game
  NashConv. On one three-combo river sample, first-action 500-iteration NashConv
  was 0.4894 chips in 2.69 seconds; after a check/bet, 2,000 iterations reached
  0.1110 chips in 1.63 seconds with manually supplied, unconditioned ranges.
- Built a separate flop/turn one-street CFR+ CPU reference with exact public-card
  runout enumeration. On a four-deal fixture, 200 iterations measured 0.08849
  chips turn and 0.06864 chips flop NashConv in the restricted game. The model
  assumes no betting after the current street, so these are not full-street
  strategy-quality metrics.
- Built a typed shared postflop solve API returning each supported acting-hand
  strategy, conditional chip/BB EVs, action gaps, range aggregates, and exact
  finite-game NashConv. It matched the preserved river solver exactly on one
  four-deal fixture and traversed turn-to-river betting with exact card chance.
  A one-deal 100-iteration turn run measured 0.00160-chip DCFR NashConv in
  2.42 seconds; the one-iteration flop stress failed latency/convergence gates.
- Profiled a restricted river value-network prototype against direct shallow
  solving; the network was faster in the measured micro-workload but had
  materially higher held-out EV error and failed the accuracy gate.
- Trained restricted Hold'em NFSP policies that gained against random opponents
  in the fixed suite but lost chips against the tested heuristic.

Do not claim full-game DeepStack, safe re-solving, GTO advice, broad-range
multi-street quality, automatic live advice, or CUDA acceleration from the
current evidence.

## Solver budget and quality metrics added (2026-10-14)

Solver Lab accepts an optional NashConv target and returns the requested chip threshold, exact measured checkpoint trace, achieved iteration count, final restricted-game NashConv, and `restricted_target_met` or `restricted_residual_measured`. On one three-combo-per-seat river fixture under a 40-iteration cap, target 0.25 chips was unmet at 4.314712 chips after 40 iterations (525.89 ms); target 0.05 had the same result (499.52 ms). Target 1,000,000 chips stopped at iteration 20 with 10.003315 chips residual (282.52 ms). Track target hit rate only alongside final NashConv, per-hand action regret/EV error against deeper references, runtime, and the exact range/action abstraction. The loose-target result is a plumbing check and must not count as a strategic quality win. One run per condition is insufficient for stable latency claims.

## Multi-street scale evidence (2026-10-14)

On one fixed one-deal flop position, profiler evidence linked repeated hand evaluation to most of the search cost. Canonical rank caching and exact all-in chance integration reduced a one-iteration unprofiled run from 7.466 seconds to 2.261 seconds with the same root EVs and NashConv. In the saved [multi-street experiment](../experiments/multistreet_scale.json), 20 exact CFR+ flop iterations reached 0.102053 chips restricted NashConv in 6.257 seconds. On one four-deal turn position, 300 exact CFR+ iterations reached 0.106631 chips in 16.507 seconds. Three seeded 1,000-iteration sampled CFR runs on those positions were faster but had 0.529–0.534 and 0.659–0.754 chips NashConv, respectively. These are meaningful architecture and correctness measurements, but not evidence of a playable full-range GTO adviser. For future project claims, pair any throughput number with range size, street, betting abstraction, final exact NashConv, action-EV delta to a deeper reference, board/range split, and end-to-end latency distribution. The next resume-worthy milestone is a held-out multi-position result at a documented accuracy/latency gate, not the speedup alone.

On a follow-up [three-board turn comparison](../experiments/turn_board_suite.json) with the same four-deal ranges, 100-iteration exact CFR+ had restricted NashConv 0.006712–0.210419 chips and 5.518–5.785-second latency. Relative to 300-iteration profiles, the maximum loss from its modal action across six hero-hand decisions was 0.040842 chips, while maximum conditional action-EV delta was 0.234463 chips. The deeper profiles themselves had 0.000751–0.106631 chips residual and took 16.507–19.918 seconds. This supports a narrow, measured decision comparison for the finite game. It does not support an ordinary-play GTO or live-advice claim.

## User-facing solver demo evidence (2026-10-14)

The one-click Play & Learn demo calls the shared CFR+ solver on a documented four-deal river abstraction. One saved CPU run measured 2,000 iterations, **0.080251-chip restricted NashConv**, and **3.689-second end-to-end latency**; a separate real Chrome run measured 3.729 seconds. The browser test verified five real action frequencies and chip/BB EVs against the HTTP response, 169 range-grid cells, selected-action EV/loss, one solver request after two action selections, hidden opponent cards, and normal game resumption. These are concrete product-integration metrics. Do not present the demo as automatic range inference, full-game GTO, or a sub-second live adviser.

The opt-in saved-hand review now proves a real decision snapshot can reach the same solver without leaking opponent cards. One seeded passive-opponent hand took 838.75 ms for a 1,000-iteration river solve and measured 0.00007175-chip exact NashConv in a three-deal abstraction. Its three opponent combos are fixed illustrative assumptions, so neither the low NashConv nor the calculated zero loss for the selected check establishes real-game advice quality. Useful future project metrics are **supported-decision coverage under a declared abstraction**, **held-out range-model log loss and calibration by action/street**, **root action-EV error and selected-action regret versus deeper solves**, and **p50/p95 end-to-end feedback latency at a specified quality threshold**. None of those distributional metrics has been measured yet; do not quote them on a résumé.

The [four-street diagnostic](LIVE_SOLVER_DIAGNOSIS.md) now separates **coverage** from **strategic accuracy**. One seeded hand produced opt-in solver reports on flop, turn, and river, while preflop remained unsupported. The flop/turn reports use a no-future-betting cutoff: 500 iterations took 1.532/0.467 seconds and left 0.000365/0.000218 chips NashConv **inside those cutoff games**. The full turn tree took 13.454 seconds for one iteration and had 43.592 chips NashConv. These numbers justify the cutoff as a bounded integration path, not an accuracy improvement. Before a résumé claim about four-street advice, measure preflop coverage, held-out downstream action regret and EV error, range calibration, and latency percentiles across many positions.

## Offline CFR scenario cache smoke — 2026-10-20

One reproducible random river position (four compatible joint deals, two private combos per seat, 1,000 CFR+ iterations) reached **0.000288735-chip exact NashConv within its supplied restricted game** in **2.2475 seconds**. It passed the 0.1-chip cache screening threshold. This demonstrates a reproducible CFR solve and artifact write for that specific finite game. It is not a multi-street policy, a full-range result, or a strength comparison; do not use it as a résumé claim of broad GTO ability. The next meaningful résumé metrics are the share of a declared random-scenario batch passing the gate, p50/p95 solve latency, exact cache coverage/hit rate, and downstream action-EV error against deeper references.

## Restricted tree profile and sizing metrics — 2026-10-20

On one fixed four-compatible-deal river game, changing the menu from baseline 50%/100% plus minimum size to dense 25%/50%/75%/100% sizing increased the enumerated public decision states from **26 to 42**, information-set action slots from **148 to 244**, and median solve latency from **233.79 to 384.59 ms** at 100 CFR+ iterations. Allowing a second raise increased those counts to 48 states / 280 slots and median latency to 469.88 ms. This is a controlled local runtime/tree-size comparison, not a strategic quality comparison across menus.

Seven declared one-deal postflop profiles were solved and checked for normalized per-hand strategies, finite action EVs, and exact restricted-game NashConv. Five reached ≤0.1 chips on their own restricted games: river root 0.005978 chips at 100 iterations / 36.43 ms; river facing-bet 0.001188 at 100 / 15.35 ms; full turn 0.013959 at 100 / 2,598.55 ms; turn cutoff 0.011192 at 100 / 152.20 ms; flop cutoff 0.008380 at 100 / 567.22 ms. Experimental exact full flop reached 0.851082 chips at its three-iteration cap (the one-iteration catalog point was 2.426894 chips / 2,849.30 ms). Sampled full-flop CFR reached 0.482945 chips at 2,000 iterations / 2,826.95 ms; three seeds ranged 0.457810–0.489940 chips. Both full-flop modes fail the 0.1-chip gate. Full per-run outputs are linked from `docs/BENCHMARKS.md`. The suite is one board/range fixture, not representative coverage; cutoff profiles have no future betting. Do not frame these values as project-wide accuracy or GTO performance.

On same-profile iteration sweeps, 30-iteration full turn was within **0.0657-chip action-EV MAE** and **0.00218-chip root policy regret** versus 100 iterations. Turn cutoff at 30 was within **0.0668-chip MAE** and **0.00165-chip regret**; flop cutoff at 30 was within **0.0351-chip MAE** and **0.0343-chip regret**, with **0.232 strategy L1 deviation**. The deeper finite-iteration result is a comparison reference, not ground truth. Exact request/source-fingerprinted river cache lookups now have a regression-tested consumer in Scenario Lab; cache hit rate on user-selected positions has not been measured.

## Held-out small-tree follow-up — 2026-10-20

Three seeded fixtures per profile measured river full-tree NashConv median/max **0.00567/0.18942 chips** (200 iterations, 229 ms median); full turn **0.01305/0.01429** (100, 1.96 s); turn cutoff **0.01858/0.02116** (100, 49.6 ms); flop cutoff **0.00931/0.02201** (100, 408 ms). Deepening the worst river fixture to 1,000 iterations lowered residual from 0.18942 to 0.01914 chips in 1.125 s; 500-iteration root action EV MAE and policy regret versus the finite 1,000-iteration reference were 0.00396 and 0.00165 chips. Small synthetic ranges only; these are not broad coverage or GTO-quality metrics.

On the E058 held-out fixtures, matched-budget DCFR reduced median restricted NashConv versus CFR+ from 0.01305 to 0.00099 chips on full turn, 0.01858 to 0.00083 on turn cutoff, and 0.00931 to 0.00041 on flop cutoff, with about 4–7% higher median runtime. On river, medians were 0.00567 (CFR+) and 0.00015 (DCFR), but worst cases were 0.18942 and 0.25337 respectively. Three fixtures per profile; no global algorithm win or GTO claim.

## 30-position constrained postflop validation — 2026-10-21

Thirty deterministic synthetic positions (six each: river root, river facing bet, full turn, turn cutoff, flop cutoff), with one or two exact combos per seat. Fixed budgets returned supported outputs in 30/30 and met the 0.1-chip exact restricted NashConv screen in 26/30. With per-street iteration ceilings and the built-in exact residual target, 30/30 met the target in this batch. Median/p95 latency under that mode: river root 1.20/2.55 s, river facing 0.19/0.32 s, full turn 1.03/1.04 s, turn cutoff 0.20/0.32 s, flop cutoff 1.90/2.20 s. A parallel run of the Shark 2.0-inspired menu on full river/turn trees also met 30/30 target; cutoff trees retained their fixed menu. These figures are restricted-game residuals on a small synthetic cohort, not full-game GTO, abstraction accuracy, or production reliability.

An additional 20 action-history states extended the synthetic validation set to 50. All 20 supported and met ≤0.1-chip exact restricted NashConv under adaptive per-street budgets. River bet/raise response median residual was 0.000007 chips at 68 ms median; full-turn facing-bet 0.00481 chips at 115 ms; turn cutoff facing-bet 0.000441 at 98 ms; flop cutoff facing-bet 0.000824 at 1.58 s; flop cutoff after check 0.000960 at 1.67 s. Each group has four fixtures and at most two combos per seat; flop/turn cutoff omits later betting. This is a bounded validation cohort, not arbitrary-state or GTO coverage.

River range-width experiment: at 8 exact combos per seat (64 compatible private deals), two Shark-inspired scenarios both met the 0.1-chip restricted NashConv target; median latency was 5.57 seconds and the two-sample max was 10.06 seconds. Widths 2/3/4/6 also passed in 12 additional cases. This demonstrates current sparse range handling through 8 entries, not the full 1,326-combo matrix or broad-range equilibrium quality.

Deepening the slowest 8x8 fixture from 500 to 2,000 CFR+ iterations reduced restricted NashConv **0.09274 → 0.03197 chips** and took **9.98 → 39.64 seconds**. For its selected hand, action-EV MAE against the 2,000-iteration finite reference fell **0.24557 → 0.09367 chips** (500 versus 1,000 iterations); policy regret was **0.01229 → 0.00591 chips**. This exposes a useful quality metric: range-level NashConv can pass while per-hand conditional EVs still move. The single-fixture run is a development diagnostic, not a résumé claim of broad accuracy. Raw reproducible input and outputs: [`deepen_widest_river_range.json`](../experiments/deepen_widest_river_range.json).

The solver core also completed one sparse 10x10 and one 11x11 river position (100/121 compatible deals). At 500 iterations, 10x10 measured **0.00748-chip** restricted NashConv in **15.32 s**; 11x11 needed 1,000 iterations and **36.70 s** to reach **0.07445 chips**. On the 11x11 selected hand, 500-vs-1,000 iteration action-EV MAE was **0.04788 chips**, strategy L1 **0.02480**, and reference-policy regret **0.01062 chips**. This expands demonstrated solver-core input width, while the product still caps user ranges at eight combos per seat. Single fixtures and finite references; not full-range or general accuracy claims. Raw runs: [`river_range_width_extension.json`](../experiments/river_range_width_extension.json) and [`deepen_river_range_width_extension.json`](../experiments/deepen_river_range_width_extension.json).

The default Scenario Lab example exposed why a **per-result convergence screen** matters: its nine-deal river solve returned 87.4% jam at 500 iterations with 0.4894-chip restricted NashConv, then 98.3%/99.1% jam at 2,000/5,000 iterations with 0.0511/0.0313-chip residual. The jam EV remained about +36.7 chips because the nut hand was paid by the assumed opponent range roughly one-sixth of the time; an independent payoff identity reproduced it to <1e-9 chips. This is a useful arithmetic and UI integrity check, not evidence of real-player strategy accuracy or a résumé strength claim. Raw data: [`audit_default_lab_river.json`](../experiments/audit_default_lab_river.json).
