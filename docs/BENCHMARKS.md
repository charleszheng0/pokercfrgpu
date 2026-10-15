# Measured benchmarks

## Scheduled training and solver cache smoke (2026-10-14)

On Apple M2/Python 3.10.6, a disposable 24-hand NFSP run using a 16×16 network, batch 4, and checkpoints every eight hands produced policy losses 1.3391/1.2409/1.2874; the four-hand-per-opponent evaluations are too small for a strength conclusion and no model was promoted. A separate 32-hand smoke with a 20-iteration CFR+ solve of the fixed explicit-range river demo measured 2.09864-chip exact restricted NashConv in 0.0505 seconds and failed the 0.1-chip cache gate. A headless Chrome end-to-end run rendered a 2,000-iteration demo solution with 0.080251-chip NashConv and 5,433.7 ms reported solve latency, then sampled an action from its cached hand strategy. These are single smoke runs, not latency distributions or general strategy-quality estimates. The Python suite passed 95 tests; the Chrome browser suite passed.

A later disposable 64-hand NFSP run with a 20% held-out supervised split recorded validation sample counts 1/4/5/6 and policy cross-entropy 1.2305/1.2851/1.3199/1.2219 at 16/32/48/64 hands. This confirms the split/metric is wired, not that it predicts playing strength.

## River continuation-value diagnosis (2026-10-13)

The prior network's reported 2.07–2.35-chip MAE was measured on 120 hand labels
from 20 held-out public boards (six labels per position), with 180 training
boards, 200-iteration average-CFR root values, and the same data split across
three initialization seeds. The label is root hand-conditional EV in chips,
not a depth-cutoff continuation value. Targets were divided by 14 during MSE
training and predictions restored to chips. MAE is therefore only a regression
metric on a small clustered test set; it does not show accurate action values.

The controlled diagnosis at
`experiments/river_value_diagnosis.json` found less than 9e-15 chips label
regeneration error on four positions and zero payoff-mapping error across 72
representative cases.
Suit augmentation (four suit mappings per training board) used the same model,
seed, optimizer, epoch count, and held-out boards as the raw-data run:

| Training data | Train labels | Fit time | Held-out MAE | Held-out RMSE |
|---|---:|---:|---:|---:|
| One encoding per training board | 1,080 | 1.65 s initially; 4.63 s after range refactor | 2.134 chips | 2.915 chips |
| Four suit mappings per training board | 4,320 | 5.02 s initially; 8.19 s after range refactor | 2.201 chips | 2.881 chips |

This augmentation did not improve MAE. The single 20-position test split makes
the difference inconclusive. Repeat fit times varied materially on the same
Apple M2 process, so these small-run timings are not a stable performance
benchmark.

For three fixed held-out positions, direct root values from 20 CFR iterations
had 0.725, 0.838, and 0.806 chips MAE against the 200-iteration labels; 200 vs
500 iterations had 0.069, 0.112, and 0.059 chips MAE. NashConv at 20/200/500
iterations was respectively 1.911/0.570/0.344, 2.346/0.597/0.347, and
1.165/0.116/0.047. This supports better value stability at 200 than 20
iterations on these three positions, but the 500-iteration strategy remains
measurably exploitable. Deeper reference checks have not yet covered all 20
held-out positions, and no model-driven action-regret comparison exists.

**Status:** no validated neural continuation-value accuracy gate has been
passed. The root value model is not integrated into search or live play. See
`docs/PROJECT_METRICS.md` for the metrics required before making solver or
resume claims.

## Playable Hold'em engine and browser (2026-10-13)

| Check | Conditions | Result |
|---|---|---|
| Complete Python tests | `python3 -m unittest discover -s tests -q`, four final runs | 43/43 passed in 3.832 s, 4.090 s, 3.852 s, and 3.950 s |
| Randomized legal play | Existing 150 seeded hands plus 75 additional games sampling arbitrary legal integer wager targets | All hands terminated; no illegal actions or chip-conservation failures |
| Cards and public events | Full runout, two hole cards per seat, five board cards, three burns | 52 unique cards; deal/blind/action/board/award events recorded |
| Arbitrary raise | Browser API custom preflop raise to 7 chips, not in preset action list | Accepted and logged as target 7; bot completed the hand |
| HTTP play loop | Local server, heuristic opponent, complete action sequence | Eight actions, five-card showdown board, then hand 2 started with button at seat 1 |
| Frontend validation | Extracted embedded script | `node --check` passed |

These checks validate functionality and tested rule cases, not strategic quality,
multiway poker, bankroll carry-over, or solver analysis. The heuristic remains
a baseline and is not presented as a strong opponent.

Hardware: Apple M2 (`arm64`, macOS Darwin 23.5.0), Python 3.10.6. PyTorch 2.10.0 is installed, `torch.cuda.is_available()` returned `False`, and `nvcc` was not found. Earlier measurements recorded Git HEAD `f0f9fc7feeb704d51bb77bf8763e00ea30d5c0fe`; newer Hold'em NFSP records report `23308d7946b52a247b480e15aa7ec0a93723c4f1`. Both were dirty working trees. Results are from the actual commands below and do not represent a clean released revision.

| Measurement | Command / conditions | Result |
|---|---|
| CPU engine suite | `python3 -m unittest discover -s tests -v` | 7/7 passed; 150 seeded random hands within the suite; final run 0.086 s |
| Heuristic vs random | `python3 -m experiments.baseline --hands 100 --seed 7 --samples 32` | −2.68 ± 5.91 standard-error chips/hand; 27/100 wins; 4.75 s total |
| Independent evaluator check | `PYTHONPATH=/tmp/pokercfrgpu-oracle:. python3 -m experiments.crosscheck_evaluator --trials 5000 --seed 19` | 0 rank disagreements against Treys 0.1.8 |
| Kuhn synchronous CFR | `python3 -m experiments.kuhn_cfr` | Exact NashConv: 0.0513 at 100, 0.0145 at 1,000, 0.0046 at 10,000 iterations; seat-0 EV at 10,000: −0.05555 chips |
| Leduc synchronous CFR | `python3 -m experiments.leduc_cfr` | Exact NashConv: 1.5848 at 20, 0.4322 at 100, 0.1395 at 500 iterations; 30.79 s elapsed through 500 including checkpoint evaluation |
| Leduc NFSP, η=0.1 | 3 seeds, 20,000 episodes, exact best responses | Final NashConv: 3.8052, 3.2425, 2.9777; mean 3.3418 |
| Leduc NFSP, η=0.3 | Same 3 seeds and horizon, only η changed | Final NashConv: 2.5385, 2.9604, 2.5559; mean 2.6849 |
| Leduc NFSP, longer run | Seed 11, η=0.1, 100,000 episodes | NashConv 2.6698; one seed only |
| CPU operation profile | `python3 -m experiments.profile_cpu` | 0.189 ms seven-card evaluation; 14.09 ms heuristic decision (32 equity samples); 0.034 ms NFSP policy inference; 110 ms profiled Leduc CFR step |
| Tiny-range Hold'em river CFR | `python3 -m experiments.river_cfr` | 9 compatible deals; exact NashConv 2.0366 at 20, 0.7513 at 100, 0.3018 at 500 iterations; 1.83 s through 500 |
| River value network, raw cards | 60 train/20 held-out boards, 200-iteration labels | 4.28 chips held-out MAE; zero predictor 3.95; shallow 20-iteration CFR versus reference 0.96 |
| River value network + equity | Same saved boards and labels, one extra feature | 2.59 chips held-out MAE at 60 training boards |
| River value network + equity, more data | 180 train/20 fixed held-out boards, seeds 53/54/55 | 2.35, 2.23, 2.07 chips held-out MAE; still above shallow CFR's 0.96 |
| River value end-to-end latency | Saved 180-board model, one held-out board, feature construction included | 1.73 ms network path versus 70.3 ms 20-iteration shallow solve (about 40.5× faster on this tiny case) |
| NFSP reward target scaling | η=0.3, seeds 11/23/37, 20,000 episodes, scale 20 versus 1 | Scale-20 final NashConv 2.6916/2.6157/2.1579, mean 2.4884; unscaled mean 2.6849; one seed worsened |
| River range-size scaling | 100 CFR iterations, 3/6/10 hands per seat, Python `tracemalloc` enabled | Training 1.47/5.37/13.80 s; Python peak 0.89/1.52/2.54 MB; exact BR 0.011/0.038/0.112 s |
| Restricted Hold'em NFSP, initial test | Seeds 11/23/37, 10,000 self-play hands each, 200 evaluation hands per opponent | Versus random: +1.97/+1.17/+2.14 chips/hand; versus heuristic: −1.71/−1.77/−2.85 |
| Restricted Hold'em NFSP, fixed suite | Same saved policies, 1,000 hands per opponent, evaluation seed 900000 | Versus random: +1.078/+0.857/+1.477 ± 0.453/0.450/0.449 SE; versus heuristic: −1.037/−1.750/−2.077 ± 0.435/0.414/0.401 SE |
| Untrained initialized policy control | Corresponding seeds, same fixed suite and policy-action RNG | Versus random: −0.379/−0.146/+0.073 ± 0.376/0.375/0.381 SE; versus heuristic: −0.912/−0.924/−0.968 ± 0.402/0.395/0.406 SE |
| Saved Hold'em NFSP CPU inference | 1,000 repetitions on one preflop observation, one PyTorch CPU thread | Median encoding 0.050 ms; forward 0.038 ms; complete action 0.129 ms (p95 0.135 ms) |
| NFSP faced-bet guard ablation | Seed-11 policy, 500 fixed hands versus heuristic; 32-sample heuristic replaces only faced-bet actions | NFSP −0.708 ± 0.636 SE chips/hand, 1.05 ms/decision; guard −0.758 ± 0.553, 5.51 ms/decision |
| NFSP hand-level diagnostic | Seed 11, 500 new fixed hands versus heuristic, trained/initialized policy | Trained/control large losses (≥10 chips): 191/148; folds 14/80; all-in hands 215/286 |
| Hold'em NFSP training horizon | Seed 11, 10,000 versus 30,000 self-play hands; same 1,000-hand fixed suite | 10k: +1.078 random, −1.037 heuristic; 30k: +0.817 random, −1.505 heuristic chips/hand |

The benchmark script alternates the heuristic seat and saves exact seed, sample count, elapsed time, and revision in [raw JSON](../experiments/baseline_seed7.json). The [evaluator result](../experiments/evaluator_crosscheck_seed19.json) uses an optional Treys install in `/tmp`; Treys is not a runtime dependency. Small-game raw results are in [Kuhn CFR](../experiments/kuhn_cfr.json), [Leduc CFR](../experiments/leduc_cfr.json), [NFSP η=0.1](../experiments/leduc_nfsp_seed11_long.json), [NFSP η=0.3](../experiments/leduc_nfsp_eta03_seed11.json), and the corresponding `other` files for seeds 23 and 37. The [CPU profile](../experiments/cpu_profile.json) includes cProfile's top functions. These timings were taken on a working tree with other runs possible; they are bottleneck evidence, not a clean hardware comparison.

NashConv is the sum of unilateral best-response gains under the custom small-game rules. The η comparison is paired by random seed, but three seeds are too few for a strong statistical claim. The heuristic's simple standard error does not account for all poker variance or adaptive policy uncertainty. One 100-hand run is not sufficient to rank Hold'em strength.

The tiny river experiment uses a five-card board, three private hands per seat, pot 8, effective stack 10, and one raise maximum. [River CFR](../experiments/river_cfr.json) and [value model](../experiments/river_value.json) results are exact for that abstracted game or measured against its 200-iteration reference. The 200-iteration labels still average 0.398 NashConv, so they are approximate. [Equity-feature](../experiments/river_value_equity.json), [180-board seed 53](../experiments/river_value_equity_180_seed53.json), and [end-to-end latency](../experiments/river_value_latency.json) are recorded separately. Network latency excludes model loading but includes range/hand feature construction. The value model has **not** been inserted into CFR; no strategy-quality-preservation claim follows from its MAE or speed.

The [reward-scale experiment](../experiments/leduc_nfsp_eta03_scaled20.json) changed only DQN reward targets and kept the three seeds, η, and episode horizon fixed. Two of three seeds improved; this does not establish a reliable default change. The [range-size experiment](../experiments/river_scale.json) compares different sampled games, so NashConv across sizes is not a controlled strength trend. `tracemalloc` adds CPU overhead and measures traced Python allocations, not total process memory.

The [Hold'em NFSP training record](../experiments/holdem_nfsp.json) includes model SHA-256 hashes and per-seed training times. The [fixed evaluation](../experiments/holdem_nfsp_fixed_eval.json) records hashes of evaluated models; these hashes were added as provenance metadata after the initial run, without changing scores. It and the [untrained control](../experiments/holdem_nfsp_untrained_eval.json) use the same 1,000 seeded deals per opponent, seat/button rotation, and reset PyTorch sampling seed. Results are separate seat-0 policies trained in independent self-play runs; the three seeds are not 3,000 independent policy samples. Standard errors describe hand-to-hand variation within each run and are not paired-difference confidence intervals. A higher win fraction alongside negative chip EV shows why wins alone are misleading. The [inference profile](../experiments/holdem_nfsp_profile.json) isolates one decision, excludes model loading and HTTP overhead, and uses a different dirty-tree revision than the older table rows.

The [hand-level diagnosis](../experiments/holdem_nfsp_diagnostic.json) and [faced-bet guard ablation](../experiments/nfsp_faced_bet_ablation.json) use different fixed deal suites, so their chip means should not be compared directly. The guard's timed decisions include policy selection and equity sampling in active games, whereas the isolated inference profile uses one preflop state and no opponent computation. No improvement or deployment decision follows from the guard's small point-estimate difference.

The [30,000-hand training record](../experiments/holdem_nfsp_30k.json) and [fixed evaluation](../experiments/holdem_nfsp_30k_fixed_eval.json) preserve the seed-11 longer-horizon test; its evaluation hash was likewise added afterward as factual provenance metadata. The same fixed evaluation seed, opponents, seats, and buttons were used for the original 10,000-hand policy, but per-hand paired returns were not saved, so no paired-difference interval is reported. One seed does not settle whether additional training is beneficial in general.

Full-hand Hold'em CFR exploitability, depth-cutoff strategy error, and CPU-versus-GPU speedup have **not been measured**. The existing `.cu` snippets are not benchmark evidence.

## Explicit-range river CFR analysis (2026-10-13)

`python3 -m experiments.river_variants --checkpoints 20 100 500 1000 2000` compared three synchronous regret updates on the same five-card board, two 3-hand ranges, 9 compatible joint deals, 8-chip dead pot, 10-chip effective stack, 2-chip minimum bet, and min/half-pot/pot/all-in targets with at most one raise. The exact finite-game best-response NashConv decreased at each recorded checkpoint:

| Update | 20 iterations | 100 | 500 | 1,000 | 2,000 | Train seconds at 2,000 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla CFR | 2.0366 | 0.7513 | 0.3018 | 0.2032 | 0.1385 | 7.52 |
| Synchronous CFR+ variant | 1.1958 | 0.3336 | 0.0827 | 0.0459 | 0.0184 | 7.81 |
| DCFR (α=1.5, β=0, γ=2) | 1.7983 | 0.4505 | 0.1036 | 0.0509 | 0.0200 | 8.88 |

NashConv is the sum of exact unilateral best-response gains in chips for this finite abstract game. At 2,000 iterations it remains nonzero for all updates; CFR+ and DCFR are still exploitable. On this one range pair, CFR+ had the lowest measured NashConv but was not faster than vanilla CFR. At the final checkpoint, the exact analysis report took 11.3–14.0 ms separately from training. Comparing each checkpoint's root action EVs under its own average continuation profile with the 2,000-iteration values, mean/max absolute differences were 0.0245/0.0951 chips for 500-iteration CFR, 0.0078/0.0146 for CFR+, and 0.0084/0.0187 for DCFR. These are policy-dependent values, not a fixed-policy prediction-error test. The action-EV path was tested against a separate brute-force traversal that forces each root action and enumerates every compatible deal; the solver also reports per-hand frequencies and action EVs. This is a convergence demonstration for one tiny restricted river game, not a reference result for full Hold'em.

Raw output records Python 3.10.6, Apple `arm64`, revision `32d3fb353cc5338ac83e3e13a602762f5e2612f0`, and a dirty working tree in [river_variants.json](../experiments/river_variants.json). Training-only timing excludes NashConv and action report calculation; wall-clock variation across runs was material, and this is one run rather than a performance distribution. Adapter end-to-end latency includes exact best-response reporting, while `search_latency_ms` isolates the search budget; both can exceed the requested budget due to atomic iterations and exact diagnostics. It is not yet an interactive latency claim.

The CPU adapter supports only an explicit, root-only river abstraction with supplied ranges and a matching action set. Live browser decisions still lack validated range inference; frequencies, EVs, and EV loss remain unavailable there. Prior river action histories, turn/flop/preflop states, neural continuation evaluation, and safe re-solving are unsupported.

The separate [five-position weighted-range suite](../experiments/river_range_suite.json) used seed 729, five generated public boards, three hands per seat, nonuniform positive range weights, nine compatible deals per position, and 500 CFR+ iterations. Exact NashConv was 0.00559, 0.10842, 0.10353, 0.20381, and 0.04657 chips (0.0028–0.1019 BB). Training ranged from 0.78 to 2.11 seconds; exact analysis took 4.9–12.6 ms. This is evidence that the restricted solver runs on several real card/range inputs and that convergence depends on the position. It is still only five generated cases with no overlap between any private hands, and it is not a held-out strategic generalization estimate.

The [10 ms adapter smoke](../experiments/river_adapter_budget.json) completed only 3 CFR+ iterations: search took 10.23 ms and the exact diagnostics brought total latency to 20.11 ms. NashConv was 5.432 chips and the acting-hand check EV gap was 1.255 chips. This confirms that a short time budget does not make the returned strategy useful; the API returns the residual exploitability so a caller can reject it. No browser advice is connected.

## River Analysis UI smoke benchmark (2026-10-13)

The dedicated trainer mode calls the same `analyze_river_position()` function tested by the server endpoint. The saved [raw request and response](../experiments/river_analysis_ui_baseline.json) use board `As Kh Qd 7c 2s`, hero hand `Jc Tc`, pot 20, stacks 100/100, a 2-chip big blind, and three weighted combos per player. CFR+ used 500 iterations, checking NashConv at 250 and 500. The exact restricted-game NashConv decreased from 0.828481 to 0.489393 chips. End-to-end solver latency was 2,722.8 ms in the saved run. Across three invocations latency ranged from 2,722.8 to 3,248.8 ms; this is not a latency distribution.

At 500 iterations the hero-hand frequencies were check 0.140%, bet-to-2 0.097%, bet-to-10 1.548%, bet-to-20 10.837%, and bet-to-100 87.377%. Corresponding conditional action EVs were 13.6805, 14.2651, 19.0477, 24.6360, and 26.8015 chips; EV losses versus the best root action were 13.1211, 12.5364, 7.7539, 2.1656, and 0 chips. The EV values are conditional on the entered hero hand, opponent combo weights, blockers, and the final average CFR+ continuation policy. NashConv is exact best response only in this supplied-range finite game and action tree; a 0.489-chip residual remains and does not support an equilibrium-quality claim.

The complete suite passed 53 tests in the final 5.827-second run (the prior 52-test run took 5.722 seconds). A local HTTP smoke returned 200 for both `GET /` and `POST /api/river-analysis`; the latest 20-iteration request returned five actions, normalized frequencies, action EVs/losses, and NashConv 10.0033 chips in 140.9 ms (an earlier call took 136.1 ms). This low-budget smoke also illustrates why the UI surfaces the residual NashConv instead of labeling the output as high quality. `node --check` on the embedded script, Python compileall, and `git diff --check` passed. Hardware for the saved 500-iteration sample was Apple arm64 / Darwin 23.5.0, Python runtime in this workspace; CUDA was not involved.

The form limits each range to eight entries and the backend supports only the first river action, no prior river actions, and a 2-chip big blind. User ranges must already reflect earlier betting. The interface exposes no outputs for unsupported positions and does not present the manual ranges as observed history.

## Solver Lab and public-card reference (2026-10-13)

The UI now defaults to Play & Learn; Solver Lab holds the explicit-range CFR+ controls. The earlier first-action restriction in the preceding section describes the E032 implementation and the older `TinyRiverCFRBackend`. The new lab route replays up to two legal abstract river actions and solves from the resulting hero decision. It still requires at most eight range combos per seat and a pot measured before river betting. The 13×13 matrix generated 169 cells and all 1,326 distinct two-card combos in the [Node editor smoke](../tests/range_editor_smoke.js); the default three-combo ranges serialized without JSON entry. A local HTTP smoke returned 200 for the page, matrix script, range catalog, new hand, first river analysis, and post-bet river analysis.

The reproducible command `python3 -m experiments.solver_lab_milestone` saved [raw requests and responses](../experiments/solver_lab_milestone_v2.json). All solves used the same three manually supplied combos per seat, board `As Kh Qd 7c 2s`, 20-chip pot before river betting, 100-chip effective stack, and 2-chip big blind. Measured Apple arm64 end-to-end times include CFR+ training and exact restricted-game best responses:

| River root state | Iterations | NashConv (chips) | Latency (ms) |
|---|---:|---:|---:|
| Hero acts first | 500 | 0.489393 | 2,689.5 |
| Hero checked; opponent bet to 10 | 500 | 0.850238 | 416.9 |
| Hero checked; opponent bet to 10 | 2,000 | 0.111001 | 1,626.1 |
| Opponent checked to hero | 500 | 0.452920 | 1,421.9 |

The 2,000-iteration post-bet run measured halfway NashConv 0.363542 at 1,000 iterations. Frequencies and conditional EVs are in the raw file. The post-action requests reused the original manual ranges without conditioning them on the observed check/bet, so these are implementation/convergence measurements for finite abstract games, not evidence of realistic post-action recommendations. None of the NashConv values is zero. Latency is one run per condition and does not establish a distribution.

`public_transition()` enumerated the 49 candidate turn card codes on the sampled flop in 3.049 ms, including blocked outcomes, and their probability sum was 1.0000000000000004. Tests checked fixed-hand flop/turn/river probabilities and a weighted-range turn sum. This validates chance and card-removal arithmetic for the tested cases; no turn/flop CFR search, earlier-street action EV benchmark, or human-range inference accuracy has been measured. The full Python suite passed 64 tests in 6.337 seconds, the Node matrix smoke passed, and Python compilation, JavaScript syntax, and diff checks passed. A real browser screenshot or interaction automation was unavailable in this environment; the HTTP and simulated DOM checks cover program behavior rather than visual layout.

## Exact-runout one-street flop/turn reference (2026-10-13)

`python3 -m experiments.one_street_baseline` saved the [raw result](../experiments/one_street_baseline.json). Each game used two private combos per seat, four compatible joint deals, an 8-chip pot, a 10-chip effective stack, a 2-chip big blind, and 200 synchronous CFR+ iterations. The solver allows betting only on the current street. If play reaches showdown, remaining public cards are enumerated exactly with **no later betting**. Timings are one Apple arm64 CPU run per street; equity precomputation, search, and exact finite-game best-response analysis are separate.

| Current street | Future runouts per private deal | Equity precomputation | CFR+ search | Exact analysis | NashConv in this one-street game |
|---|---:|---:|---:|---:|---:|
| Turn | 44 | 68.66 ms | 337.36 ms | 5.12 ms | 0.08849 chips |
| Flop | 990 | 1,535.41 ms | 339.60 ms | 4.73 ms | 0.06864 chips |

The flop root's example hero hand mixed 14.60% check and 76.37% all-in under this continuation rule; these are **not** full Hold'em flop frequencies. The turn terminal payoff was cross-checked against 44 independently scored river showdowns. A post-bet turn-root test exercises the same CFR+ interface after an earlier current-street action. A separate test replayed real engine public events through range inference, but used assumed likelihoods. The complete Python suite passed 69 tests. Neither earlier-street result validates a DeepStack-style multi-street search, neural cutoff, preflop strategy, or automatic live feedback. The next benchmark needs tiny-range full flop→turn→river continuation games and paired root action EVs.

## Shared postflop solver integration (2026-10-13)

`python3 -m experiments.shared_solver_milestone` saved the [raw configuration and result](../experiments/shared_solver_milestone.json) on Apple arm64 / Python 3.10.6 with the working tree marked dirty. The shared API uses the same chance-aware CFR recursion for river, turn, and the research-only flop path. Its river output matched the preserved river reference exactly at 50 CFR+ iterations on one four-deal game: maximum acting-hand frequency difference **0.0**, maximum conditional action-EV difference **0.0 chips**, and both NashConv values **1.229164 chips**. Shared end-to-end latency was **139.68 ms**. This is numerical parity on one fixture, not broad strategic validation.

The exact turn-to-river fixture had one private deal, an 8-chip pot, 4-chip effective stacks, 2-chip big blind, and min/all-in targets with one raise. River card chance was enumerated; river betting remained active. One run per variant at 100 iterations gave:

| Update | Exact restricted NashConv | End-to-end CPU latency |
|---|---:|---:|
| CFR | 0.187533 chips | 2,394.35 ms |
| CFR+ | 0.016962 chips | 2,403.65 ms |
| DCFR | 0.001604 chips | 2,419.60 ms |

The 10-iteration CFR+ turn profile had NashConv 1.407744 chips and took 629.71 ms. Its check and bet-to-2 conditional action EVs were −4.6656 and −6.1977 chips; the separate one-street no-future-betting reference produced −4.4045 and −5.8060 chips. Those strategies are trained in different finite games, so the EV differences are evidence that future betting changes the modeled decision, not a measured error against a common equilibrium. The three-variant result is one hand pair and board; it does not establish the best update across ranges. The app retains CFR+ as a measured river default while broader turn tests are pending.

One exact flop-to-river run with one private deal, 1-chip effective stacks, no raises, and **one** CFR+ iteration took **8,766.32 ms** and left **2.426894-chip** restricted NashConv. It failed the interactive latency and convergence gates, so Solver Lab does not expose flop frequencies. The API marks this path experimental and requires an explicit flag. Preflop has no blueprint or search path. The full Python suite passed **81 tests**; the 13×13 Node smoke, Python compilation, and diff checks passed. A local HTTP smoke served the app and returned restricted river and turn reports from `/api/solve`. Automatic Play & Learn advice remains unavailable because betting-history ranges are not validated; no live decision-quality result has been measured.

## Targeted stopping diagnostic (2026-10-14)

The advanced Solver Lab can optionally stop after an exact NashConv checkpoint meets a user-entered chip target. On one three-combo-per-seat river fixture with a 40-iteration cap, targets 0.25 and 0.05 chips were not met: both runs completed 40 iterations and returned 4.314712 chips NashConv (end-to-end 525.89 ms and 499.52 ms). Checkpoints were non-monotonic (10.003315 at 20, 9.193891 at 24, 5.729797 at 28, 5.797374 at 32, 6.341446 at 36, 4.314712 at 40). A loose 1,000,000-chip target stopped at the halfway checkpoint, iteration 20, with 10.003315 chips NashConv in 282.52 ms. This verifies target reporting and early stopping; it is not a quality target. Each time is one Apple arm64 CPU run and includes CFR search plus finite-game best-response checkpoints. No multi-seed timing distribution was collected. [Raw fixture and outputs](../experiments/nashconv_target_followup.json).

## Exact traversal cache and chance-sampling comparison (2026-10-14)

The profile of one pre-change exact flop iteration attributed most search time to repeated seven-card evaluation: 27,720 evaluator calls and 582,120 five-card subset evaluations. On separate cold, unprofiled runs of the same one-deal flop state, baseline end-to-end/search was 7,466/5,844 ms; canonical board/hand rank caching gave 3,381/1,458 ms; adding exact all-in runout collapse gave 2,261/784 ms. The root check/bet action EVs and 2.426894-chip NashConv stayed the same to reported precision. These are single sequential observations, subject to system load and cache effects; they show a concrete bottleneck reduction, not a general throughput ratio.

The reproducible [fixed-position experiment](../experiments/multistreet_scale.py) saved [raw results](../experiments/multistreet_scale.json). It used one private deal for flop and four for turn, an 8-chip pot, no raises, exact final best-response analysis, and a cold board/hand rank cache for each condition.

| Position and method | Iterations | End-to-end | Exact restricted NashConv |
|---|---:|---:|---:|
| Flop, exact CFR+ | 20 | 6,256.68 ms | 0.102053 chips |
| Flop, sampled CFR, seeds 17/23/41 | 1,000 each | 1,671–1,739 ms | 0.529049–0.534386 chips |
| Turn, exact CFR+ | 100 | 5,610.77 ms | 0.210419 chips |
| Turn, exact CFR+ deeper reference | 300 | 16,507.18 ms | 0.106631 chips |
| Turn, sampled CFR, seeds 17/23/41 | 1,000 each | 2,098–2,316 ms | 0.658756–0.754197 chips |

Maximum per-hand conditional action-EV delta against the deeper exact profile was 0.032338–0.080622 chips for sampled flop and 0.226397–0.517927 chips for sampled turn. The reference profiles still have nonzero NashConv, and different continuation policies contribute to these EV deltas. Sampling did not pass this quality comparison and is not used for app advice. The exact flop result remains too narrow for real-game feedback, despite the improved runtime.

## Additional exact turn boards (2026-10-14)

The [fixed-board suite](../experiments/turn_board_suite.json) reused the same four compatible private deals, 8-chip pot, 4-chip stacks, and no-raise abstraction. Boards were selected before solving. Each timing is one cold-rank-cache run and includes exact best response.

| Turn board | 100-iteration NashConv / latency | 300-iteration NashConv / latency | Largest modal-action loss under deeper profile | Largest conditional action-EV delta |
|---|---:|---:|---:|---:|
| `2c 7h Jd Qs` | 0.145911 chips / 5.518 s | 0.064895 chips / 16.646 s | 0.013981 chips | 0.234463 chips |
| `9c 6d 3s Th` | 0.006712 chips / 5.785 s | 0.000751 chips / 19.918 s | 0 chips | 0.002749 chips |
| Earlier `As Kh Qd 7c` fixture | 0.210419 chips / 5.611 s | 0.106631 chips / 16.507 s | 0.040842 chips | 0.157285 chips |

The modal-action metric evaluates the 100-iteration profile's most frequent move using the 300-iteration profile's root action EVs for each hero hand. It is a practical decision comparison under an approximate deeper continuation policy, not equilibrium action regret. Only three boards and one range/stack/action setting have been measured. The cost and narrow coverage keep automatic turn advice gated.

## One-click river demo (2026-10-14)

The [saved demo request and raw output](../experiments/demo_analysis_baseline.json) used board `As Kh Qd 7c 2s`, hero hand `8h 8s`, two equal-weight combos per seat, an 8-chip pot, 10-chip effective stacks, and 2,000 synchronous CFR+ iterations. One local CPU run took **3,689.11 ms end to end** and returned **0.080251-chip exact restricted-game NashConv** (0.040126 BB). The five hero-hand conditional action EVs were check −4.000022, bet-to-2 −6.001050, bet-to-4 −6.130421, bet-to-8 −4.315231, and bet-to-10 −3.955995 chips. The corresponding best-action gaps are in the raw response. These numbers are from the real solver, not UI constants; the position is a four-joint-deal finite abstraction with assumed priors.

The headless Chrome end-to-end run observed a separate **3,728.87 ms** solve through `/api/demo-analysis`, checked every displayed frequency and chip/BB EV against the HTTP response, selected two actions, confirmed one solver request, checked the 169-cell range grid and diagnostics, and resumed a playable hand. The [screenshot](../artifacts/solver_demo.png) records the rendered page after selecting bet-to-10. These two timings are single runs, not a latency distribution.

## Experimental saved-hand river review (2026-10-14)

The [seeded gameplay run](../experiments/live_review_baseline.json) used 100-chip stacks, 1/2 blinds, a passive check/call opponent, and seed 29. The saved hero decision was a check after the opponent checked the river `5s 4h Js Qh 2s` with hero `8s Qc`, four chips in the pot, and 98 chips per stack. The opt-in review used the observed hero hand against **three equally weighted illustrative opponent combinations** (`Ac Ad`, `Kd Kh`, `9h 9s`); these were not inferred from the opponent's actions. Its 1,000-iteration CFR+ run took **838.75 ms** and returned **0.00007175-chip exact NashConv inside that three-deal game**. The selected check had 99.9995% frequency, −0.666667-chip conditional EV, and zero estimated action loss. Bet sizes and other EVs are saved in the raw result. This one hand establishes request capture and solver plumbing, not range accuracy or meaningful player advice. The latest isolated Chrome demo rerun took 3,821.81 ms after a test-startup polling fix; the saved 3,728.87 ms was the earlier passing run.

A further headless Chrome run played the seeded hand using table controls, clicked its river review, and verified that the displayed selected-action loss, action EV, and restricted NashConv matched the `/api/review-decision` response. That run took **849.26 ms** for the review solve and **4,567.55 ms** for the separate demo solve. These browser timings are individual samples and should not be treated as p50/p95 latency.

## Four-street live pipeline diagnosis (2026-10-14)

The [seeded trace](../experiments/live_pipeline_diagnosis.json) and [diagnostic report](LIVE_SOLVER_DIAGNOSIS.md) use one complete 100-chip-stack hand with hero `8s Qc` and three disclosed, unconditioned opponent combinations. Preflop had no solver. Exact full flop rejected three deals before search. Exact full turn took **13,454.3 ms for one CFR+ iteration** and left **43.5924 chips restricted NashConv**. The same turn decision under the existing exact-showdown, no-future-betting cutoff took **466.7 ms for 500 iterations** and had **0.0002181-chip NashConv within that cutoff game**. Flop cutoff took **1,532.0 ms for 500 iterations** and had **0.0003649-chip cutoff-game NashConv**. River full restricted CFR+ took **850.3 ms for 1,000 iterations** and had **0.00007175-chip restricted NashConv**. These are one-run latencies, not speedup claims for the same game: full turn and cutoff turn have different continuation rules. The cutoff's root action EV error versus deeper multi-street solving is unmeasured.
