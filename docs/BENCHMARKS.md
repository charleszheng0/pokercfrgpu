# Measured benchmarks

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
