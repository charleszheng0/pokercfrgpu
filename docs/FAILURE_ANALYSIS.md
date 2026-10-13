# Failure analysis

## E001: two failed test expectations

The first rule suite reported 5/7 passing. The conservation test added `pot` to stacks after a terminal award, double-counting chips; terminal `stacks` already include the award while `pot` remains as a hand-history amount. The short-stack test expected a three-chip flop bet although the opponent had only one chip left. Correcting the assertions made 7/7 pass. No engine defect was demonstrated by these failures.

## E002: heuristic strength unproven

In 100 hands versus random, the heuristic lost 2.68 chips/hand with standard error 5.91. This is too noisy to infer that it is weaker or stronger. It estimates showdown equity against uniformly random cards, which can misprice decisions when future betting, ranges, and fold equity matter. This is a hypothesis for future diagnosis, not a measured root cause. The current fix attempt is only to retain fixed seeds and seat rotation; no threshold tuning has been performed.

## Open validation risks

- The independent Treys cross-check found no disagreement in 5,000 seeded comparisons, but it is statistical rather than exhaustive.
- The first cross-check invocation failed with a syntax error in the new script. After fixing the missing parenthesis, the intended comparison completed with zero mismatches.
- Random games cover many transitions but do not exhaust all betting histories or inspect every unusual short all-in case.
- Historical Kuhn and Leduc scripts were preserved. Their CFR convergence and NFSP strategic quality are not inferred from saved weights or local unit checks.
- The later exploratory river model has per-hand root-value labels, but no validated depth-cutoff counterfactual-value labels or CUDA benchmarks exist.

## NFSP early plateau and incomplete convergence

At 3,000 episodes, all three η=0.1 seeds had higher exact Leduc NashConv than their untrained policies: seed 11 rose 6.51→7.19, seed 23 6.49→6.57, and seed 37 6.50→7.43. At 20,000 episodes they improved to 3.81, 3.24, and 2.98, but remained far above the synchronous CFR reference's 0.14 at 500 iterations. A 100,000-episode η=0.1 run on seed 11 reached only 2.67. Its greedy Q policy measured 5.37 NashConv, which is expected to be vulnerable in an imperfect-information game but also suggests the learned best response is still poor.

**Hypothesis tested:** too few best-response episodes and supervised examples at η=0.1 slow the average policy. We changed only η to 0.3, held the three seeds and 20,000-episode horizon fixed, and observed final NashConv 2.54, 2.96, and 2.56. The mean moved from 3.34 to 2.68. This supports a local improvement at that horizon, but three seeds and one changed setting do not establish robust optimal tuning. Other plausible contributors include fixed exploration, small memories, Adam/learning-rate choice, and the custom game's payoff scale; these are untested hypotheses.

**Gate status:** Small-game NFSP shows reproducible learning, but its policy remains substantially exploitable. A separate restricted Hold'em NFSP learner has since been trained and evaluated; the Leduc result does not establish its strength. CUDA and integration of a validated neural continuation model remain unimplemented.

## First neural river values underperformed the shallow solver

On 20 held-out public boards, the raw-card value model had 4.28 chips MAE against 200-iteration river CFR labels, worse than a zero-value predictor at 3.95. This falsified the initial hypothesis that a small raw-card MLP and 60 training boards were sufficient. Shallow 20-iteration CFR was much closer at 0.96 chips MAE.

**Root-cause hypothesis and controlled test.** River hand rank relative to the opponent's range is difficult to infer from sparse one-hot cards at this data scale. Adding one exact showdown-equity feature, with the same saved positions and labels, reduced MAE to 2.59. Adding 120 training boards while preserving the same held-out boards moved seed-53 MAE to 2.35; two other initializations gave 2.23 and 2.07. These results support the hypothesis that feature and data scarcity contribute, but do not isolate every cause. The labels themselves are approximate (reference NashConv mean 0.398).

**Decision.** Keep this network as an experimental artifact. Although its end-to-end path was about 40.5× faster than a 20-iteration solve on one tiny position, its error remains substantially larger. It is not used for live action EV or depth cutoffs. Whether it preserves strategic choices has not been tested.

### Follow-up audit: no payoff/range defect found; target does not match intended use

The historical 2.07–2.35-chip errors were MAE on 120 hand examples grouped into only 20 held-out public positions, using the same fixed split for all three network seeds. The six labels within each position share a board, ranges, and solve, so 120 is not an independent sample count. Each target is root hand-conditional EV under the average strategy of 200-iteration CFR. It is not a continuation value at a search cutoff. The 0.961-chip shallow-CFR number is the same conditional root value computed by 20-iteration CFR and compared with those 200-iteration labels; it is a direct solve of the known tiny game, not learned advice or action-regret evidence.

`experiments/diagnose_river_value.py` recomputed saved labels on four positions (maximum error 0) and checked 72 representative terminal payoff mappings against engine stack deltas under the root-pot baseline (maximum error 0). Joint deals exclude overlapping cards and normalize products of the supplied range weights. The encoded features cover the root board, private hand, explicit range hands and weights, player, pot, stack, and exact range equity; no public history is needed for this root-only target because all samples begin at the same root. No label, payoff, or range-weight implementation defect was found in this audit. A remaining limitation is that only a small sample of the saved targets was independently regenerated, and both generator and checker use the same CFR implementation.

The held-out public positions are disjoint from training boards and generated by the same sampler, but only 20 test positions support generalization claims. The full-card one-hot features expose suit identities even though suit permutations preserve poker values; a controlled four-mapping augmentation produced 2.201 MAE versus 2.134 without augmentation (RMSE 2.881 vs 2.915) on the same held-out boards. This did not improve the chosen primary metric and is not treated as a successful fix.

**Root cause of the failure to support DeepStack-style search:** the target and model input do not represent a search cutoff. They omit prior public betting history, current actor/action history, and a range-belief state at that cutoff. A model can fit a root EV target and still give invalid action values after a bet or raise. No downstream action recommendation metric was measured, so the current model has not demonstrated strategic value. The justified next step is to generate and validate actual cutoff counterfactual-value targets and report action regret against a deeper solve. The current neural river artifact remains experimental and is not a live analysis engine.

## Reward scaling gave mixed NFSP results

Dividing the DQN target reward by 20 reduced mean 20,000-episode NashConv from 2.68 to 2.49 across three paired seeds at η=0.3. One seed worsened (2.54→2.69), while two improved (2.96→2.62 and 2.56→2.16). This does not isolate reward magnitude as a consistent root cause. The default remains scale 1; the option supports later replicated tests.

## Restricted Hold'em NFSP loses chips to the heuristic

Three 10,000-hand self-play-trained average policies each played 1,000 fixed seeded hands against random and 32-sample heuristic agents. All gained chips against random (+0.86 to +1.48 per hand), and each exceeded its initialized-policy control by point estimate. All lost to the heuristic (−1.04 to −2.08 chips per hand). The initialized controls also lost to it (about −0.9), but seeds 23 and 37 got worse after training by point estimate. The trained policies still won 58.6–58.9% of all heuristic hands, so win rate alone hid the chip losses.

**Follow-up observation.** A separate 500-hand fixed-deal diagnostic for seed 11 found 191 trained-policy losses of at least 10 chips versus 148 for its initialized control. The trained policy folded 14 times versus 80, called 163 times versus 80, and raised 342 times versus 244. It went all-in less often (215 versus 286 hands), so excessive all-in frequency alone is not the explanation. Hands containing a trained-policy call averaged −9.21 chips, but that conditional result is selection-biased: calls occur in more threatening states.

**Failed fix attempt.** Replacing only faced-bet NFSP decisions with the existing 32-sample heuristic on 500 fixed deals reduced raises from 300 to 77 and increased folds from 9 to 35, but changed return from −0.708 ± 0.636 to −0.758 ± 0.553 chips/hand. Mean policy-decision latency rose from 1.05 to 5.51 ms. This narrow substitution did not support the hypothesis that a generic equity guard would repair the losses; it was not deployed to the browser. The difference is too small relative to hand variance to rank the policies confidently.

**Root-cause status.** Faced-bet decision quality remains an unproven hypothesis. Reward scaling by 20 and legal masking are already present, so neither can simply be asserted as the fix. Next, inspect range and hand-strength calibration by street or test a controlled training schedule change against the same fixed suite. No strength or equilibrium claim is supported yet.

## Longer Hold'em NFSP training did not improve the fixed suite

For seed 11, changing only the training horizon from 10,000 to 30,000 self-play hands moved the 1,000-hand fixed-suite point estimates from +1.078 to +0.817 chips/hand against random and from −1.037 to −1.505 against heuristic. The 30,000-hand checkpoint is saved separately; the playable opponent still uses the 10,000-hand policy. This one-seed result does not prove degradation, but it gives no support for the hypothesis that simply training longer resolves the current weakness. Compare multiple seeds or a targeted change to exploration, replay, or opponent diversity before selecting a new default.

## E026: playable engine and browser edge-case failures

The first regression assertions for odd-chip ties and short-blind conservation were wrong: the tie test left the constructor's 100-chip initial-stack baseline while replacing its stacks with 10-chip values, and the short-blind test added the displayed terminal pot to stacks after that pot had already been awarded. The tests were corrected to use consistent initial stacks and to distinguish in-progress chip conservation (`stacks + pot`) from terminal conservation (`stacks`). The engine's tie payout and short-blind refund then passed.

A table test that expected showdown used a random opponent and could terminate by fold. It now uses a passive deterministic test agent. During that regression, terminal-card privacy exposed a real bug: `winner is None` means a tied showdown as well as the initial/no-winner state, so it cannot distinguish folds from showdowns. Added `game.showdown`, which is set only by `_showdown`; the browser now reveals opponent cards for wins and ties at showdown and keeps them hidden after folds. The 43-test suite passed after the fix.

The first end-to-end HTTP smoke attempt failed an assertion in its completion checks. A traced rerun verified the returned action history and showed the heuristic can terminate a hand through fold as well as showdown; the final smoke used a complete legal action loop and passed through showdown and the alternating next hand. The initial assertion did not isolate an additional product defect. The showdown-vs-fold disclosure bug described above was an application defect and is fixed.

The underlying engine limitation that preset actions were also treated as the complete legal action set is resolved: arbitrary integer targets are checked against minimum full raise, effective stack, and raise-reopening rules. Short stacks below the big blind and odd-chip ties are now covered. Still open: carry-over bankroll semantics, tournament elimination, and multiway side pots are outside the current heads-up per-hand trainer.
