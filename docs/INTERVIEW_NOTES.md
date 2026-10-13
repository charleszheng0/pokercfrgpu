# Interview notes: evidence available so far

**Why a CPU rules reference first?** Search and self-play can silently learn from illegal state transitions or leaked cards. The reference engine makes betting and observation semantics explicit and can later serve as the CPU comparison for optimized operations.

**Why separate NFSP and CFR?** NFSP learns an approximate best response and average policy from experience; CFR minimizes counterfactual regret over information sets. The historical code contains both paths, but no evidence yet shows one improves the other or that either yields strong Hold'em play.

**Hardest current rules detail.** Heads-up blinds differ from postflop order: after the small blind calls preflop, the big blind still has an option. Effective stack caps also prevent a bet that the opponent cannot match. Both are tested explicitly.

**What improved measurably?** The first rule suite moved from 5/7 to 7/7 after correcting two mistaken expectations. This did not measure playing strength. The first heuristic benchmark was inconclusive (−2.68 ± 5.91 standard-error chips/hand versus random), so no strategic improvement is claimed.

**What remains?** Independent hand-evaluator comparison, exhaustive small-game CFR and NFSP validation, range-based continuation values, a full trainer UI, CPU profiling, and hardware-dependent CUDA comparisons. DeepStack, Libratus, and ReBeL results and guarantees are not reproduced.
