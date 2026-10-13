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
- No neural continuation labels or CUDA benchmarks exist for this milestone.
