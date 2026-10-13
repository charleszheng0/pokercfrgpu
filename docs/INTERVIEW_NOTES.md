# Interview notes: evidence available so far

**Why a CPU rules reference first?** Search and self-play can silently learn from illegal state transitions or leaked cards. The reference engine makes betting and observation semantics explicit and can later serve as the CPU comparison for optimized operations.

**Why separate NFSP and CFR?** NFSP learns an approximate best response and average policy from experience; CFR minimizes counterfactual regret over information sets. The project has a new full-hand restricted Hold'em NFSP policy and a separate tiny river CFR solver. No evidence shows one improves the other or that either yields strong Hold'em play.

**Hardest current rules detail.** Heads-up blinds differ from postflop order: after the small blind calls preflop, the big blind still has an option. Effective stack caps also prevent a bet that the opponent cannot match. Both are tested explicitly.

**What improved measurably?** Exact Kuhn CFR NashConv fell from 0.051 at 100 iterations to 0.0046 at 10,000. Synchronous Leduc CFR fell from 1.58 at 20 to 0.14 at 500 iterations. NFSP η=0.3 reduced mean Leduc NashConv at 20,000 episodes from 3.34 to 2.68 across three paired seeds; this is a limited local improvement. Three separately trained Hold'em NFSP policies beat their initialized controls by point estimate against random play on a 1,000-hand fixed suite, but all lost chips to the heuristic.

**Where is the current bottleneck?** cProfile attributed nearly all measured Leduc CFR time to recursive Python traversal and its state/value work. A 32-sample heuristic decision cost about 14 ms. The saved Hold'em NFSP policy took 0.129 ms median to encode and choose one action on a preflop state. This points CPU work toward tree representation and batching before any GPU kernel selection.

**What did the first neural search-cost experiment show?** On a nine-deal river abstraction, deeper CPU CFR labels took about 0.75 s per sampled position. An equity-aware network produced six hand values in about 1.73 ms end to end versus 70.3 ms for a shallow 20-iteration solve, but its 2.07–2.35-chip held-out MAE with 180 training boards was still well above shallow CFR's 0.96. Speed alone did not justify replacing the solver.

**What remains?** Expansion beyond the tiny river range, true public-state continuation labels, downstream depth-cutoff decision testing, diagnosing the trained Hold'em policy's heuristic losses, solver EV feedback, stronger NFSP convergence, broader poker-quality evaluations, and hardware-dependent CUDA comparisons. DeepStack, Libratus, and ReBeL results and guarantees are not reproduced.
