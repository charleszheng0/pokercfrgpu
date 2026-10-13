# Implementation roadmap and gates

1. **CPU rules reference (current):** preserve existing work; implement Hold'em deal, action abstraction, blinds, all-ins, showdown, observations, random and heuristic bots. Gate: deterministic rule and seeded property tests pass, with independent evaluator comparison. Both checks now pass; further unusual betting cases still need coverage.
2. **Small games:** audit existing Kuhn/Leduc code without overwriting it. Build exact exploitability and game-value checks for Kuhn, then Leduc. Train NFSP with at least three seeds and fixed evaluation snapshots. Gate: CFR converges toward known value; NFSP shows reproducible learning and legal masked actions.
3. **Restricted Hold'em solving:** define a finite hand/range/action abstraction and generate CPU reference solves. Validate opponent/chance reach weighting and value vectors against tiny enumerated trees.
4. **Neural continuation model:** save public position plus two ranges to hand-indexed counterfactual value labels from deeper reference solves; split by public position, train PyTorch, report held-out error and downstream decision error.
5. **CPU experiment cycles:** profile search and inference, vary one factor at a time, compare fixed opponents with seat rotation and uncertainty. Gate: all rule tests, small-game validation, NFSP seeds, multiple Hold'em baselines, value-model held-out error, depth-limited versus deeper solves, and latency profile.
6. **Trainer:** add a richer table, hand histories, selectable policies, and honest estimated EV/latency displays from real models. The terminal table in milestone 1 is the functional starting point.
7. **CUDA only after gate 5:** if NVIDIA hardware becomes available, benchmark PyTorch CUDA against custom C++/CUDA for profiled bottlenecks, including transfers and end-to-end game quality. Otherwise retain CPU operation and label CUDA code untested.

Each experiment should save command/config, seed, revision and dirty status, hardware, raw output, and an interpretation distinct from observation. Do not promote prior `.pt` files to benchmark evidence without provenance.
