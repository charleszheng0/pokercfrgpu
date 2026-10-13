# CPU quality gates before CUDA

These are forward-looking release criteria. A passing toy-game check does not substitute for a full Hold'em result.

| Gate | Evidence required | Current status |
|---|---|---|
| Hold'em rules | Deterministic rules, seeded conservation/legality tests, hidden-card isolation, independent hand-rank comparison | **Partial pass:** 30 combined tests currently pass; 5,000 Treys comparisons agree. Broader short all-in and exhaustive betting tests remain. |
| Kuhn CFR | Exact NashConv < 0.01 after 10,000 iterations and value within 0.005 of −1/18 | **Pass:** 0.0046 NashConv, value −0.055546. |
| Leduc CFR | Exact NashConv < 0.2 after 500 synchronous iterations | **Pass:** 0.1395. |
| NFSP small-game learning | At least three seeded runs reduce exact NashConv; a later strength target is < 1.0 for all three seeds | **Learning pass, strength pending:** η=0.3 reduced all three by 20,000 episodes, but finals were 2.54–2.96. |
| Restricted Hold'em search | Exact best-response check in at least one explicit range/action abstraction; agreement with engine action legality; then validation on larger ranges/public histories | **Tiny river pass only:** nine compatible deals, 0.3018 NashConv at 500 iterations, opening actions checked against engine. |
| Neural continuation values | Held-out board/range split; value MAE ≤0.25 BB, RMSE ≤0.5 BB, mean action regret ≤0.125 BB and p95 ≤0.5 BB against deeper reference; faster end-to-end than shallow CPU solve | **Fail:** current model predicts root hand EV, not continuation values; MAE is 2.07–2.35 chips versus 0.96 for shallow CFR. No downstream action error has been measured. Gates are prospective and defined in `docs/PROJECT_METRICS.md`. |
| Hold'em agent strength | Multiple fixed opponents, seat rotation, enough hands for useful confidence intervals, no reliance on training reward alone | **Partial, strength fails:** three 10,000-hand self-play-trained 20-chip NFSP policies each played 1,000 fixed hands versus random and heuristic, with initialized-policy controls. Trained returns versus random were +0.86 to +1.48 chips/hand; versus heuristic −1.04 to −2.08. A faced-bet guard and a 30,000-hand seed-11 run did not improve measured return. Diagnose losses before claiming a strong agent. |
| CPU performance | Profile search and inference, record end-to-end latency and memory | **Partial:** CPU operation profile, tiny river latency/range scaling, and saved Hold'em NFSP isolated inference (0.129 ms median complete action) measured; full Hold'em search and process-memory profiles pending. |
| River range scaling | Measure traversal and memory beyond the nine-deal toy | **Partial:** 3/6/10 hands per seat measured with `tracemalloc`; 100 iterations took 1.47/5.37/13.80 s. Larger realistic ranges remain untested. |
| CUDA eligibility | All preceding gates pass on a stable CPU reference; compatible NVIDIA hardware available for numerical and end-to-end tests | **Locked:** preceding gates fail and this Apple M2 host has no NVIDIA CUDA GPU. |

The next useful cycles are: diagnose the restricted Hold'em NFSP policy's losses against the heuristic; improve small-game NFSP vulnerability with controlled changes; expand river ranges and test belief updates; generate richer public-state value labels; compare model-assisted action choices against deeper reference solves. Keep CUDA archived until the gate is met.
