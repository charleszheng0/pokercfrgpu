# External poker solver research and project build plan

**Reviewed:** 2026-10-20
**Scope:** Public repositories whose source or documentation explains how poker action frequencies and EV are produced. GitHub star counts below are snapshots from GitHub pages reviewed on the date above; stars measure interest, not solver correctness. The repositories are not all directly comparable: OpenSpiel is a general game framework, while the postflop projects solve particular restricted poker games.

## The shared computation behind EV and frequency

The credible pattern is: define a complete finite game, solve that game, then read the outputs from its strategy and value tables.

```text
Scenario input
  ├─ public state: board, pot, effective stack, actor, position, prior actions
  ├─ private beliefs: weighted combos for both players, blocker-aware
  ├─ legal tree: allowed sizes, raise cap, all-in rule, exact turn/chance rules
  └─ terminal utility: fold payoff or showdown payoff in a documented chip convention
          ↓
Finite extensive-form game / information sets
          ↓
CFR-family iterations
  ├─ current policy: regret matching on positive cumulative regrets
  ├─ regret update: action counterfactual value minus node counterfactual value
  └─ average policy: accumulated realization-weighted strategy, normalized per hand
          ↓
Evaluation
  ├─ strategy frequency by combo and information set
  ├─ conditional value by hand/action (must divide out range/reach mass correctly)
  ├─ range-average EV under supplied range weights
  └─ best response / NashConv for this finite game, where exactly computable
          ↓
UI and cache
  ├─ show frequencies and signed action EVs with units and assumptions
  └─ reuse only for a matching full state + ranges + abstraction + solver version
```

This resolves the project's recurring “missing information” problem. CFR cannot infer a player's private range or the betting policy that produced it. The engine can solve after receiving a belief model, but no algebraic tweak to regret matching can create those missing beliefs. The path forward is to either (a) make a restricted scenario fully specified by the user/preset or (b) make a range inference model explicit, calibrate it on held-out behavior, and label the inference. A low NashConv is a certificate about the finite game passed to CFR, not arbitrary no-limit Hold'em.

## Repository findings

| Repository | GitHub stars at review | What its source/docs establish | Relevance and limits |
|---|---:|---|---|
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | 5.5k | General game framework; ships CFR/CFR+ solvers, universal poker/game definitions, average policies, and exploitability evaluation. Its example loads a custom limited Hold'em game definition and prints exploitability while iterating. | Best independent reference for Kuhn/Leduc and small abstract games, plus game API concepts. It is not a ready full no-limit postflop product. Use it as an independent correctness oracle, not as the product solver. |
| [TexasSolver](https://github.com/bupticybee/TexasSolver) | 2.6k | Open C++ source for tree construction, per-hand/action regret storage, discounted CFR variants, strategy extraction, showdown utility, and best response. Input requires a defined game tree/range/state. | Strong source study for no-limit tree abstractions and flat strategy/regret arrays. Its AGPL licensing and code structure mean we should not copy code into this project without a deliberate license review. |
| [WASM Postflop](https://github.com/b-inary/wasm-postflop) and its [solver library](https://github.com/b-inary/postflop-solver) | 600 (web app) | Rust DCFR postflop engine; exact range hand combinations; merges isomorphic turn/river chance outcomes; optional 16-bit storage compression; Rayon multithreading; SIMD-oriented hot loops. README gives a reproducible comparison setup and reports strategy/equity/EV comparisons against commercial solvers. Development is suspended. | Most useful architecture for later turn/river scaling. It still solves a restricted postflop tree and explicitly pays memory/runtime costs. Its AGPL-3.0 license is a constraint. Its benchmark table is the authors' experiment, not independently reproduced here. |
| [Desktop Postflop](https://github.com/b-inary/desktop-postflop) | 352 | Desktop interface around the same Rust engine; editable ranges, tree setup, solving and per-hand strategy inspection. | Useful UX model; same solver/license limitations as above. |
| [noambrown/poker_solver](https://github.com/noambrown/poker_solver) | 170 | Recent research implementation explicitly splits Python reference and C++ optimized river solver; includes CFR, CFR+, external-sampling MCCFR, fictitious play and DCFR, plus Kuhn/Leduc validation. JSON subgame inputs include board, pot, stack, bet sizes, all-in flag, raise cap, and weighted hand combos for both players. | Very close to this project's current river milestone and a useful clean comparison implementation. It is still river-specific, and should be treated as a reference rather than a dependency until quality and licensing are reviewed. |
| [TexasSolverGPU](https://github.com/bupticybee/TexasSolverGPU) | 114 | Public repo explicitly says it is a Windows/NVIDIA binary-distribution repository and excludes the private `gpu_solver` source tree. The viewer is source, but the solver, build system and kernels are private. | Evidence that GPU acceleration can be productized, not evidence about which algorithm or kernels are used. The public v0.3.0 release cannot establish its internal regret updates. Do not infer “GPU CFR” internals from speed claims. The older [CPU TexasSolver source](https://github.com/bupticybee/TexasSolver) is auditable separately. |
| [Shark 2.0](https://github.com/24parida/shark-2.0) | 94 | Open C++ HU solver with a deliberately small action menu (flop 50/100% bets and 100% raises; turn/river 33/66/100% bets and 50/100% raises), explicit tree/position/range/stack inputs, and a GUI that prioritizes a low-clutter workflow. | Good reference for product simplification and action abstraction. Its modest stars and promotional claims are not quality validation; independently compare its results if used as a reference. |

### What these systems actually call “EV” and “frequency”

- **Frequency** is the average strategy's probability for an action at a private-hand information set. It is not the probability that a sampled game happened to take that action, and it is not a DQN/NFSP policy probability unless specifically labeled as such.
- **Action EV** is computed by evaluating the action's continuation under a defined opponent strategy/belief and terminal payoff convention. In a two-player CFR tree, raw counterfactual values are reach-weighted. To display conditional chips EV for a hand or range, divide by the relevant own/opponent/chance reach mass exactly once. TexasSolver and the Rust engine retain hand-indexed value/strategy vectors; the UI reads those vectors after solving.
- **NashConv/exploitability** requires best-response evaluation in the same game tree. It does not validate a range assumption or the omitted actions/streets.
- **Action abstraction** is a first-class input. Restricting sizes makes a smaller solvable game but changes its equilibrium. It is a modeling choice to report, not a hidden implementation detail.
- **GPU speed** comes after the game representation and update are stable: flatten/batch per-hand vectors, keep data resident, reduce launch/synchronization overhead, and parallelize suitable hand/action updates. A private binary alone is not enough to reproduce the algorithm.

## Project diagnosis based on current code

The current project already has the important mathematical skeleton: a typed `SolveRequest`/`SolveResult` in `poker_cpu/solve.py`, weighted combo ranges and blocker-compatible deals in `poker_cpu/river_solver.py`, an exact-chance postflop tree in `poker_cpu/multi_street.py`, a no-later-betting flop/turn reference in `poker_cpu/one_street_solver.py`, `/api/solve` and `/api/river-analysis` routing in `trainer/server.py`, and explicit unsupported-state handling in Play & Learn.

The hard blockers are scope and beliefs, rather than an observed proof that CFR regret matching is fundamentally broken:

1. **Preflop is not represented** by the postflop tree. There is no decision model for blinds, open/3-bet/4-bet options, or a verified blueprint, so preflop live analysis must stay unavailable.
2. **Ordinary live hands lack credible opponent ranges.** `request_from_game()` intentionally yields no ranges unless a source is supplied. The current range update arithmetic is not backed by a fitted, held-out-calibrated action-likelihood model. The UI correctly declines to label such results solver advice.
3. **Flop/turn cost grows with ranges, chance outcomes and future action branches.** Exact tiny-range multi-street code exists, but its Python combinatorics do not support broad ranges under interactive latency. One-street cutoff runs omit later betting, so their EVs are not full-street continuation values.
4. **A cached root solve is not a policy blueprint.** It can only be reused when the public state, ranges, action abstraction, payoff convention and solver version match. Random pre-solved river roots will almost never match a hand reached from preflop gameplay.
5. **The worker is useful as a bounded experiment, not automatic completion.** `trainer/cfr_scenario_worker.py` solves small sampled river games and only qualifies its exact restricted-game NashConv threshold. It neither constructs preflop-to-river strategies nor infers live ranges.

The offline NFSP worker is a separate scheduled system. It must remain separately reported: its self-play checkpoints do not become CFR solutions, and CFR cache items do not become NFSP policy training data unless an explicit dataset/policy-distillation experiment is implemented and evaluated.

### Low-priority CFR worker snapshot

At the time of this report, the already-running river scenario worker (seed stream `20261021`, 1,000 CFR+ iterations per request, four compatible joint deals, 0.1-chip restricted-NashConv acceptance threshold, 15-second pause, OS nice level 15) was still active. A read of `artifacts/cfr_scenario_cache/` found 46 qualified scenario records. Across those saved records, exact restricted-game NashConv ranged from 0.000113 to 0.074130 chips (mean 0.03242); solve latency ranged from 0.791 to 3.406 seconds (median 2.272 seconds). This is a small-range river workload only. It does not establish strategy accuracy for broad ranges, earlier streets or live histories, and the cache is not wired as a full-hand blueprint. The active worker will continue under its existing bounded run settings; these numbers are a point-in-time snapshot, not a promised morning total.

## A concrete build plan from the restricted game outward

This order turns every “unsupported” response into a specific schema/test/solver gap rather than an open-ended research task.

### Gate 1 — Freeze a canonical restricted-game contract

For each supported request persist a canonical JSON record containing: street/board; button, actor and positions; pot and effective stacks at the street boundary; contributions already made; complete public betting line; both weighted **combo** ranges; legal abstract bet/raise sizes including all-in rule; maximum raises; chip/BB payoff convention; CFR variant, iteration/time budget and seed. Validate cards, blockers, positive compatible joint mass, actor/action legality, and stack/pot conservation before constructing a game. Hash this record as the cache key.

For river, turn and flop separately add a minimal reference suite with: tiny ranges (2–8 combos at first), all legal root actions, a deep deterministic solve, exact best-response/NashConv where supported, independently enumerated conditional root action EVs, and chip-to-BB invariants. Save raw JSON results and code revision. Do not qualify from training iteration count alone; qualify from convergence and action-EV stability against the deeper run.

### Gate 2 — Make the river Scenario Lab dependable

1. Keep the consumer form to board, hero hand, pot, effective stack, position/actor, and a small set of named assumptions. Hide combo weights and solver settings under Advanced.
2. Internally expand a chosen range preset to concrete suit-aware combos; remove board/hero blockers; report remaining compatible combo count and source label.
3. Use CFR+ with a configurable target/budget; return per-combo average strategy, conditional action EV in chips and BB, range-average EV, exact restricted-game NashConv, iterations, latency and quality gate status.
4. Add deterministic regression fixtures and a cached result keyed by the complete canonical request. A selection change can re-render EV loss without re-solving; a request change cannot reuse stale output.
5. Put exact restrictions and range provenance beside the output in a compact disclosure. If invalid/unsupported, return a reason instead of partial numbers.

This is the first dependable deliverable because it asks no model to guess ranges, has no future card chance, and can be evaluated exhaustively on compact hand sets.

### Gate 3 — Use solved roots for a restricted playable mode

Do not promise arbitrary hands. Build a “study deal” mode that samples from an explicit collection of canonical solved river scenario families (or replays a fully known line that terminates at one). Verify the exact cache key before showing advice. If the hand differs in board, pot, stack, history, range or abstraction, show “no matching solve.” This is how cached work can serve the game without pretending a sparse cache is a full-game strategy.

For actual hands, first use a **declared prior range model** keyed by position and line. Track a separate action likelihood table learned only from opt-in/local histories. Evaluate calibration on held-out decisions with log loss/Brier score and coverage; do not call ranges “inferred” as a quality claim until calibration passes. Then pass posterior combo weights into the same solver API. Store the range model version and posterior with the hand history.

### Gate 4 — Expand turn, then flop, by a controlled game abstraction

Reuse one canonical betting-state transition implementation shared with the game engine. Begin turn with tiny ranges and exact river-card enumeration, then add restricted betting on turn and river. Compare to exhaustive per-deal reference cases and report root-action EV error/regret versus deeper solve, not just NashConv of a truncated game. Add flop only after turn passes; represent turn/river chance with exact enumeration on tiny fixtures and validated sampling for larger cases. Cache hand ranks/equities and isomorphic suit structures. Profile before changing language or parallelism.

Neural continuation values come **after** this reference exists: sample actual nonterminal cutoff states and ranges; label hand-conditioned counterfactual values by deeper solves; ensure train/test split is by board and range family; compare shallow baseline vs neural at equal time; gate on root action EV error and selected-action regret. Reuse none of the previous root-value network as a cutoff evaluator without passing this test.

### Gate 5 — Preflop and live full-hand strategy are their own milestone

Preflop needs a full abstract betting game or a validated blueprint. It cannot be solved by supplying a 5-card board to the current tree. Define position, stacks, blind/ante, raise menu, limp/3-bet/4-bet caps and hand-combo abstraction; validate against OpenSpiel-compatible small variants or independent exact enumeration before integrating a 169-class UI. Until then preflop frequencies and postflop decisions reached from unsupported histories stay unavailable.

### Gate 6 — Optimize and consider GPU

First profile CPU tree construction, evaluator, terminal payoff, chance traversal, regret updates, average-strategy extraction, best-response/NashConv and serialization separately. Store per-node/per-combo action vectors contiguously and avoid allocating in traversal. Benchmark end-to-end solve and memory for the same canonical requests. Only then port measured kernels; compare CPU/GPU per-hand regrets, strategy sums, action EVs and NashConv within stated floating-point tolerances. The TexasSolverGPU public binary/release is not a source-level blueprint for this work.

## Suggested success criteria

These are proposed gates, not measured achievements:

- **Engine:** all current Hold'em tests plus solver/game transition parity and pot conservation fixtures pass.
- **River:** deterministic requests produce identical outputs for fixed seed/config; exact best responses available; root action EV error versus deeper solve below 0.05 BB and strategy convergence tracked across increasing budgets.
- **Turn/flop:** same metrics plus future-card chance correctness; compare selected-action regret vs deep reference, with a preregistered threshold before showing advice.
- **Live ranges:** held-out probability calibration and coverage are reported separately from solver convergence; no live advice until posterior range provenance is shown.
- **Cache:** exact request hash matches; cache hit reports solver revision and quality status; any mismatch recomputes or explicitly declines.
- **Performance:** report p50/p95 solve latency and peak memory per request family, with board, combo counts, tree nodes, bet sizes and hardware.
- **CUDA:** CPU reference retained; end-to-end speedup and numerical/decision quality both measured on identical inputs before claiming success.

## Reproducible source inspection

At review, the relevant source clones were inspected under `/tmp/poker-research/` (not copied into the project): `TexasSolver` commit from its checked-out `master`, `TexasSolverGPU` release repository, `shark-2.0`, `noambrown/poker_solver` commit `6a10442`, and `b-inary/postflop-solver` commit `9d1509f`. In the Rust engine, `src/solver.rs` computes DCFR discount parameters and alternates player updates, checking exploitability periodically; `src/interface.rs` defines the per-hand game interface and strategy storage. In noambrown's river solver, the README's JSON schema establishes the essential state inputs; `cpp/src/mccfr.h` stores regret and average-strategy arrays per player/node/hand; `cpp/src/river_game.h` models board, pot, stack, sizes, ranges and a finite betting tree.

No external repository was integrated into the application during this research pass. No performance number in this report is a benchmark of this project's solver. Public star counts and claims may change; the linked repository pages and source commits are the audit trail.
