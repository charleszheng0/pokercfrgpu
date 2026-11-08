# CLAUDE.md

Read SPEC.md first. It is the contract: scope, acceptance tests, limits.

## Rules
- C++17 for solver code (CUDA later in `cuda/`). No Python in the product.
- Keep the flat layout: nodes in one array, regrets and strategy sums in two float arenas.
- Every change to terminal values or the traversal must keep `./build/tests` green,
  especially `[0]` (fast vs slow terminal math) and `[3]` (toy game vs theory).
- Report exploitability in % of pot. Do not claim speed without a measured number.
- UI style: ASCII, Undertale-like (black, white mono text, heart cursor, block bars).

## Commands
- Build: `cmake -S . -B build && cmake --build build -j8`
- Tests: `./build/tests` (or `./build/tests --quick`)
- UI: `./build/pokerlab 8080`

## Out of scope for v1
Preflop, multiway, neural networks, NFSP, sampling CFR.
