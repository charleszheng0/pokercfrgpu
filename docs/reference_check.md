# Reference check: pokercfrgpu vs TexasSolver

Date: 2026-11-08. Machine: Apple M2, 8 GB. Verdict: **agree** (details below).

## Reference

- TexasSolver (bupticybee), branch `console`, commit `02c8021`, AGPL-3.0.
  Built only as an outside tool in a scratch folder. No TexasSolver code is in this repo.
- Build: CMake Release, OpenMP from Homebrew `libomp`, `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`,
  target `console_solver`.

## Matching the trees

| Setting | Ours | TexasSolver | How we matched |
|---|---|---|---|
| Units | any | bets rounded to whole chips | Spots scaled 10x: pot 100, stack 950 |
| Bet sizes | 33%, 75% of pot, all-in | same (`bet` and `donk` sizes) | same |
| Raise | to_call + 60% of pot after call | same formula, rounded | 133 vs 132.6 chips: negligible |
| Raise limit | `max_raises` | fixed at 4 | `--max_raises 4` |
| All-in threshold | added ≥ 0.67 × remaining stack | commit ≥ 0.67 × total stack | Checked by hand: both trees go all-in at the same points for these sizes |
| Ranges | `22+` style | explicit classes | Expanded our ranges to 52 / 58 classes |

Ranges: OOP `TT-22,AJs-A2s,KTs+,K9s-K6s,QTs+,Q9s,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo+,QTo+,JTo`,
IP `AA-22,A2s+,K7s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,A8o+,KTo+,QTo+,JTo`.

TexasSolver's dump holds strategies but no EVs, so the check compares strategies. Each tool
reports its own exploitability in % of the pot.

## Results

Range-wide action frequency, ours minus TexasSolver, in percentage points (largest absolute
difference over the actions of the node).

| Spot | Exploitability (ours / TS) | OOP root | IP after check | IP vs bet 33% | IP vs bet 75% |
|---|---|---|---|---|---|
| River Ks9d5c2h2d, loose | 0.233% / 0.249% | 3.1 | 3.7 | 1.8 | 0.1 |
| River, tight | 0.045% / 0.049% | **0.4** | **2.1** | **0.2** | **0.7** |
| Turn Ks9d5c2h, loose | 0.297% / 0.269% | 2.1 | 4.6 | — | 0.9 |

Per-combo strategies (river, tight): 69–85% of combos within 5 points of each other, 97–100%
within 25 points. The largest per-combo gaps are hands close to indifferent between two
actions (for example Kc6c at the root, QdTx after a check). Several different mixes of those
hands are all near-equilibrium, so per-combo gaps are expected; the range-wide numbers and the
exploitability are the meaningful comparison.

## Verdict

- Agree. When both solvers converge further (0.25% → 0.05%), the gap between them shrinks
  (root: 3.1 → 0.4 points). That is what two correct solvers of the same tree do.
- Facing-bet nodes agree within 1 point already at 0.25%.

## Speed (not a benchmark)

| Spot | Ours | TexasSolver |
|---|---|---|
| River, to ~0.25% | 0.24 s, 200 iterations | 0.58 s, 201 iterations |
| River, to ~0.05% | 1.9 s, 600 iterations | 701 iterations |
| Turn, to ~0.3% | 54 s, 250 iterations | 85 s, 251 iterations |

These runs shared the CPU with the training job and another benchmark, so treat the times as
rough. The controlled benchmark (idle machine, 8 threads, 3 runs each) is in
[benchmark-texassolver.md](benchmark-texassolver.md).

## Reproduce

Scratch scripts (not in the repo): TexasSolver input files built from the settings above,
and a short comparison script that matches combos and actions between the two outputs.
