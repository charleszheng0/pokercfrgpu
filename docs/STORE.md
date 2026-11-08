# Training database (`data/spots.db`)

One SQLite file. No server. The trainer reads it; `pokerbatch` writes it.

## Why SQLite

| Option | Decision | Reason |
|---|---|---|
| SQLite | Yes | One file, no server process, built into macOS, C API, SQL queries by category and score, safe concurrent read while a batch writes (WAL mode). |
| JSON files | Only on the wire | Good for one node at a time (`/api/node`, `/api/play/spot`). A full turn tree as JSON would be hundreds of MB. |
| MongoDB / Postgres | No | They need a server. A single-user, offline trainer gains nothing. Revisit only for an online multi-user version. |

## Stability rule

Rows never hold the solver's internal layout (array offsets, representative cards, 16-bit
codes). They hold text that any future solver version can reproduce:

- hands as combo text (`AhKd`), with the suit symmetry already undone;
- action lines as text (`OOP Check > IP Bet 1.82`);
- the full spot settings (`config`), the ranges, the solver version (git commit) and the
  exploitability.

So the suit isomorphism, the 16-bit storage and a future CUDA kernel cannot break stored data.
`meta.format_version` is `2`: the node is a zlib blob (`"Z1"`, 4-byte length, data) of the
node JSON without the 13x13 grid (the trainer does not use it). Format-1 rows (plain JSON
text) still read correctly; `pokerbatch --count 0 --compact 1` converts them and runs VACUUM.
Size: about 6.5 KB per spot (30 KB in format 1).

## Tables

`solves`: one row per solve, also the rejected ones (gate failed).
`preset, oop_pos, ip_pos, board, street, pot, stack, oop_range, ip_range, config, solver_version,
iterations, seconds, exploitability_pct, ev_oop, ev_ip, memory_mb, status ('ok' | 'rejected')`.

`spots`: one row per decision point of a solve that passed the gate.
`solve_id, path, line, actor, actor_pos, num_actions, combos, mixed, ev_loss, bluff, fundamental,
category, score, story (JSON array of strings), node (the node JSON, as /api/node returns it)`.

Stored decision points: every decision node of the solve's first street, breadth first, up to
`--depth` actions deep (default 4): the first action, the reply to a check or a bet, facing a
raise, facing a re-raise. Lines through an all-in are not followed. About 10-15 spots per solve.

Duplicates: `solves.board_key` = `canonical_board_key(board)`: boards that differ only by suit
names (or card order) get the same key. With suit-symmetric ranges (all presets are) they have
the same solution. `pokerbatch` skips keys that the database already has for that preset and
street, and removes later copies at start.

## Model limits (say these to users)

- Ranges are hand-built "house ranges" (docs/research.md). Solutions are exact for these
  ranges, not for the real preflop equilibrium.
- River spots assume the flop and the turn were checked through; turn spots assume the flop
  was checked through. The story shown to the player says so.
- Bet sizes: 33% and 75% of the pot, raises 60%, all-in. Max 3 raises per street.

## Scores (weighted by each hand's reach)

| Score | Definition |
|---|---|
| `mixed` | Share of hands whose most frequent action is below 70%. |
| `ev_loss` | Mean EV gap between the best and the second-best action, in % of the pot. |
| `bluff` | Share of the bet/raise frequency that comes from weak hands. Weak = the passive action (check or call) wins less than 15% of the pot. |
| `fundamental` | Share of hands with a near-pure strategy (top action at least 90%). |

Category, checked in this order: `bluff` (aggression at least 15% and bluff at least 0.3),
`mixed` (at least 0.35), `ev_loss` (at least 5% of the pot), else `fundamental`.
These thresholds are first guesses from the first 90 spots. Tune them with real data.

## Play API

- `GET /api/play/spot?category=mixed|ev_loss|bluff|fundamental|any&exclude=1,2,3` returns one
  random spot: `{id, preset, street, board, pot, stack, oop_pos, ip_pos, story, path, line,
  category, scores, exploitability_pct, node}`. 404 with `{"error": ...}` when nothing matches.
- `GET /api/play/stats` returns `{spots, solves, solves_ok, mean_exploitability_pct, by_category}`.

## How big is "everything"? (measured 2026-11-08, Apple M2, idle)

| | Distinct boards | x 6 presets | Time per solve | Total time | Disk (6.5 KB/spot) |
|---|---|---|---|---|---|
| River | 134,459 | 806,754 | 0.13 s | about 29 hours | about 79 GB |
| Turn | 16,432 | 98,592 | 1 s (3-bet pots) to 40 s (SRP), mean about 24 s | about 27 days | about 8 GB |

So full coverage does not fit this laptop's disk (river) or a few nights (turn). A trainer does
not need it: one night of `scripts/train.sh` (8 hours) gives about 5,800 river and 1,200 turn
solves, about 100,000 spots and about 0.7 GB, spread over all textures by random sampling.
`--max_db_gb` (default 4) stops the run before the disk fills.

"Everything" here means this model: flop (and turn) checked through. Lines with earlier bets
change the ranges and need flop solves (1,755 distinct flops per preset), which is the next
step after the memory work.

## Fill it

```bash
scripts/train.sh                 # rounds of 30 river + 6 turn spots until stopped
./build/pokerbatch --db data/spots.db --count 50 --street river --seed 1
```
