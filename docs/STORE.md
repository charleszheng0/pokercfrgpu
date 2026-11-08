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
`meta.format_version` is `1`. A change to the row meaning increases it.

## Tables

`solves`: one row per solve, also the rejected ones (gate failed).
`preset, oop_pos, ip_pos, board, street, pot, stack, oop_range, ip_range, config, solver_version,
iterations, seconds, exploitability_pct, ev_oop, ev_ip, memory_mb, status ('ok' | 'rejected')`.

`spots`: one row per decision point of a solve that passed the gate.
`solve_id, path, line, actor, actor_pos, num_actions, combos, mixed, ev_loss, bluff, fundamental,
category, score, story (JSON array of strings), node (the node JSON, as /api/node returns it)`.

Stored decision points (first street of the solve): the OOP root, IP after a check, IP facing
each OOP bet, OOP facing each IP bet after a check. All-in nodes are not stored.

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

## Fill it

```bash
scripts/train.sh                 # rounds of 30 river + 6 turn spots until stopped
./build/pokerbatch --db data/spots.db --count 50 --street river --seed 1
```
