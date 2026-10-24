# Postflop Scenario Lab game definition

This is the declared game solved by the current Scenario Lab, not a claim of
unrestricted Hold'em GTO. The browser and `/api/solve` use the same CPU solver.

## Inputs and utility

- Heads-up cash-game **zero-rake** chip utility. No tournament ICM, ante,
  jackpot drop, or blind-payment history is reconstructed inside a postflop
  solve. EV is conditional on the supplied position; chips already committed
  before that decision are sunk. The UI reports both chips and BB at 2 chips/BB.
- The board may be any valid three-card flop, four-card turn, or five-card river.
  The current server checks the card count and selected hand, and the solver
  rejects duplicate/dead-card combinations.
- `pot` is the pot **before betting on the selected street**, not the displayed
  pot after a prior bet. `stacks` are the two chip budgets at that street's
  start. Earlier actions on that street are replayed through the same legal
  tree; no arbitrary off-tree wager is silently rounded.
- Postflop the BB/out-of-position seat acts first. The `first_actor` field
  identifies whether that seat is the entered hero or opponent. A supplied
  betting history may put the other seat on turn to act.
- Both players have weighted, exact two-card combinations. Weights are
  **assumptions supplied at this root**. Board blockers and mutual hand
  collision are removed when joint deals are formed. Earlier betting does
  not automatically update the weights.

## Range assumptions

`poker_cpu/ranges.py` provides named **illustrative** BTN opening and BB
defending class-weight rules. They are not measured GTO charts, empirical
player frequencies, or calibrated postflop posteriors. They are too broad for
the current interactive solver's eight-combo-per-seat river limit. The Lab's
small river/turn example presets are separate teaching inputs. An exact hand
weight is a relative frequency among the supplied combinations; it does not
assert that all omitted hands are impossible in real poker.

We will not alter preset weights to match an unmatched online chart. A valid
external comparison requires the same preflop line, positions, board, pot,
stacks, rake, exact ranges, betting menu, and EV convention. Only then can
per-combo action frequency, conditional action EV, and policy regret be
compared. At present **no matched external TexasSolver/Shark/GTO Wizard output
has been imported**, so there is no measured distance to their strategies.

## Betting tree and current coverage

The compact `shark_2_0_inspired` menu has flop bets at 50%/100% pot and
one 100%-pot raise size; turn/river bets at 33%/66%/100% and raises at
50%/100%. All-in remains available and legal targets are stack capped. The
current tree permits at most one raise per street. These sizes follow the
published Shark 2.0 menu as inspiration, not a reproduced Shark solution:
<https://github.com/24parida/shark-2.0#how-to-use-shark>.

| Street | Current full-continuation support | Important limit |
|---|---|---|
| River | Multiple abstract decisions, up to eight entered combos per seat in the Lab | Eight-combo solves can take seconds; one range-weighted NashConv does not bound every hand's EV error. |
| Turn | Current-street betting, exact river card removal and river betting | At most four compatible joint deals and 100 exact CFR+ iterations in the interactive API. |
| Flop | Research tree includes turn and river betting | Not enabled in the Lab: exact full-tree search is capped at one deal/three iterations and has not passed the quality gate. The fast showdown cutoff **omits later betting**. |

All reported NashConv values are exact best-response residuals **inside the
specified finite game**. They do not measure action-abstraction or range error.
There is no neural continuation evaluator in the deployed solver.

## Persistent backend

The local `trainer.server` HTTP process owns `/api/solve` and writes each
successful Lab solve to `artifacts/scenario_solves.sqlite3` via
`trainer/scenario_db.py`. This is the project's backend; no separate cloud
service or database account is required. Set `POKER_SCENARIO_DB` to relocate
the file. SQLite WAL allows concurrent reads/writes from the local server and
offline worker. The table records the exact request, full result, source-code
fingerprint, street, measured NashConv, timestamp, and whether the solve met
its quality ceiling. The key includes all input fields and the solver source
fingerprint. **Candidate** results are retained but never returned as cache
hits; a changed board, range, budget, target, history, or solver implementation
requires a fresh solve. The default cache ceiling is 0.1 chip, tightened if
the request supplies a lower NashConv target. File-based legacy demo/worker
caches remain readable for compatibility.

`GET /api/scenario-cache/stats` reports recorded and qualified exact-request
counts by street. It does not estimate the fraction of real hands covered.

On an exact qualified hit, the UI shows a cache badge. Its original solve
latency and current retrieval latency are reported separately. Cache counts
are not gameplay coverage; a random solved board is unlikely to match a live
hand with identical ranges and action history.

## Expansion gates

1. Repeat held-out river tests over boards, weighted ranges, and action
   histories. Track exact restricted NashConv **and** selected-hand action-EV
   stability, reference-profile regret, p50/p95 wall time, and memory. Only
   widen the range cap when these remain acceptable.
2. Repeat the same checks for full turn-to-river betting with more than one
   compatible deal, including independently checked chance and payoff cases.
3. For flop, compare a faster approximation against a deeper full betting
   reference on the **same** game. The no-later-betting cutoff cannot be
   certified by its own low NashConv alone. Enable the Lab only when root
   decision quality and latency pass a declared gate.
4. Before applying results automatically to normal gameplay, fit and validate
   betting-history range likelihoods. Manual ranges remain valid for Lab use.

For source context, the original CFR paper establishes convergence in a
defined extensive-form game, while Shark documents explicit pot, stack,
board, range, bet-menu, iteration, and exploitability inputs:
<https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html>,
<https://github.com/24parida/shark-2.0#how-to-use-shark>.
