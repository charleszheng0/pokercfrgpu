# Solver Demo: gameplay and analysis coverage

**Date:** 2026-10-15. Solver Demo contains a calculated restricted-river study and three playable hands. The playable hands do not yet have validated CFR feedback at every decision.

## What works now

Open **Try Solver Demo** in Play & Learn to see the calculated restricted-river strategy. **Play three hands** starts three fixed, reproducible 20-chip heads-up Hold'em hands from the same demo. Cards are dealt by the normal 52-card engine, the button alternates, the opponent uses a deterministically seeded heuristic policy, and the human may choose any legal action. **Next demo hand** advances after a terminal hand. Starting the hands again replays the same first hand. The predefined scenarios are:

| Hand | Hero cards | Fixed board on a showdown line | Study theme |
|---|---|---|---|
| 1. Premium pair | `Ah Ad` | `As 7c 2d 9h 3s` | Value betting |
| 2. Drawing hand | `Jh Th` | `9h 8h 2s 3c Ah` | Continuing with a draw |
| 3. River pressure | `Kc Qs` | `Qh 7d 2s 9c 8d` | Marginal hand on a changing board |

`trainer/recruiter_demo.py` defines exact full decks, including burn cards. `trainer/server.py` exposes `/api/showcase/start` and `/api/showcase/next`; regular `/api/action` performs the same legal game transition as ordinary play. A browser test plays all three from preflop to showdown by taking check/call lines, verifies boards and rewards, and separately confirms a user-selected raise changes the branch. Folding legitimately ends a hand before showdown.

## Why CFR feedback is intentionally absent

The request requires a **precomputed, validated restricted-game CFR response at every reachable human decision**, not merely a few attractive example screens. That coverage is currently zero for the three new fixtures. The only validated cached result is the four-deal **river study inside Solver Demo**. Its board, ranges, stack, pot, and action history do not match any playable-hand decision; copying its frequencies or EVs would be mathematically false.

The shared solver currently rejects preflop outright. On flop and turn, its fast cutoff assumes no later betting, while the full continuation tree is capped and too slow for 20-chip multi-branch play. No fitted betting-history likelihood model supplies coherent beliefs after arbitrary user/opponent actions. Every alternate legal wager can create a new state, so solving one passive path would not satisfy the all-decision requirement. The playable hands therefore show a compact “No matching CFR solve” status, suppress the separate heuristic rollout EV control, and do not use NFSP probabilities as CFR frequencies. Ordinary post-action experimental reviews remain separate and opt-in.

## Exact release gate for the requested demo

1. Define the demo's **restricted but real** action and chance abstraction, including blinds and preflop BB option. Decide whether the user may choose every legal integer wager or only a disclosed set of cached actions. The engine can still accept arbitrary legal wagers; the demo must state when analysis lacks a matching key.
2. Implement and independently test preflop payoff/chance handling and compare action/root payoffs against `HoldemGame` on seeded cases. Do not treat a deterministic future board known to the solver as normal preflop poker.
3. Define documented starting ranges and update them after every public action and board reveal. Verify blockers, normalization, and posterior sensitivity. If beliefs are illustrative, say so at every solve.
4. Enumerate the reachable hero-decision tree under the demo opponent and allowed user actions. Save a cache entry keyed by **board, hero hand, pot, stacks, button, contributions, full action history, both weighted ranges, abstraction, algorithm, and solver revision**. Reject missing or mismatched keys.
5. For each entry, record iterations, exact restricted-game NashConv where tractable, root action EVs under the same continuation profile, and latency. Run deeper-reference and independent payoff checks on a held-out subset. A low NashConv in a no-later-betting cutoff game does not validate full-hand action EV.
6. Only then reveal frequencies, signed action EV, and EV loss in the playable hands. Add a browser test that traverses **multiple user branches in all three hands** and checks each displayed number against its matching saved solve. Report coverage as matched decisions / all reachable decisions, not as a count of cached files.

Until these gates are met, describe the product as **playable Hold'em plus a restricted-river CFR study**, not full-street GTO analysis. The restricted river study is the verified, user-visible strategy example.
