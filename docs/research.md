# Research: starting ranges and project differentiation

Date: 2026-11-08. Method: web search and page fetches. Some pages could not be fetched. I mark every gap. Several search hits were low-quality or spam pages. I did not use those.

# Part 1. Where do the starting ranges come from?

## What GTO Wizard does

- GTO Wizard says it solved its preflop trees in MonkerSolver. The tree allows calling in all situations. The run needed a very powerful server. Source: https://blog.gtowizard.com/status-and-info-about-our-solutions/
- Its cash solutions use the PokerStars Zoom rake, both preflop and postflop. NL500 is 5% with a cap of 0.6 bb per hand. Source: https://blog.gtowizard.com/all-you-need-to-know-about-our-solutions/
- Its "General" 6-max open sizes are about 2x (UTG, HJ), 2.3x (CO), 2.5x (BTN) and 3x (SB). Source: https://blog.gtowizard.com/status-and-info-about-our-solutions/
- The simplified cash set is 100 bb deep. Source: https://blog.gtowizard.com/simplified-solutions-and-a-new-interface/
- It says each flop is solved separately with no approximations, to 0.4% of pot for main spots. Source: https://blog.gtowizard.com/all-you-need-to-know-about-our-solutions/
- Not found: any page that explains exactly how the preflop output becomes the flop starting ranges. I could not verify the hand-off method. Its newer pages say preflop ICM ranges carry into the postflop tree. Source: https://blog.gtowizard.com/preflop-icm-solving/ (seen in search snippets only).
- Press reports say paid tiers now allow custom preflop solving with custom rake, stacks and nodelocking. These are trade-press claims, not GTO Wizard documentation. Source: https://www.pokerlistings.com/news/gto-wizard-rolls-out-custom-multiway-preflop-solving
- I could not fetch GTO Wizard's own terms of service (the URL I tried returned 404). I could not verify what they say about copying ranges. Assume copying and redistributing their charts is not allowed.

## What papers and tools do

- Libratus (Brown and Sandholm, 2017) solves the whole game with an abstract blueprint. It then re-solves subgames during play. It has no separate "range input" step. Sources: https://www.ijcai.org/proceedings/2017/0772.pdf and https://arxiv.org/pdf/1705.02955v3.pdf. This design does not help us, because our solver starts at the flop.
- Postflop solvers take ranges from the user. The postflop-solver library by b-inary is scoped to postflop only. Its README does not mention preflop ranges. Source: https://github.com/b-inary/postflop-solver
- The desktop app by b-inary accepts PioSOLVER-style text: comma-separated tokens, `22+`, `AQs-ATs`, and `:weight`. Source: https://www.github.com/b-inary/desktop-postflop/discussions/20. Our syntax matches this style.
- TexasSolver reads ranges from a config file as hand lists with optional weights. I only saw this in a fork's sample, so I rate it unverified. Source: https://github.com/Textcat/TexasSolver
- PioSolver: I could not find its manual. One source says Pro has no preflop solver and Edge adds one (low-quality source, unverified). Source: https://www.deucescracked.com/poker/strategy/solver-fundamentals
- MonkerSolver is the preflop tool GTO Wizard names. Sellers of Monker ranges export them to Pio format with weights. Source: https://sites.google.com/view/fedorholzcheap/rangeconverter-solutions-ranges-cheap (a seller page, weak source). A glossary says 6-max results are approximations. Source: https://www.pokerskill.com/poker-glossary/solver/

## Free, citable sources and their terms

- I found no clearly licensed open dataset of solver-output 6-max 100 bb preflop ranges. A search turned up only these: a repo with hand-made ranges and no visible license (https://github.com/data614/Ignition), an MIT range-editor app that holds no solver data (https://github.com/HoldemPokerTools/RangeAssistant), and a solver-answer dataset for LLM training whose license I did not check (https://huggingface.co/datasets/RZ412/PokerBench/blob/main/README.md).
- Free web charts exist, for example https://www.pokerskill.com/charts/gto/. The page names no stack depth and no license. Treat them as "all rights reserved".
- Commercial chart libraries (e.g. https://poker.academy/llms.txt) are paid. Do not copy them.
- License note: b-inary's solver is AGPL-3.0 (https://github.com/b-inary/postflop-solver). TexasSolver is AGPL-3.0, and its author forbids re-uploading binaries and asks for a commercial license to embed the source (https://github.com/bupticybee/TexasSolver). Do not copy code from either into this repo unless you accept AGPL.

## Recommendation: ONE approach

Write your own "house ranges" by hand. Label them as approximations in the file. Store them as plain text presets in this repo under your own license.

Why:
1. No clean, free, solver-grade source exists. Copying a paid chart risks a terms violation.
2. Your project is a solver, not a preflop product. Postflop strategy is valid for any range pair you give it. You can state "solutions are exact for these ranges, not for the real preflop equilibrium".
3. It keeps the pipeline open. Later you can accept Pio-style range text from any user-owned export.
4. Optional upgrade: tune the ranges against numbers you can cite (open sizes and rake from GTO Wizard's pages above).

Caveat: this means your database teaches "solutions for house ranges". Say so on the README and in the trainer.

## Six presets (all are approximations)

I wrote these ranges myself from general knowledge of 6-max strategy. I did not copy any chart. I did not check them against a solver. Real equilibrium ranges differ, especially the 3-bet ranges, which are probably too narrow here. Syntax: I did not run these through the repo parser. Test items like `66-22:0.5` (a range plus a weight) before use.

Assumptions: 100 bb effective, no rake, SB 0.5 bb and BB 1 bb. Pot at flop = all chips in plus any dead blind money. Effective stack = 100 minus the amount each player put in. The SB open size is 3x because GTO Wizard uses 3x for SB (source above). The 3-bet sizes (11 bb and 10 bb) are my assumption. I did not verify them.

Shared strings:
- BTN_OPEN = `22+,A2s+,K4s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,64s+,54s,A2o+,K9o+,Q9o+,J9o+,T9o`
- BTN_CALL_3BET = `QQ:0.5,JJ-77,66-22:0.5,AKs:0.3,AQs,AJs,ATs,A9s-A6s:0.5,KQs,KJs,KTs,K9s:0.5,QJs,QTs,JTs,J9s:0.5,T9s,98s,87s,76s,65s,54s:0.5,AKo:0.3,AQo,AJo:0.5,KQo:0.5`

| # | Spot | OOP range | IP range | Pot (bb) | Eff. stack (bb) | Source |
|---|---|---|---|---|---|---|
| 1 | BTN vs BB SRP (open 2.5, BB calls) | BB call: `22-TT,JJ:0.5,A2s-AQs,K2s+,Q4s+,J6s+,T6s+,96s+,85s+,75s+,64s+,54s,43s,A2o-AJo,K8o+,Q9o+,J9o+,T9o,98o,87o` | BTN_OPEN | 5.5 | 97.5 | House range, hand-built |
| 2 | CO vs BB SRP | BB call: `22-TT,JJ:0.5,A2s-AQs,K4s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,A8o-AJo,KTo+,QTo+,JTo,T9o` | CO open: `22+,A2s+,K6s+,Q8s+,J8s+,T8s+,97s+,87s,76s,65s,54s,A9o+,KTo+,QTo+,JTo` | 5.5 | 97.5 | House range, hand-built |
| 3 | UTG vs BB SRP | BB call: `22-TT,JJ:0.5,A2s-AJs,AQs:0.5,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,54s,ATo-AJo,KQo,KJo,QJo` | UTG open: `22+,A2s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,65s,54s,AJo+,KQo` | 5.5 | 97.5 | House range, hand-built |
| 4 | SB vs BB SRP (SB raises to 3, BB calls; SB is OOP) | SB raise: `22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,86s+,75s+,65s,54s,A2o+,K8o+,Q9o+,J9o+,T9o` | BB call: `22-99,TT:0.5,A2s-AJs,K2s+,Q2s+,J4s+,T6s+,95s+,85s+,74s+,64s+,53s+,43s,A2o-AJo,K5o+,Q8o+,J8o+,T8o+,98o,87o,76o` | 6 | 97 | House range, hand-built |
| 5 | BB 3-bets BTN (BTN opens 2.5, BB 3-bets to 11, BTN calls; BB is OOP) | BB 3-bet: `QQ+,JJ:0.5,AKs,AKo,AQs,AQo:0.5,AJs:0.5,A5s-A2s,KQs:0.5,K9s:0.5,98s:0.5,87s:0.5,76s:0.5,65s:0.5` | BTN_CALL_3BET | 22.5 | 89 | House range, hand-built |
| 6 | SB 3-bets BTN (BTN opens 2.5, SB 3-bets to 10, BTN calls; SB is OOP) | SB 3-bet: `JJ+,TT:0.5,AKs,AKo,AQs,AQo:0.5,AJs:0.5,A5s-A2s,KQs,K9s:0.5,QJs:0.5,T9s:0.5,98s:0.5,87s:0.5,76s:0.5` | BTN_CALL_3BET | 21 | 90 | House range, hand-built |

Pot arithmetic: spot 1-3: 2.5 + 2.5 + 0.5 (SB dead) = 5.5. Spot 4: 3 + 3 = 6. Spot 5: 11 + 11 + 0.5 = 22.5. Spot 6: 10 + 10 + 1 (BB dead) = 21.

Known weak points: the BB defence ranges contain no 3-bets, so spot 1-3 IP/OOP pairs are slightly capped on the BB side. In spots 5 and 6 the 3-bet range is only about 6-7% of hands, narrower than most published charts (I believe, but did not verify).

# Part 2. What could make the project stand out?

## Landscape

- GTO Wizard: study (solution browser), a practice trainer, a hand analyzer, and an AI solver for custom spots. The trainer shows EV change per action (third-party review, not GTO Wizard docs): https://www.vip-grinders.com/poker-tools/gto-wizard/. Heads-up preflop AI solver details: https://blog.gtowizard.com/introducing-gto-wizard-ai-heads-up-preflop-solver/ (the page does not say how the AI works).
- PioSolver, GTO+, Simple Postflop: desktop solvers. Prices vary by source and are unreliable: https://www.deucescracked.com/tools/gto-solvers
- postflop-solver and Desktop Postflop (b-inary, Rust): AGPL, suit isomorphism for turn and river deals, optional 16-bit storage. Development is suspended. Source: https://github.com/b-inary/postflop-solver
- TexasSolver (bupticybee, C++): open source, AGPL. A GPU build, TexasSolverGPU, exists. It is closed source, Windows only, needs an NVIDIA GPU, and gives no speed numbers on its page. Sources: https://github.com/bupticybee/TexasSolver and https://github.com/bupticybee/TexasSolverGPU
- GPU CFR research: a 2026 preprint, GPU-CFR (Li and Huang), claims 29.8-80.4x over the previous fastest GPU CFR on an A100 and includes an HUNL turn subgame (my search snippet; the abstract page does not name poker). Claims are the authors' own. Sources: https://arxiv.org/abs/2609.11923 and code link in the search result at https://github.com/lbn187/GPU-CFR (not opened). Kim and Sandholm (2026) recast CFR as linear algebra on GPU, up to four orders of magnitude faster than OpenSpiel's CPU code. The abstract does not mention poker. Source: https://arxiv.org/abs/2605.14277. An earlier GPU CFR paper: https://arxiv.org/abs/2408.14778.
- LLM and solver work: "Solver-Guided Reasoning for Mixed-Equilibrium Strategies" turns NLHE solver output into readable rules. Source: https://arxiv.org/abs/2608.06741. An open-source terminal trainer, poker-tui, has a solver screen and cached explanations (from a search snippet, not verified): https://pypi.org/project/poker-tui/

## Ranked ideas

Score = impact for recruiters (1-5) x feasibility in a few weeks (1-5).

| Rank | Idea | Impact | Feas. | Score | Prior art and honest note |
|---|---|---|---|---|---|
| 1 | Spot-difficulty and "interestingness" score, computed from the solution: strategy entropy, EV gap between top actions, EV loss of the second-best action, range-wide sensitivity. It drives the database categories and the trainer. | 4 | 5 | 20 | I found no product that publishes such a score. I did not search exhaustively. GTO Wizard shows EV loss per action, so EV-loss ideas are not new. The aggregated per-spot score is the new part. |
| 2 | Public, reproducible benchmark: exploitability vs time and memory, your solver vs postflop-solver and TexasSolver, on 20 fixed spots (your SPEC v4). | 4 | 4 | 16 | b-inary's repo describes its own comparisons (not verified in detail). A neutral table with scripts and spot files is still useful. Mind the AGPL when you run their binaries. |
| 3 | Explainable "why" layer from solver features: equity, equity realization, blocker effects on opponent's continuing range, EV gap. Use deterministic rules, not an LLM. | 4 | 3 | 12 | Partial prior art: the MDT paper (above), poker-tui explanations, and a GTO Wizard feature request I saw (not a shipped feature). Keep it rule-based and validate it against the solver. |
| 4 | Range-sensitivity study: re-solve each stored spot with perturbed ranges and bet sizes, and report which strategies are stable. This answers the "your ranges are house ranges" objection. | 3 | 4 | 12 | I found no prior work, but my search was shallow. Costs compute only. |
| 5 | GPU CFR with published numbers on real flop/turn spots, with exploitability checked by your own best-response code. | 5 | 2 | 10 | Not new: TexasSolverGPU (closed, unquantified) and GPU-CFR (turn subgame). The open, honest, same-code CPU vs GPU chart is the part you can add. Needs NVIDIA hardware (Colab per your SPEC) and a hard memory layout. The risk is high. |
| 6 | Adaptive trainer: track the user's EV loss by spot category, then serve more spots from weak categories. | 4 | 3 | 12 | Exists in spirit: GTO Wizard's trainer and analyzer track EV loss and leaks. Your angle is the difficulty score and the open database. |
| 7 | Solution quality certificates: each stored solution carries exploitability, iterations, tree size and a checksum. Release the database as an open dataset with a license. | 3 | 5 | 15 | PokerBench exists for LLM training. A license-clean strategy dataset with verified exploitability is a small gap. Pure engineering. |

Reordered by score: 1 (20), 2 (16), 7 (15), then 3, 4 and 6 (12), then 5 (10).

My advice: do 1, 2 and 7 first. They fit the pipeline you already planned. Add 3 or 4 as the headline. Treat GPU as a stretch goal and only publish it with a benchmark.

# Gaps in this research

- GTO Wizard's range hand-off, its terms of service, and the PioSolver manual: not verified.
- Price figures: sources conflict. Check vendor sites.
- Preflop ranges above: unvalidated approximations.

# Summary (5 lines)

1. GTO Wizard solves preflop in MonkerSolver and publishes sizes and rake, but I could not verify how ranges reach the flop.
2. No clean, free, solver-grade preflop range set exists, so use hand-built, clearly labeled "house ranges" under your own license.
3. Six presets are in the table (pots 5.5, 5.5, 5.5, 6, 22.5, 21 bb); they are approximations and not run through your parser.
4. Top ideas by impact x feasibility: spot-difficulty score, public benchmark vs reference solvers, license-clean verified dataset.
5. GPU CFR has prior art (GPU-CFR, TexasSolverGPU), so treat it as a stretch goal and publish honest numbers only.
