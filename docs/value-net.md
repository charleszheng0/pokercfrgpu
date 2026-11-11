# Depth-limited flop solving with a value network

This document tells how `pokervnet` solves a flop without the turn and river tree. The method
is the one of DeepStack (Moravčík et al. 2017) and ReBeL (Brown et al. 2020): solve only the
current street with CFR, and use a learned model for the value of the rest of the hand.

## 1. The idea

A full flop tree has the flop, all turn cards, and all river cards. It uses 0.5–3.9 GB and
takes minutes to hours to solve. A depth-limited tree stops at the end of the flop. Each line
that ends the flop (check-check, bet-call, bet-raise-call, ...) becomes a **leaf**. The tree is
then very small, and DCFR on it takes seconds.

The solver must still know what each hand wins from a leaf to the end of the hand. A **leaf
evaluator** gives that value. It gets the reach of every hand of both players at the leaf (the
two ranges) and returns the value of every hand.

In each DCFR iteration (the CFR-D / ReBeL scheme):
1. Walk the flop tree with the current strategy and find the two ranges at each leaf.
2. Ask the evaluator for the leaf values of these ranges.
3. Do the normal DCFR update on the flop tree with these values.

## 2. The value model

Units: `y(h) = v(h) / (contrib x M(h))`. `v(h)` is the value of hand h in solver units,
`contrib` is the chips each player has in the pot at the leaf, and `M(h)` is the opponent
reach that does not share a card with h. So y is "pot share won per unit of opponent range".

Base value: `d(h)` is the **check-down value** of h: E[win − lose] against the opponent range
over all turn and river cards. If nobody bets again, `y = d` exactly. The solver gets `d` from
one matrix product with a precomputed hand-vs-hand equity matrix.

The network predicts the **correction** `y(h) − d(h)`: how much betting on the turn and river
changes the value of the hand. Hands with draws, nut hands and bluff catchers get different
corrections.

Inputs (25 per hand), all independent of suits and of the exact board:

| Feature | Meaning |
|---|---|
| d | check-down value |
| E[d_runout²], E[d²] − d² | spread of the value over the runouts (draws, polarization) |
| sd over turn cards | how much one turn card changes the value |
| percentile in own range | where the hand is in its own range |
| own range mean d | range advantage |
| IP flag | position |
| log(1 + SPR), 1 / (1 + SPR) | stack-to-pot ratio at the leaf |
| 8-bin histograms of d, own and opponent range | shape of both ranges |

Network: 25 → 64 → 64 → 1, ReLU, Adam, mean squared error. It is written in C++ (no
external library). After the network, a zero-sum correction (as in DeepStack) makes the two
players' values cancel.

The draw features need a pass over all 1,176 runouts. They change slowly, so the evaluator
recomputes them on the first 4 calls of a leaf and then on every 16th call.

## 3. Training data

`pokervnet gen` solves random flops with the full tree (DCFR, 16-bit storage) and stores, at
every line that ends the flop, the average-strategy reach of each hand and the value of each
hand. These values come from the real turn and river play of the solve. The flops of
`data/flop_list.txt` are never used for training, so they can be test flops.

## 4. How the quality is measured

The depth-limited solve gives a flop strategy only. To score it on the real game:
1. Lock this flop strategy into the full tree (node locking).
2. Solve the turn and river with DCFR.
3. Measure the exploitability of the whole strategy (% of the pot).

The lowest value seen is an upper bound for the flop strategy. The same check is done for the
flop strategy of full DCFR at the same solve time, and for check-down leaves (no network).

## 5. Tests

`./build/tests` test [8]: with no betting after the flop, the check-down value is the exact
leaf value. The depth-limited tree must then give the same EV and exploitability as the full
tree (measured: identical to 4 decimals), the stored samples must match `d` (error < 1e-6),
and a locked flop strategy must give the same EV.

## 6. Results (measured 2026-11-10 to 11)

Training loop (ReBeL style): each round runs the depth-limited solve with the current network,
records the ranges that reach the flop-end leaves, labels each range with an exact turn+river
DCFR solve, and retrains. 8 rounds: 0-4 on the M2 laptop, 5-7 on a Colab T4 GPU with the CUDA
solver ([gpu.md](gpu.md)). Validation boards are not in training.

| Label source | Time per exact label | Labels per hour |
|---|---|---|
| M2 CPU, 8 threads (540 labels) | 11.88 s | about 300 |
| T4 GPU, CUDA solver (1,200 labels) | 3.38-3.56 s | about 860 |

CUDA labels match CPU labels to a relative L2 error of 4.5e-5.

Leaf value error (mean absolute error, % of the leaf pot, unseen boards): check-down heuristic
about 15%; round 3 network 4.42-4.52%; round 7 network 4.16%.

Held-out 3-bet flop Ac9d8c, flop-locked exploitability (% of pot) by round:

| Round | 0 | 1 | 2 | 3 | 4 | 5 (GPU) | 6 (GPU) | 7 (GPU) |
|---|---|---|---|---|---|---|---|---|
| Exploitability | 8.8 | 6.7 | 4.7 | 4.25 | 4.66 | 4.41 | 4.07 | 4.28 |

Three held-out 3-bet flops, round 7 network (depth-limited solve on the Colab CPU) against full
DCFR on the T4:

| Flop | Check-down leaves | Network | Full DCFR (T4) |
|---|---|---|---|
| Ac9d8c | 23.7% | 4.28% in 6.0 s | 0.26% in 5.1 s |
| QcJd5d | 25.1% | 4.25% in 6.5 s | 0.66% in 8.0 s |
| Tc9d4h | 24.4% | 3.90% in 7.7 s | 0.36% in 19.6 s |

On the M2 laptop the depth-limited network solve takes 1.2-1.7 s. A full DCFR solve of Ac9d8c
takes 39.6 s, and 20 DCFR iterations (the same 4% quality) take 6.8-18.3 s.

Conclusions:
- The network cuts the error of a depth-limited solve about 6x (24% to 4%).
- It does not reach the 0.5% quality gate, so stored answers still come from full DCFR.
- With a GPU, full DCFR is about as fast and about 10x more accurate. The network helps only
  where there is no GPU.
- Only 3-bet flops were tested. The round 5-7 networks were lost when the Colab runtime reset;
  `models/vnet_r3.net` is the best saved network.
