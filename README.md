# pokercfrgpu

A heads-up No-Limit Hold'em postflop solver in C++, with an ASCII web UI.
It uses vector-form Discounted CFR: one tree pass handles every hand of both players.

![spec](docs/spec.png)

## Run

```bash
cmake -S . -B build && cmake --build build -j8
./build/tests            # acceptance tests (about 10 s)
./build/pokerlab 8080    # then open http://127.0.0.1:8080 (a guided tour starts on the first visit; press TOUR to replay)
```

Command line:

```bash
./build/pokersolve --board "Ks 9d 5c 2h 2d" --pot 10 --stack 95 \
  --oop "TT-22,AJs-A2s,KTs+,QTs+,AJo-A8o" --ip "AA-22,A2s+,K7s+,A8o+,KTo+" \
  --bets 33,75 --raises 60 --target 0.3
```

Size keys: `bets`, `raises`, and overrides such as `ip_bets`, `river_bets`,
`oop_turn_raises`. The value `none` removes all sizes.

## How it works

1. **Tree builder** (`src/game.cpp`): betting nodes, fold/showdown leaves, and chance
   nodes for each turn and river card. All nodes sit in one flat array.
2. **DCFR** (`Game::walk`): the traverser gets a value vector per hand; the opponent
   passes a reach vector per hand. Regrets update with discounts alpha = 1.5, beta = 0,
   gamma = 2.
3. **Showdown in O(n)**: hands are sorted by strength once per river board. One sweep up
   and one sweep down give each hand its win and loss totals. Per-card sums remove
   hands that share a card (blockers).
4. **Exploitability**: an exact best response for each player. A solve stops when
   exploitability is below the target (default 0.5% of the pot).

## Measured (Apple M2, 8 threads)

| Spot | Combos (OOP / IP) | Result |
|---|---|---|
| River, 2 bet sizes | 1,081 / 1,081 | 0.38% of pot in 0.28 s |
| Turn, 2 bet sizes | 458 / 458 | 0.37% of pot in 17 s |
| Flop, 1 bet size, 1 raise | 270 / 314 | 0.51% of pot in 7 min, 2.4 GB |
| Flop, 2 bet sizes | 270 / 314 | refused: needs 12.4 GB |

## Tests (`tests/tests.cpp`)

- Evaluator: all 2,598,960 five-card hands give the known category counts.
- Ranges: combo counts for `AA`, `AKs`, `22+`, `A5s-A2s`, weights, errors.
- Terminal math: the fast sweep equals a slow pairwise loop (error about 1e-7).
- Toy river game: call frequency 0.500 and bluff share 0.333, as theory predicts.
- Speed: river under 2 s, turn under 60 s, both under 0.5% of pot.

## Status and next steps

See [SPEC.md](SPEC.md). Next: suit isomorphism and compression for full flop solves,
a solution store and spot queue, play mode, then CUDA kernels benchmarked on Colab.
The old Python prototype is kept at tag `python-prototype`.
