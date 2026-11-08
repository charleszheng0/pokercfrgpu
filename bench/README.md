# Benchmarks

## flop_memory.sh

Memory of a full flop tree for three board textures, with and without suit isomorphism and
16-bit storage. Builds the tree only (no solve).

## threads.sh

Thread load balance: wall time and CPU use ((user+sys)/real) for a fixed iteration count on

- turn: `Ks 9d 5c 2h`, bets 33,75, raise 60, 60 iterations
- small flop: `Ks 9s 5s`, bet 50, no raises, 10 iterations

```
bench/threads.sh [NEW_BIN] [OLD_BIN] [RUNS] [THREADS...]
bench/threads.sh build/pokersolve /path/to/old/pokersolve 5 8 4
```

With OLD_BIN, old and new runs alternate, because the machine is noisy. The summary shows
the median wall time per case, binary and thread count; the cpu and exploitability columns
come from the median run. The exploitability of old and new must be the same (up to float
summation order).

### Thread pool: nested parallel_for

`src/thread_pool.cpp` lets idle threads take items of nested parallel_for groups (a river
chance node inside a turn-card item). A nested group is published only when a thread is
idle; otherwise it runs serially as before. Exploitability is identical to the old pool
(turn 4.087%, flop 21.385%).

Apple M2 (4P + 4E), 5 alternating runs each, median wall time. The machine was shared with
other solver jobs (load average 25-140, so the solver got about 1-3 cores): the numbers
show no regression, but they are too noisy to show the expected gain from less idle time.

| case | threads | old s | new s | old cpu | new cpu |
|------|---------|-------|-------|---------|---------|
| turn | 8       | 14.67 | 14.26 | 0.91    | 0.94    |
| turn | 4       | 22.54 | 19.87 | 0.60    | 0.67    |
| flop | 8       |  3.30 |  3.22 | 2.72    | 2.80    |
| flop | 4       |  5.79 |  5.64 | 1.60    | 1.65    |

The turn tree has only one chance level below the root (river cards), so it has no nested
groups; the flop case is the one that can gain. Repeat on a quiet machine.

An idle thread blocks at once (`POOL_SPIN_US`, default 0). With a 50 us yield spin, the
turn case was slower on the oversubscribed machine (19.4 s vs 14.6 s, 3 runs each).
