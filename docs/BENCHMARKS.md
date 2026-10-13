# Measured benchmarks

Hardware: Apple M2 (`arm64`, macOS Darwin 23.5.0), Python 3.10.6. PyTorch 2.10.0 is installed, `torch.cuda.is_available()` returned `False`, and `nvcc` was not found. Git HEAD at measurement: `f0f9fc7feeb704d51bb77bf8763e00ea30d5c0fe`; working tree was dirty. Results are from the actual commands below and do not represent a clean released revision.

| Measurement | Command / conditions | Result |
|---|---|
| CPU engine suite | `python3 -m unittest discover -s tests -v` | 7/7 passed; 150 seeded random hands within the suite; final run 0.086 s |
| Heuristic vs random | `python3 -m experiments.baseline --hands 100 --seed 7 --samples 32` | −2.68 ± 5.91 standard-error chips/hand; 27/100 wins; 4.75 s total |
| Independent evaluator check | `PYTHONPATH=/tmp/pokercfrgpu-oracle:. python3 -m experiments.crosscheck_evaluator --trials 5000 --seed 19` | 0 rank disagreements against Treys 0.1.8 |

The benchmark script alternates the heuristic seat and saves exact seed, sample count, elapsed time, and revision in [raw JSON](../experiments/baseline_seed7.json). The [evaluator result](../experiments/evaluator_crosscheck_seed19.json) uses an optional Treys install in `/tmp`; Treys is not a runtime dependency. The simple standard error does not account for all poker variance or adaptive policy uncertainty. One 100-hand run is not sufficient to rank strategic strength.

CFR exploitability, NFSP multi-seed learning, value-network held-out error, search latency, and CPU-versus-GPU speedup have **not been measured** in this milestone. The existing `.cu` snippets are not benchmark evidence.
