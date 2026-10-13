# Small-game research code

This package keeps the project's original Kuhn CFR and Leduc/NFSP experiments separate from the new Hold'em reference engine. The files were moved and formatted; the learning and solving algorithms have not been redesigned.

| File | Role |
|---|---|
| `kuhn.py` | Original tabular Kuhn CFR experiment; now runs only when invoked as a module. |
| `leduc.py` | Original two-round Leduc environment. |
| `encoder.py`, `dqn.py` | Leduc observation encoder and Q network. |
| `train_nfsp.py` | Original NFSP self-play trainer with Q, target, and average-policy networks. New runs write to `artifacts/nfsp/`. |
| `compare_cfr_nfsp.py` | Independent Leduc rules/CFR comparison and exact best-response computation. Defaults to the preserved checkpoints in `artifacts/legacy/`. |
| `replay_buffer.py` | Original standalone replay-buffer example; the trainer also has its own buffer classes. |

Run from the repository root:

```sh
python3 -m unittest discover -s tests -v
python3 -m small_games.kuhn
python3 -m small_games.leduc
python3 -m small_games.compare_cfr_nfsp --iterations 1500
python3 -m small_games.train_nfsp --episodes 100000 --seed 42
```

The trainer's default is 100,000 Leduc episodes and requires PyTorch. It accepts `--episodes`, `--batch-size`, `--seed`, and `--output-dir`; new weights go to `artifacts/nfsp/` unless another directory is specified. The old checkpoint files have unknown training provenance; keeping them loadable does not validate NFSP convergence. `compare_cfr_nfsp` uses those old files by default; pass `--model` and `--model1` to compare newly trained weights.
