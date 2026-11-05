# Heads-up Hold'em solver & trainer (CPU research prototype)

Heads-up no-limit Hold'em engine, CFR+ river/turn solver, NFSP agent, and a local browser trainer.

**Status:** the river solver works for small supplied ranges only. Flop is experimental, preflop is unsupported, the NFSP agent loses to the heuristic baseline, and CUDA has not been started. Solver NashConv measures exploitability inside the restricted game you supply, not full-game poker strength. Ranges in the UI are assumptions, not inferred from play.

## Run

```sh
python3 -m pip install -r requirements-research.txt   # PyTorch, for NFSP + full tests
python3 -m unittest discover -s tests -v              # Python tests
npm ci && npm run test:browser                        # Playwright browser test
python3 -m trainer.server --port 8765                 # open http://127.0.0.1:8765
python3 -m poker_cpu.play --opponent heuristic        # terminal play
```

## Layout

- `poker_cpu/` engine, evaluator, CFR solvers, NFSP, range tools
- `trainer/` local web server and UI (`static/`)
- `small_games/` Kuhn/Leduc CFR and NFSP references
- `experiments/` scripts and saved measurements
- `tests/` unit and browser tests
- `cuda_experiments/` archived, unused
