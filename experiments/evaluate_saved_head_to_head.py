"""Direct saved NFSP candidate/incumbent match; no training or promotion.

Run: python3 -m experiments.evaluate_saved_head_to_head --hands 1000
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import date
from pathlib import Path
from time import perf_counter

import torch

from experiments.evaluate_offline_pair import _agent
from experiments.holdem_nfsp import evaluate_head_to_head
from trainer.offline_training import ROOT, STORE, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    if args.hands < 2:
        parser.error("--hands must be at least two")
    history = read_json(STORE / "history.json", [])
    if not history:
        parser.error("No saved candidate")
    candidate = history[-1]
    active = read_json(STORE / "active.json", None)
    candidate_paths = [Path(candidate["model_dir"]) / f"policy_player_{seat}.pt"
                       for seat in (0, 1)]
    incumbent_dir = (Path(active["model_dir"]) if active
                     else ROOT / "artifacts/holdem_nfsp/seed11")
    incumbent_paths = [incumbent_dir / f"policy_player_{seat}.pt"
                       for seat in (0, 1)]
    if not all(path.is_file() for path in candidate_paths + incumbent_paths):
        parser.error("A policy file is missing")
    torch.set_num_threads(1)
    started = perf_counter()
    report = evaluate_head_to_head(
        tuple(_agent(path, candidate["settings"]["hidden_layers"])
              for path in candidate_paths),
        tuple(_agent(path, active["hidden_layers"] if active else (128, 128))
              for path in incumbent_paths), args.hands, args.seed)
    record = {
        "experiment": "saved_nfsp_direct_head_to_head",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()),
        "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "evaluation_seed": args.seed,
        "rules": "20-chip heads-up Hold'em, 1/2 blinds, preset wager actions; both checkpoint teams use their seat-specific policies, alternating candidate team seat and button",
        "candidate_run_id": candidate["run_id"],
        "candidate_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in candidate_paths},
        "incumbent_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in incumbent_paths},
        "result": report, "elapsed_seconds": perf_counter() - started,
        "limitations": ["One direct match against one incumbent does not measure equilibrium proximity.",
                        "Normal interval is a screening approximation; replicate with further seeds before broad strength claims."],
    }
    output = args.output
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": report, "elapsed_seconds": record["elapsed_seconds"]}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
