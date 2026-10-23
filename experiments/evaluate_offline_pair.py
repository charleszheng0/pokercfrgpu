"""Paired fixed-seed NFSP candidate/incumbent evaluation without training.

Run: python3 -m experiments.evaluate_offline_pair --hands 1000 --seed 900000
The earlier user stop request is respected: this script only loads saved models.
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

from experiments.holdem_nfsp import evaluate
from poker_cpu.nfsp import HoldemNFSPAgent
from trainer.offline_training import ROOT, STORE, paired_policy_difference, read_json


def _agent(path, hidden):
    model = HoldemNFSPAgent(hidden=hidden)
    model.policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.mode = "average"
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    if args.hands < 2:
        parser.error("--hands must be at least two")
    history = read_json(STORE / "history.json", [])
    if not history:
        parser.error("No saved offline NFSP candidate exists")
    candidate = history[-1]
    active = read_json(STORE / "active.json", None)
    candidate_path = Path(candidate["model_dir"]) / "policy_player_1.pt"
    incumbent_path = (Path(active["model_dir"]) / "policy_player_1.pt" if active
                      else ROOT / "artifacts/holdem_nfsp/seed11/policy_player_1.pt")
    if not candidate_path.is_file() or not incumbent_path.is_file():
        parser.error("A candidate or incumbent policy file is missing")
    torch.set_num_threads(1)
    started = perf_counter()
    new = evaluate(_agent(candidate_path, candidate["settings"]["hidden_layers"]),
                   args.opponent, args.hands, args.seed, return_gains=True,
                   agent_seat=1)
    old = evaluate(_agent(incumbent_path, active["hidden_layers"] if active
                          else (128, 128)), args.opponent, args.hands, args.seed,
                   return_gains=True, agent_seat=1)
    pair = paired_policy_difference(new.pop("per_hand_gains"),
                                    old.pop("per_hand_gains"))
    result = {
        "experiment": f"saved_nfsp_candidate_vs_incumbent_paired_{args.opponent}",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(), "torch": torch.__version__,
        "machine": platform.machine(),
        "rules": ("20-chip heads-up Hold'em, 1/2 blinds, preset wager actions; "
                  "seat-1 policy stays in seat 1, button alternates; " + ("32-sample heuristic" if args.opponent == "heuristic"
                                                 else "random legal-action") + " opponent"),
        "method": "Candidate and incumbent seat-1 policies separately play seat 1 on the same seeded deals and alternating buttons. Opponent RNG seeds match. Per-hand candidate-minus-incumbent chip returns form a paired normal-approximation 95% interval. No training or promotion occurs.",
        "hands": args.hands, "evaluation_seed": args.seed,
        "candidate_run_id": candidate["run_id"],
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "incumbent_sha256": hashlib.sha256(incumbent_path.read_bytes()).hexdigest(),
        "candidate": new, "incumbent": old, "paired_difference": pair,
        "elapsed_seconds": perf_counter() - started,
        "limitations": ["One heuristic opponent is not a full strength suite.",
                        "A positive interval would not establish Nash proximity or strength against other agents.",
                        "Shared seeds reduce deal variance but policies induce different later histories."],
    }
    output = args.output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": new, "incumbent": old,
                      "paired_difference": pair,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
