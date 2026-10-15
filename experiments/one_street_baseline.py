"""Benchmark exact-runout one-street flop/turn CFR+ as a research baseline."""

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from poker_cpu.one_street_solver import OneStreetGame
from poker_cpu.river_solver import RiverCFR


RANGES = (
    (("Jc", "Tc"), ("8h", "8s")),
    (("9h", "9s"), ("Ac", "Ad")),
)


def main():
    runs = []
    for board in (("As", "Kh", "Qd", "7c"), ("As", "Kh", "Qd")):
        started = time.perf_counter()
        game = OneStreetGame(board, RANGES, pot=8, stack=10)
        equity_ms = (time.perf_counter() - started) * 1000
        solver = RiverCFR(game, variant="cfr+")
        started = time.perf_counter()
        solver.train(200)
        search_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        report = solver.analyze_root(player=0, private_hand=RANGES[0][0])
        analysis_ms = (time.perf_counter() - started) * 1000
        runs.append({
            "street": game.street,
            "board": board,
            "joint_deals": len(game.deals),
            "future_runouts_per_deal": 44 if len(board) == 4 else 990,
            "iterations": 200,
            "equity_precomputation_ms": equity_ms,
            "search_ms": search_ms,
            "analysis_ms": analysis_ms,
            "nash_conv": report["nash_conv"],
            "hero_action_frequencies": [
                {"kind": action.kind, "amount": action.amount, "frequency": value}
                for action, value in report["per_hand"][0]["frequency"].items()
            ],
            "hero_conditional_action_evs": [
                {"kind": action.kind, "amount": action.amount, "ev": value}
                for action, value in report["per_hand"][0]["action_evs"].items()
            ],
        })
    record = {
        "experiment": "exact_runout_one_street_cfr_reference",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty; one-street reference not committed",
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "ranges": RANGES,
        "pot": 8, "effective_stack": 10, "big_blind": 2,
        "continuation_assumption": "No betting after the current street; all remaining public cards are enumerated exactly at showdown. This is not a full multi-street strategy.",
        "methodology": "Two hands per seat, four compatible joint deals, 200 synchronous CFR+ iterations. Equity precomputation, CFR search, and exact root analysis timed separately. One CPU run per street; no random sampling.",
        "runs": runs,
    }
    path = Path(__file__).with_name("one_street_baseline.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)
    for run in runs:
        print(run["street"], "NashConv", round(run["nash_conv"], 6),
              "equity_ms", round(run["equity_precomputation_ms"], 2),
              "search_ms", round(run["search_ms"], 2),
              "analysis_ms", round(run["analysis_ms"], 2))


if __name__ == "__main__":
    main()
