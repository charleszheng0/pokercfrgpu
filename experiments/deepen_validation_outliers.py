"""Deepen the worst river/cutoff cases from the deterministic 30-position batch."""
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import mean
from time import perf_counter

from poker_cpu.engine import Action
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve

SOURCE = Path(__file__).with_name("postflop_validation_batch.json")
BUDGETS = {
    "river_root": (500, 1000, 2000),
    "river_facing_bet": (500, 1000, 2000),
    "turn_showdown_cutoff": (100, 200, 500),
    "flop_showdown_cutoff": (100, 200, 500),
}


def make_request(row, iterations):
    p = row["request"]
    ranges = tuple(WeightedRange(tuple(tuple(hand) for hand in seat["hands"]),
                                 tuple(seat["weights"])) for seat in p["ranges"])
    abstraction = p["abstraction"]
    history = tuple(Action(kind, amount) for kind, amount in p["history"])
    return SolveRequest(
        board=tuple(p["board"]), street=p["street"], pot_before_street=p["pot"],
        stacks_at_street_start=tuple(p["stacks"]),
        betting=BettingContext(button=1, first_actor=p["first_actor"],
                               current_player=p["current_player"], street_actions=history),
        ranges=ranges,
        abstraction=ActionAbstraction(
            bet_fractions=tuple(abstraction["bet_fractions"]),
            raise_fractions=(tuple(abstraction["raise_fractions"])
                             if abstraction["raise_fractions"] is not None else None),
            max_raises=abstraction["max_raises"],
            include_min_size=abstraction["include_min_size"],
        ),
        config=SolverConfig(iterations=iterations, max_joint_deals=4,
                            continuation_mode=p["continuation_mode"]),
        acting_hand=tuple(p["ranges"][p["current_player"]]["hands"][0]),
        range_source="seeded_batch_assumption",
    )


def main():
    batch = json.loads(SOURCE.read_text())
    outliers = {}
    for profile, budgets in BUDGETS.items():
        candidates = [row for row in batch["scenarios"] if row["profile"] == profile]
        outliers[profile] = max(candidates, key=lambda row: row["nash_conv_chips"])
    results = []
    for profile, row in outliers.items():
        runs = []
        for budget in BUDGETS[profile]:
            request = make_request(row, budget)
            started = perf_counter()
            result = solve(request)
            wall_ms = (perf_counter() - started) * 1000
            actions = result.selected_hand.actions if result.selected_hand else ()
            runs.append({
                "iterations": result.iterations,
                "status": result.status,
                "nash_conv_chips": result.nash_conv_chips,
                "latency_ms": result.latency_ms,
                "wall_latency_ms": wall_ms,
                "actions": [
                    {"kind": item.action.kind, "amount": item.action.amount,
                     "frequency": item.frequency, "ev_chips": item.ev_chips}
                    for item in actions
                ],
            })
            print(profile, budget, result.status,
                  "NashConv", round(result.nash_conv_chips, 6),
                  "ms", round(wall_ms, 1), flush=True)
        ref = { (a["kind"], a["amount"]): a for a in runs[-1]["actions"] }
        comparisons = []
        for run in runs[:-1]:
            current = {(a["kind"], a["amount"]): a for a in run["actions"]}
            keys = list(ref)
            best = max(ref[k]["ev_chips"] for k in keys)
            policy_value = sum(current[k]["frequency"] * ref[k]["ev_chips"] for k in keys)
            comparisons.append({
                "iterations": run["iterations"],
                "action_ev_mae_chips": mean(abs(current[k]["ev_chips"] - ref[k]["ev_chips"])
                                             for k in keys),
                "strategy_l1": sum(abs(current[k]["frequency"] - ref[k]["frequency"])
                                    for k in keys),
                "root_policy_regret_chips": best - policy_value,
            })
        results.append({
            "scenario_id": row["scenario_id"], "profile": profile,
            "original_request": row["request"], "runs": runs,
            "comparisons_vs_deepest_finite_reference": comparisons,
        })
    record = {
        "experiment": "deepened_30_batch_outliers",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": "For each profile with any >0.1-chip residual, deepen its worst exact same-game scenario. Compare root EV/frequency with the deepest finite iteration reference; it is not ground truth.",
        "cases": results,
        "limitations": [
            "Only the worst one of six scenarios per selected profile is deepened.",
            "The deepest solve is a finite-iteration reference, not equilibrium ground truth.",
            "No full-turn case is deepened because the API currently caps exact full turn at 100 iterations.",
            "All ranges contain at most two combos per seat; assumptions are synthetic.",
        ],
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
