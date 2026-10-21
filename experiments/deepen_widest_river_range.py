"""Check action stability as the widest validated river fixture is deepened."""

import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.engine import Action
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import (ActionAbstraction, BettingContext, SolveRequest,
                             SolverConfig, solve)


def main():
    source = json.loads(Path("experiments/river_range_width_validation.json").read_text())
    candidate = max(
        (row for row in source["scenarios"] if row["range_width_per_seat"] == 8),
        key=lambda row: row["wall_latency_ms"],
    )
    raw = candidate["request"]
    ranges = tuple(WeightedRange(
        tuple(tuple(hand) for hand in row["hands"]), tuple(row["weights"])
    ) for row in raw["ranges"])
    base = SolveRequest(
        board=tuple(raw["board"]), street="river",
        pot_before_street=raw["pot"],
        stacks_at_street_start=tuple(raw["stacks"]),
        betting=BettingContext(
            button=1, first_actor=0, current_player=raw["current_player"],
            street_actions=tuple(Action(kind, amount)
                                 for kind, amount in raw["street_actions"]),
        ),
        ranges=ranges,
        abstraction=ActionAbstraction.shark_style_postflop(),
        config=SolverConfig(iterations=500, max_joint_deals=64),
        acting_hand=ranges[0].hands[0], range_source="seeded_river_range_width_validation",
    )

    rows = []
    for iterations in (500, 1000, 2000):
        request = replace(base, config=replace(base.config, iterations=iterations))
        started = perf_counter()
        result = solve(request)
        rows.append({
            "iteration_budget": iterations,
            "iterations": result.iterations,
            "nash_conv_chips": result.nash_conv_chips,
            "latency_ms": result.latency_ms,
            "wall_latency_ms": (perf_counter() - started) * 1000,
            "hero_hand": list(result.selected_hand.hand),
            "actions": [{
                "kind": row.action.kind, "amount": row.action.amount,
                "frequency": row.frequency, "ev_chips": row.ev_chips,
            } for row in result.selected_hand.actions],
        })
        print(iterations, "NashConv", round(result.nash_conv_chips, 6),
              "latency_ms", round(result.latency_ms, 1), flush=True)

    reference = rows[-1]
    reference_actions = {(row["kind"], row["amount"]): row
                         for row in reference["actions"]}
    comparisons = []
    for row in rows[:-1]:
        actions = {(item["kind"], item["amount"]): item for item in row["actions"]}
        keys = tuple(reference_actions)
        best_reference_ev = max(reference_actions[key]["ev_chips"] for key in keys)
        policy_value_on_reference = sum(
            actions[key]["frequency"] * reference_actions[key]["ev_chips"]
            for key in keys
        )
        comparisons.append({
            "iterations": row["iterations"],
            "hero_action_ev_mae_chips_vs_2000": sum(
                abs(actions[key]["ev_chips"] - reference_actions[key]["ev_chips"])
                for key in keys
            ) / len(keys),
            "hero_strategy_l1_vs_2000": sum(
                abs(actions[key]["frequency"] - reference_actions[key]["frequency"])
                for key in keys
            ),
            "reference_policy_regret_chips": best_reference_ev - policy_value_on_reference,
        })

    record = {
        "experiment": "deepen_widest_shark_river_range_fixture",
        "date": date.today().isoformat(),
        "seed": source["seed"],
        "source_scenario_id": candidate["scenario_id"],
        "source_revision": source["revision"],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version,
        "position": {key: raw[key] for key in (
            "board", "pot", "stacks", "current_player", "street_actions", "ranges",
            "compatible_joint_deals", "action_tree",
        )},
        "runs": rows,
        "comparisons_vs_2000_iteration_finite_reference": comparisons,
        "limitations": [
            "One seeded synthetic river fixture; no statistical population claim.",
            "2,000 iterations is a finite-run comparison reference, not ground truth.",
            "Action EV/frequency comparison is for the selected hero combo only.",
            "Exact NashConv applies only to this supplied range and restricted action tree.",
        ],
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("comparisons", json.dumps(comparisons, indent=2))
    print(path)


if __name__ == "__main__":
    main()
