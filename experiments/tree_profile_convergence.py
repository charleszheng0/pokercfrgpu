"""Exact restricted NashConv versus iteration budgets on declared tiny profiles."""

import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter

from poker_cpu.solve import solve
from poker_cpu.tree_profiles import supported_tree_profiles


def main():
    profiles = {profile.profile_id: profile for profile in supported_tree_profiles()}
    budgets = {
        "river_root": (1, 5, 20, 100),
        "river_facing_bet": (1, 5, 20, 100),
        "turn_full_tree": (1, 2, 5, 10, 30, 100),
        "turn_showdown_cutoff": (1, 10, 30, 100),
        "flop_showdown_cutoff": (1, 10, 30, 100),
        "flop_full_tree_experimental": (1, 2, 3),
        "flop_full_tree_sampled": (100, 500, 2000),
    }
    rows = []
    for profile_id, iterations_list in budgets.items():
        profile = profiles[profile_id]
        for iterations in iterations_list:
            request = replace(profile.request,
                              config=replace(profile.request.config, iterations=iterations))
            started = perf_counter()
            result = solve(request)
            rows.append({
                "profile_id": profile_id,
                "iterations_requested": iterations,
                "iterations_completed": result.iterations,
                "status": result.status,
                "nash_conv_chips": result.nash_conv_chips,
                "latency_ms": result.latency_ms,
                "wall_latency_ms": (perf_counter() - started) * 1000,
                "root_actions": [
                    {"kind": action.kind, "amount": action.amount}
                    for action in result.actions
                ],
                "selected_hand_actions": [
                    {"kind": item.action.kind, "amount": item.action.amount,
                     "frequency": item.frequency, "ev_chips": item.ev_chips}
                    for item in result.selected_hand.actions
                ] if result.selected_hand else [],
            })
            if result.status != "supported_restricted_game":
                raise RuntimeError(f"{profile_id} unsupported at {iterations}: {result.reason}")
    by_profile = {}
    for profile_id in budgets:
        group = [row for row in rows if row["profile_id"] == profile_id]
        reference = group[-1]
        reference_evs = {(item["kind"], item["amount"]): item["ev_chips"]
                         for item in reference["selected_hand_actions"]}
        reference_freqs = {(item["kind"], item["amount"]): item["frequency"]
                           for item in reference["selected_hand_actions"]}
        checks = []
        for row in group[:-1]:
            action_rows = row["selected_hand_actions"]
            current_evs = {(item["kind"], item["amount"]): item["ev_chips"]
                           for item in action_rows}
            current_freqs = {(item["kind"], item["amount"]): item["frequency"]
                             for item in action_rows}
            action_keys = sorted(set(reference_evs) & set(current_evs))
            if not action_keys:
                continue
            best_reference_ev = max(reference_evs[key] for key in action_keys)
            current_policy_reference_ev = sum(
                current_freqs[key] * reference_evs[key] for key in action_keys)
            checks.append({
                "iterations": row["iterations_completed"],
                "action_ev_mae_chips_vs_deepest_budget": sum(
                    abs(current_evs[key] - reference_evs[key]) for key in action_keys
                ) / len(action_keys),
                "strategy_l1_vs_deepest_budget": sum(
                    abs(current_freqs[key] - reference_freqs[key]) for key in action_keys),
                "root_policy_regret_chips_vs_deepest_action_values": (
                    best_reference_ev - current_policy_reference_ev),
            })
        by_profile[profile_id] = {
            "reference_iterations": reference["iterations_completed"],
            "reference_nash_conv_chips": reference["nash_conv_chips"],
            "comparisons": checks,
            "reference_scope": "same restricted profile and same private hand; deepest budget is not an exact equilibrium oracle",
        }
    sampled_profile = profiles["flop_full_tree_sampled"]
    seed_sensitivity = []
    for seed in (7, 19, 20261020):
        request = replace(
            sampled_profile.request,
            config=replace(sampled_profile.request.config, sampling_seed=seed),
        )
        result = solve(request)
        seed_sensitivity.append({
            "seed": seed, "iterations": result.iterations,
            "nash_conv_chips": result.nash_conv_chips,
            "latency_ms": result.latency_ms,
        })
    record = {
        "experiment": "tiny_restricted_tree_convergence_sweep",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "methodology": (
            "Exact restricted-game NashConv measured at each requested iteration budget; "
            "each profile fixes board, pot, stacks, range, action menu, and root history. "
            "No values are compared across profile IDs because their games differ."
        ),
        "profiles": rows,
        "same_game_decision_quality_vs_deepest_budget": by_profile,
        "sampled_flop_seed_sensitivity_at_2000": seed_sensitivity,
        "limitations": [
            "One private combo per seat and a tiny sizing abstraction; results do not generalize.",
            "The full flop tree is measured at one iteration only due its high computation cost.",
            "Cutoff profiles omit later-street betting and use a different game than full-tree solves.",
            "No independent external solver reference is used.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["profile_id"], row["iterations_completed"],
              "NashConv", row["nash_conv_chips"],
              "ms", round(row["wall_latency_ms"], 2))
    for profile_id, values in by_profile.items():
        print(profile_id, "decision comparisons", values["comparisons"])
    print(output)


if __name__ == "__main__":
    main()
