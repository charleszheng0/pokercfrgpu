"""Trace one seeded Play & Learn hand through the existing shared solve API."""

import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from poker_cpu.engine import Action, HoldemGame
from poker_cpu.solve import SolverConfig, request_from_game, solve
from trainer.server import TableSession, experimental_review_ranges


class PassiveAgent:
    def act(self, observation):
        call = next((action for action in observation["legal_actions"]
                     if action.kind == "call"), None)
        return call or next(action for action in observation["legal_actions"]
                            if action.kind == "check")


def solve_trace(request):
    started = perf_counter()
    report = solve(request)
    hand = report.selected_hand
    return {
        "status": report.status, "reason": report.reason,
        "validation_status": report.validation_status,
        "wall_ms": (perf_counter() - started) * 1000,
        "reported_latency_ms": report.latency_ms,
        "iterations": report.iterations,
        "nash_conv_chips": report.nash_conv_chips,
        "joint_deals": report.joint_deals,
        "actions": [] if hand is None else [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev_chips": row.ev_chips,
             "ev_bb": row.ev_bb} for row in hand.actions],
    }


def main():
    table = TableSession()
    table.game = HoldemGame(button=0, seed=29)
    table.hand_number = 1
    table.opponent = "passive_test"
    table.agent = PassiveAgent()
    traces = []
    state = table.state()
    while not state["done"]:
        request = request_from_game(table.game)
        ranges = experimental_review_ranges(request)
        baseline = solve_trace(request)
        configured = replace(
            request, ranges=ranges, range_source="experimental_illustrative_prior",
            use_context="experimental_live",
            config=SolverConfig(iterations={"preflop": 1, "flop": 1,
                                            "turn": 1, "river": 1000}[request.street],
                                max_joint_deals=3, allow_experimental_flop=True),
        )
        assumed = solve_trace(configured)
        cutoff = None
        if request.street in ("flop", "turn"):
            cutoff = solve_trace(replace(configured, config=SolverConfig(
                iterations=500, max_joint_deals=3,
                continuation_mode="showdown_cutoff")))
        observation = table.game.observe(0)
        index = next((i for i, action in enumerate(state["legal_actions"])
                      if action["kind"] == "call"), None)
        if index is None:
            index = next(i for i, action in enumerate(state["legal_actions"])
                         if action["kind"] == "check")
        chosen = state["legal_actions"][index]
        traces.append({
            "street": request.street, "observation": {
                "board": observation["board"], "hero_hand": observation["hole"],
                "pot": observation["pot"], "stacks": observation["stacks"],
                "street_paid": observation["street_paid"],
                "to_call": observation["to_call"],
                "legal_actions": [action.__dict__ for action in observation["legal_actions"]],
            },
            "request": {
                "board": request.board, "pot_before_street": request.pot_before_street,
                "stacks_at_street_start": request.stacks_at_street_start,
                "button": request.betting.button,
                "first_actor": request.betting.first_actor,
                "current_player": request.betting.current_player,
                "street_actions": [action.__dict__ for action in request.betting.street_actions],
                "range_source": request.range_source,
            },
            "assumed_ranges": [{"hands": belief.hands, "weights": belief.weights}
                               for belief in ranges],
            "live_request": baseline, "assumed_request": assumed,
            "cutoff_request": cutoff,
            "chosen_action": chosen,
        })
        state = table.act(index)
    output = {
        "experiment": "play_and_learn_full_pipeline_diagnosis",
        "seed": 29, "opponent": "passive_call_check",
        "rules": {"stacks": [100, 100], "blinds": [1, 2], "button": 0},
        "traces": traces,
        "hand_history": state["actions"],
        "review_api_statuses": [table.review_decision(1, row["index"])["status"]
                                for row in state["solver_reviews"]],
        "hardware": platform.platform(), "python": platform.python_version(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_tree": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps([{ "street": row["street"], "live": row["live_request"]["reason"],
                        "assumed": row["assumed_request"]["reason"],
                        "assumed_ms": round(row["assumed_request"]["wall_ms"], 1),
                        "cutoff": None if row["cutoff_request"] is None else {
                            "nash_conv": row["cutoff_request"]["nash_conv_chips"],
                            "ms": round(row["cutoff_request"]["wall_ms"], 1)}}
                      for row in traces], indent=2))


if __name__ == "__main__":
    main()
