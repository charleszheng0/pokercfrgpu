"""Reproducible opt-in gameplay review on one seeded passive hand."""

import json
import platform
import subprocess
from pathlib import Path

from poker_cpu.engine import HoldemGame
from trainer.server import TableSession


class PassiveAgent:
    def act(self, observation):
        call = next((action for action in observation["legal_actions"]
                     if action.kind == "call"), None)
        return call or next(action for action in observation["legal_actions"]
                            if action.kind == "check")


def main():
    table = TableSession()
    table.game = HoldemGame(button=0, seed=29)
    table.hand_number = 1
    table.opponent = "passive_test"
    table.agent = PassiveAgent()
    state = table.state()
    while not state["done"]:
        index = next((i for i, action in enumerate(state["legal_actions"])
                      if action["kind"] == "call"), None)
        if index is None:
            index = next(i for i, action in enumerate(state["legal_actions"])
                         if action["kind"] == "check")
        state = table.act(index)
    river = next(item for item in state["solver_reviews"]
                 if item["street"] == "river")
    result = table.review_decision(1, river["index"])
    output = {
        "experiment": "seeded_experimental_gameplay_review",
        "seed": 29, "rules": {"stacks": [100, 100], "blinds": [1, 2],
                            "button": 0, "opponent": "passive_call_check"},
        "decision": {key: value for key, value in river.items() if key != "analysis"},
        "result": result,
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_tree": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "nash_conv_chips": result.get("nash_conv_chips"),
                      "iterations": result.get("iterations"),
                      "latency_ms": result.get("latency_ms"),
                      "selected_ev_loss_chips": result.get("selected_action_ev_loss_chips")}))


if __name__ == "__main__":
    main()
