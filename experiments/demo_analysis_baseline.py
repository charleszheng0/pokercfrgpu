"""Reproduce the fixed one-click river demo's real CFR+ output."""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

from trainer.demo import DEMO_RIVER_REQUEST
from trainer.server import analyze_demo_position


def main():
    result = analyze_demo_position()
    report = result["analysis"]
    record = {
        "experiment": "one_click_restricted_river_demo",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty", "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "request": DEMO_RIVER_REQUEST,
        "methodology": "One local CPU run of the exact trainer demo route; end-to-end latency includes setup, 2000 synchronous CFR+ iterations and exact finite-game best-response/action-EV analysis. One run is not a latency distribution. The browser test separately checks rendering and action comparison.",
        "result": result,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)
    print("NashConv", report["nash_conv"], "latency_ms", report["latency_ms"])
    print("actions", [(row["action"], row["ev"], row["ev_loss"])
                      for row in report["action_evs"]])


if __name__ == "__main__":
    main()
