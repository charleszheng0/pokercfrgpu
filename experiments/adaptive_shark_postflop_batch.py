"""Run the same quality-targeted batch on Shark-inspired full-tree menus."""
import json
from pathlib import Path

from experiments.adaptive_postflop_batch import run_adaptive


def main():
    record = run_adaptive(shark_style=True)
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["profiles"], indent=2))
    print("total seconds", round(record["total_wall_seconds"], 1))
    print(path)


if __name__ == "__main__":
    main()
