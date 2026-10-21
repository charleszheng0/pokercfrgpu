"""Run the 30-position batch on the Shark 2.0-inspired full-tree menu."""
import json
from pathlib import Path

from experiments.postflop_validation_batch import run_batch


def main():
    record = run_batch(shark_style=True)
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["profiles"], indent=2))
    print("total seconds", round(record["total_wall_seconds"], 1))
    print(path)


if __name__ == "__main__":
    main()
