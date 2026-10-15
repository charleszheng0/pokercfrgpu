"""Record a short-budget explicit-range river adapter request."""

import json
import platform
import subprocess
from pathlib import Path

from poker_cpu.engine import Action
from poker_cpu.river_solver import RiverGame, RiverState, WeightedRange
from poker_cpu.solver_interface import SolverRequest, TinyRiverCFRBackend

BOARD = "As Kh Qd 7c 2s"
RANGES = (
    WeightedRange((("Jc", "Tc"), ("9h", "9s"), ("3c", "3d")), (2, 1, 1)),
    WeightedRange((("Jd", "Td"), ("8h", "8s"), ("4c", "4d")), (1, 1, 2)),
)


def run(budget_ms=10):
    game = RiverGame(
        BOARD,
        tuple(belief.hands for belief in RANGES),
        pot=8,
        stack=10,
        range_weights=tuple(belief.weights for belief in RANGES),
    )
    request = SolverRequest(
        public_state={"street": "river", "board": BOARD},
        acting_player=0,
        ranges=RANGES,
        legal_action_abstraction=game.legal(RiverState()),
        stacks=(10, 10),
        pot=8,
        betting_history=(),
        computation_budget_ms=budget_ms,
        chosen_action=Action("check"),
        acting_hand=("Jc", "Tc"),
    )
    result = TinyRiverCFRBackend().analyze(request)
    return {
        "experiment": "short_budget_tiny_river_adapter_smoke",
        "budget_ms_search_only": budget_ms,
        "board": BOARD,
        "ranges": [
            {"hands": belief.hands, "weights": belief.weights} for belief in RANGES
        ],
        "pot": 8,
        "effective_stack": 10,
        "action_abstraction": [str(action) for action in game.legal(RiverState())],
        "status": result.status,
        "validation_status": result.validation_status,
        "iterations": result.iterations,
        "search_latency_ms": result.search_latency_ms,
        "end_to_end_latency_ms": result.latency_ms,
        "acting_hand_action_frequencies": result.action_frequencies,
        "acting_hand_action_evs_chips": result.action_evs,
        "acting_hand_check_ev_gap_chips": result.ev_loss,
        "nash_conv_chips": result.convergence["nash_conv_chips"],
        "python": platform.python_version(),
        "machine": platform.machine(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }


if __name__ == "__main__":
    output = Path("experiments/river_adapter_budget.json")
    result = run()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        key: result[key] for key in (
            "status", "validation_status", "iterations", "search_latency_ms",
            "end_to_end_latency_ms", "nash_conv_chips", "acting_hand_check_ev_gap_chips"
        )
    }, indent=2))
