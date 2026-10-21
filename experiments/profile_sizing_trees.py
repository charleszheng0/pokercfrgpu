"""Profile finite river betting-tree menus on one fixed explicit-range game.

Run with ``python3 -m experiments.profile_sizing_trees``. CFR+ is run through
the shared solver for every condition; a separate traversal counts reachable
public decision states and CFR information-set/action slots. Tree counts are
structural and wall times are local CPU measurements, not strength claims.
"""

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

from poker_cpu.multi_street import MultiStreetGame, _seven_strength
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig, solve


def build_request(abstraction, iterations=100):
    ranges = (
        WeightedRange((("Jc", "Tc"), ("8h", "8s")), (1.0, 1.0)),
        WeightedRange((("9h", "9s"), ("Ac", "Ad")), (1.0, 1.0)),
    )
    return SolveRequest(
        board=("As", "Kh", "Qd", "7c", "2s"), street="river",
        pot_before_street=20, stacks_at_street_start=(40, 40),
        betting=BettingContext(button=1, first_actor=0, current_player=0),
        ranges=ranges, abstraction=abstraction,
        config=SolverConfig(iterations=iterations, max_joint_deals=16),
        acting_hand=("Jc", "Tc"), range_source="fixed_equal_weight_fixture",
    )


def tree_counts(request):
    abstraction = request.abstraction
    game = MultiStreetGame(
        request.board, [belief.hands for belief in request.ranges],
        pot=request.pot_before_street,
        stack=min(request.stacks_at_street_start),
        button=request.betting.button, big_blind=request.big_blind,
        range_weights=[belief.weights for belief in request.ranges],
        bet_fractions=abstraction.bet_fractions,
        raise_fractions=(abstraction.raise_fractions
                         if abstraction.raise_fractions is not None
                         else abstraction.bet_fractions),
        bet_fractions_by_street=dict(abstraction.bet_fractions_by_street),
        raise_fractions_by_street=dict(abstraction.raise_fractions_by_street),
        max_raises=abstraction.max_raises,
        include_min_size=abstraction.include_min_size,
    )
    root = game.initial_state(request.betting.first_actor)
    public_decisions, public_edges, terminals = set(), set(), set()

    def visit(state):
        if state.done:
            terminals.add(state)
            return
        public_decisions.add(state)
        for action in game.legal(state):
            public_edges.add((state, action))
            visit(game.advance(state, action))

    visit(root)
    infoset_actions = {}
    for hand0, hand1, _ in game.deals:
        hands = (hand0, hand1)
        for state in public_decisions:
            infoset_actions[game.info(state, hands)] = len(game.legal(state))
    return {
        "public_decision_states": len(public_decisions),
        "public_action_edges": len(public_edges),
        "terminal_histories": len(terminals),
        "information_sets": len(infoset_actions),
        "information_set_action_slots": sum(infoset_actions.values()),
        "compatible_joint_deals": len(game.deals),
        "root_actions": [
            {"kind": action.kind, "amount": action.amount}
            for action in game.legal(root)
        ],
    }


def one_run(name, abstraction, iterations):
    request = build_request(abstraction, iterations)
    _seven_strength.cache_clear()
    started = perf_counter()
    result = solve(request)
    wall_ms = (perf_counter() - started) * 1000
    return {
        "name": name,
        "status": result.status,
        "iterations": result.iterations,
        "tree": tree_counts(request),
        "restricted_nash_conv_chips": result.nash_conv_chips,
        "solver_latency_ms": result.latency_ms,
        "wall_latency_ms": wall_ms,
        "selected_hand_strategy": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev_chips": row.ev_chips}
            for row in result.selected_hand.actions
        ] if result.selected_hand else None,
    }


def main():
    iterations, repeats_count = 100, 3
    variants = {
        "baseline_50_100_min_raise_once": ActionAbstraction(),
        "baseline_without_min_size": ActionAbstraction(include_min_size=False),
        "single_75pct_plus_jam": ActionAbstraction(
            bet_fractions=(0.75,), raise_fractions=(0.75,),
            include_min_size=False),
        "dense_25_50_75_100_raise_once": ActionAbstraction(
            bet_fractions=(0.25, 0.5, 0.75, 1.0),
            raise_fractions=(0.5, 1.0), include_min_size=True),
        "baseline_raise_twice": ActionAbstraction(max_raises=2),
        "shark_inspired_river": ActionAbstraction.shark_style_postflop(),
    }
    runs = {name: [one_run(name, abstraction, iterations)
                   for _ in range(repeats_count)]
            for name, abstraction in variants.items()}
    summary = {}
    for name, measurements in runs.items():
        summary[name] = {
            "tree": measurements[-1]["tree"],
            "median_wall_latency_ms": median(row["wall_latency_ms"] for row in measurements),
            "latencies_ms": [row["wall_latency_ms"] for row in measurements],
            "restricted_nash_conv_chips": [row["restricted_nash_conv_chips"] for row in measurements],
        }
    record = {
        "experiment": "fixed_river_sizing_tree_profile",
        "date": date.today().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": "dirty",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version,
        "fixture": {
            "board": ["As", "Kh", "Qd", "7c", "2s"], "pot_chips": 20,
            "stacks": [40, 40], "ranges": [
                {"hands": [["Jc", "Tc"], ["8h", "8s"]], "weights": [1, 1]},
                {"hands": [["9h", "9s"], ["Ac", "Ad"]], "weights": [1, 1]},
            ], "compatible_joint_deals": 4,
        },
        "methodology": (
            "One fixed river board, pot, stack, range, acting seat, and CFR+ budget; "
            "100 iterations, three sequential runs per action menu. Exact restricted "
            "NashConv is measured separately inside each abstract game. Public tree "
            "counts enumerate all legal betting histories; info-set counts span the "
            "four compatible private-hand deals. Latencies are local end-to-end CPU times."
        ),
        "summary": summary, "runs": runs,
        "limitations": [
            "A single river state and four compatible joint deals do not represent broad poker coverage.",
            "Different bet menus define different games; NashConv across menus is not a quality ranking.",
            "One hundred iterations is a structural/latency probe, not a convergence target.",
            "No external solver was used as an accuracy oracle.",
        ],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for name, row in summary.items():
        print(name, "nodes", row["tree"]["public_decision_states"],
              "slots", row["tree"]["information_set_action_slots"],
              "median_ms", round(row["median_wall_latency_ms"], 2),
              "nashconv", [round(x, 4) for x in row["restricted_nash_conv_chips"]])
    print(output)


if __name__ == "__main__":
    main()
