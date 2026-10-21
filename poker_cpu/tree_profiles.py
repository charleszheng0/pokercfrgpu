"""Finite, reproducible smoke profiles for the restricted postflop solver.

This catalog defines a small declared set of constrained tree shapes; passing
it does not imply support for arbitrary board/range/history combinations.
"""

from dataclasses import dataclass

from .engine import Action
from .river_solver import WeightedRange
from .solve import ActionAbstraction, BettingContext, SolveRequest, SolverConfig


@dataclass(frozen=True)
class TreeProfile:
    profile_id: str
    request: SolveRequest
    note: str


def _request(*, street, board, stack, iterations, abstraction,
             continuation_mode="full_tree", allow_experimental_flop=False,
             street_actions=(), variant="cfr+", chance_sampling=False,
             sampling_seed=0):
    ranges = (
        WeightedRange((("Jc", "Tc"),), (1.0,)),
        WeightedRange((("9h", "9s"),), (1.0,)),
    )
    return SolveRequest(
        board=tuple(board), street=street, pot_before_street=8,
        stacks_at_street_start=(stack, stack),
        betting=BettingContext(button=1, first_actor=0, current_player=0,
                               street_actions=tuple(street_actions)),
        ranges=ranges, abstraction=abstraction,
        config=SolverConfig(
            iterations=iterations, max_joint_deals=1,
            allow_experimental_flop=allow_experimental_flop,
            continuation_mode=continuation_mode,
            variant=variant, chance_sampling=chance_sampling,
            sampling_seed=sampling_seed,
        ),
        acting_hand=("Jc", "Tc"), range_source="finite_tree_profile",
    )


def supported_tree_profiles():
    """Return representative currently supported finite tree fixtures."""
    tiny = ActionAbstraction(
        bet_fractions=(0.75,), raise_fractions=(0.75,),
        max_raises=0, include_min_size=False,
    )
    river_board = ("As", "Kh", "Qd", "7c", "2s")
    turn_board = river_board[:-1]
    flop_board = river_board[:3]
    return (
        TreeProfile("river_root", _request(
            street="river", board=river_board, stack=4, iterations=100,
            abstraction=ActionAbstraction()),
            "River root with min-size, half-pot, pot, and all-in candidates."),
        TreeProfile("river_facing_bet", _request(
            street="river", board=river_board, stack=10, iterations=100,
            abstraction=ActionAbstraction(),
            street_actions=(Action("check"), Action("bet", 4))),
            "River facing a legal half-pot bet with fold/call/two raise targets."),
        TreeProfile("turn_full_tree", _request(
            street="turn", board=turn_board, stack=4, iterations=100,
            abstraction=ActionAbstraction()),
            "Turn betting followed by exact river chance and betting."),
        TreeProfile("turn_showdown_cutoff", _request(
            street="turn", board=turn_board, stack=10, iterations=100,
            abstraction=ActionAbstraction(), continuation_mode="showdown_cutoff"),
            "Turn cutoff with the backend's fixed half-pot/pot/all-in menu; no river betting."),
        TreeProfile("flop_showdown_cutoff", _request(
            street="flop", board=flop_board, stack=4, iterations=100,
            abstraction=ActionAbstraction(), continuation_mode="showdown_cutoff"),
            "Flop cutoff with the fixed menu and exact turn/river runouts; no later betting."),
        TreeProfile("flop_full_tree_experimental", _request(
            street="flop", board=flop_board, stack=1, iterations=1,
            abstraction=tiny, allow_experimental_flop=True),
            "Experimental full flop-to-river tree; one deal and one iteration only."),
        TreeProfile("flop_full_tree_sampled", _request(
            street="flop", board=flop_board, stack=1, iterations=2000,
            abstraction=tiny, allow_experimental_flop=True,
            variant="cfr", chance_sampling=True, sampling_seed=7),
            "Experimental full tree with vanilla CFR and seeded public-chance sampling."),
    )
