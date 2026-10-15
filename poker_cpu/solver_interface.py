"""Typed boundary for future validated public-state search analysis.

The current Hold'em solver cannot answer full-game requests. Its outputs are
therefore explicitly absent instead of being filled with policy probabilities
or extrapolated river values.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple


@dataclass(frozen=True)
class SolverRequest:
    public_state: Any
    acting_player: int
    ranges: Optional[Tuple[Any, Any]]
    legal_action_abstraction: Tuple[Any, ...]
    stacks: Tuple[int, int]
    pot: int
    betting_history: Tuple[Any, ...]
    computation_budget_ms: int
    chosen_action: Optional[Any] = None
    acting_hand: Optional[Tuple[str, str]] = None


@dataclass(frozen=True)
class SolverAnalysis:
    status: str
    validation_status: str
    reason: str
    action_frequencies: Optional[Tuple[float, ...]] = None
    action_evs: Optional[Tuple[float, ...]] = None
    ev_loss: Optional[float] = None
    iterations: Optional[int] = None
    latency_ms: Optional[float] = None
    search_latency_ms: Optional[float] = None
    convergence: Optional[Any] = None
    confidence: Optional[Any] = None
    ev_conditioning: Optional[str] = None


class SolverBackend(Protocol):
    def analyze(self, request: SolverRequest) -> SolverAnalysis: ...


class UnsupportedHoldemSolver:
    """Honest placeholder until a full-state backend passes validation gates."""

    def analyze(self, request: SolverRequest) -> SolverAnalysis:
        del request
        return SolverAnalysis(
            status="unsupported",
            validation_status="not_validated",
            reason=(
                "No full-hand Hold'em solver currently supports this state. "
                "The research CFR implementation is limited to tiny explicit-range river games."
            ),
        )


class TinyRiverCFRBackend:
    """Adapter for explicit-range, root-only river games.

    It intentionally refuses requests requiring range inference, a prior
    river action history, or an action set different from the solver's own.
    Earlier street history may be supplied as (street, player, kind, amount)
    tuples, but the supplied ranges must already be conditioned on that history.
    ``computation_budget_ms`` is a soft budget because a complete CFR iteration
    is atomic and may exceed the requested duration.
    """

    def __init__(self, variant="cfr+"):
        self.variant = variant

    def analyze(self, request: SolverRequest) -> SolverAnalysis:
        from .engine import Action
        from .river_solver import RiverGame, RiverState, WeightedRange
        from .solve import (BettingContext, SolveRequest as SharedSolveRequest,
                            SolverConfig, solve)

        state = request.public_state
        if not isinstance(state, dict) or not state.get("board"):
            return _unsupported("A five-card public board is required.")
        board = state["board"]
        if len(board.split() if isinstance(board, str) else board) != 5:
            return _unsupported("Only complete river boards are supported.")
        if state.get("street") not in (None, "river", 3):
            return _unsupported("The public street marker does not identify a river state.")
        for event in request.betting_history:
            if isinstance(event, dict):
                if event.get("type") == "action" and event.get("street") == 3:
                    return _unsupported("Prior river actions are outside the root-only abstraction.")
            elif isinstance(event, (tuple, list)) and len(event) >= 3 and type(event[0]) is int:
                if event[0] >= 3:
                    return _unsupported("Prior river actions are outside the root-only abstraction.")
            else:
                return _unsupported("Betting history must identify the street of each action.")
        if "current_player" in state and state["current_player"] != request.acting_player:
            return _unsupported("The acting seat does not match the public state.")
        street_paid = state.get("street_paid", (0, 0))
        if len(street_paid) != 2 or any(street_paid):
            return _unsupported("Only the first decision of a river betting round is supported.")
        if request.acting_player not in (0, 1) or len(request.stacks) != 2:
            return _unsupported("A valid acting seat and two stack sizes are required.")
        if min(request.stacks) <= 0 or request.pot < 0:
            return _unsupported("Positive effective stacks and a nonnegative pot are required.")
        if request.computation_budget_ms <= 0:
            return _unsupported("A positive computation budget is required.")
        if request.ranges is None or len(request.ranges) != 2:
            return _unsupported("Both explicit private-hand ranges are required.")
        if not isinstance(request.legal_action_abstraction, tuple):
            return _unsupported("Legal actions must be supplied as an ordered tuple.")
        try:
            beliefs = tuple(
                value if isinstance(value, WeightedRange) else WeightedRange(
                    tuple(value["hands"]), tuple(value["weights"])
                )
                for value in request.ranges
            )
            game = RiverGame(
                board,
                tuple(belief.hands for belief in beliefs),
                pot=request.pot,
                stack=min(request.stacks),
                range_weights=tuple(belief.weights for belief in beliefs),
            )
            expected_actions = game.legal(RiverState(current=request.acting_player))
            if request.legal_action_abstraction != expected_actions:
                return _unsupported("Requested actions do not match the supported river abstraction.")
            if request.chosen_action is not None and request.chosen_action not in expected_actions:
                return _unsupported("The chosen action is outside the supported river abstraction.")
            if request.acting_hand is not None:
                acting_hand = tuple(request.acting_hand)
                if acting_hand not in game.ranges[request.acting_player]:
                    return _unsupported("The acting hand is absent from the supplied range.")
        except (KeyError, TypeError, ValueError) as exc:
            return _unsupported(f"Invalid river request: {exc}")

        shared = solve(SharedSolveRequest(
            board=tuple(board.split() if isinstance(board, str) else board),
            street="river", pot_before_street=request.pot,
            stacks_at_street_start=tuple(request.stacks),
            betting=BettingContext(button=1 - request.acting_player,
                                   first_actor=request.acting_player,
                                   current_player=request.acting_player),
            ranges=beliefs,
            config=SolverConfig(variant=self.variant, iterations=5000,
                                max_time_ms=request.computation_budget_ms),
            acting_hand=request.acting_hand,
            selected_action=request.chosen_action,
        ))
        if shared.status != "supported_restricted_game":
            return _unsupported(shared.reason)
        action_rows = (shared.selected_hand.actions if shared.selected_hand
                       else shared.aggregate_actions)
        return SolverAnalysis(
            status="river_abstracted",
            validation_status="tested_restricted_implementation",
            reason=(
                "Approximate average strategy and conditional action EVs for a "
                "restricted root river game. Supplied ranges must reflect earlier "
                "public history; these are not full-game GTO values."
            ),
            action_frequencies=tuple(row.frequency for row in action_rows),
            action_evs=tuple(row.ev_chips for row in action_rows),
            ev_loss=shared.selected_action_ev_loss_chips,
            iterations=shared.iterations,
            latency_ms=shared.latency_ms,
            search_latency_ms=shared.search_latency_ms,
            convergence={"nash_conv_chips": shared.nash_conv_chips},
            confidence={
                "scope": "exact best-response diagnostic within supplied finite abstraction",
                "joint_deals": shared.joint_deals,
                "action_order": shared.actions,
            },
            ev_conditioning=(
                "conditional_on_acting_hand" if shared.selected_hand is not None
                else "averaged_over_supplied_range"
            ),
        )


def _unsupported(reason):
    return SolverAnalysis(
        status="unsupported",
        validation_status="not_validated",
        reason=reason,
    )
