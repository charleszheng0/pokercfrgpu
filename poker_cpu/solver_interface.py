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
    convergence: Optional[Any] = None
    confidence: Optional[Any] = None


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
