"""Shared, typed CPU solving contract for Solver Lab and future live analysis.

Every supported street uses MultiStreetGame and the same chance-aware CFR
updates. Exact NashConv is measured only within the supplied finite game.
"""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Optional

from .engine import Action
from .multi_street import MultiStreetGame
from .ranges import canonical_hand
from .river_solver import RiverCFR, RiverState, WeightedRange, nash_conv
from .one_street_solver import OneStreetGame


@dataclass(frozen=True)
class BettingContext:
    button: int
    first_actor: int
    current_player: int
    street_actions: tuple[Action, ...] = ()
    street_paid: Optional[tuple[int, int]] = None
    public_history: tuple = ()


@dataclass(frozen=True)
class ActionAbstraction:
    bet_fractions: tuple[float, ...] = (0.5, 1.0)
    max_raises: int = 1


@dataclass(frozen=True)
class SolverConfig:
    variant: str = "cfr+"
    iterations: int = 500
    target_nash_conv_chips: Optional[float] = None
    max_time_ms: Optional[float] = None
    max_joint_deals: int = 64
    allow_experimental_flop: bool = False
    chance_sampling: bool = False
    sampling_seed: int = 0
    continuation_mode: str = "full_tree"


@dataclass(frozen=True)
class SolveRequest:
    board: tuple[str, ...]
    street: str
    pot_before_street: int
    stacks_at_street_start: tuple[int, int]
    betting: BettingContext
    ranges: Optional[tuple[WeightedRange, WeightedRange]]
    abstraction: ActionAbstraction = ActionAbstraction()
    config: SolverConfig = SolverConfig()
    acting_hand: Optional[tuple[str, str]] = None
    selected_action: Optional[Action] = None
    range_source: str = "manual_assumption"
    use_context: str = "lab"
    big_blind: int = 2


@dataclass(frozen=True)
class ActionEstimate:
    action: Action
    frequency: float
    ev_chips: float
    ev_bb: float
    ev_loss_chips: float
    ev_loss_bb: float


@dataclass(frozen=True)
class HandStrategy:
    hand: tuple[str, str]
    marginal_probability: float
    actions: tuple[ActionEstimate, ...]


@dataclass(frozen=True)
class SolveResult:
    status: str
    reason: str
    validation_status: str
    assumptions: tuple[str, ...] = ()
    street: Optional[str] = None
    actions: tuple[Action, ...] = ()
    per_hand: tuple[HandStrategy, ...] = ()
    selected_hand: Optional[HandStrategy] = None
    aggregate_actions: tuple[ActionEstimate, ...] = ()
    selected_action_ev_loss_chips: Optional[float] = None
    selected_action_ev_loss_bb: Optional[float] = None
    iterations: int = 0
    nash_conv_chips: Optional[float] = None
    halfway_nash_conv_chips: Optional[float] = None
    convergence_trace: tuple[tuple[int, float], ...] = ()
    target_nash_conv_chips: Optional[float] = None
    search_latency_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    joint_deals: int = 0
    root_paid: tuple[int, int] = (0, 0)
    root_history: tuple[Action, ...] = ()
    chance_sampling: bool = False
    sampling_seed: Optional[int] = None


def request_from_game(game, *, ranges=None, range_source="unvalidated_inference",
                      config=SolverConfig(), abstraction=ActionAbstraction()):
    """Build a public live request without reading the opponent's private cards.

    The returned request remains unsupported for feedback until ``ranges``
    come from a separately validated betting-history model. Arbitrary played
    bets may also fall outside the requested future action abstraction.
    """
    if game.done:
        raise ValueError("A terminal game has no acting decision")
    observation = game.observe(0)
    street = observation["street"]
    first_actor = game.button if street == 0 else 1 - game.button
    paid = tuple(observation["street_paid"])
    return SolveRequest(
        board=tuple(observation["board"]),
        street=("preflop", "flop", "turn", "river")[street],
        pot_before_street=observation["pot"] - sum(paid),
        stacks_at_street_start=tuple(observation["stacks"][seat] + paid[seat]
                                      for seat in (0, 1)),
        betting=BettingContext(
            button=game.button, first_actor=first_actor,
            current_player=game.current_player,
            street_actions=tuple(Action(kind, amount)
                                 for action_street, _, kind, amount
                                 in observation["history"] if action_street == street),
            street_paid=paid, public_history=tuple(observation["history"]),
        ),
        ranges=ranges, abstraction=abstraction, config=config,
        acting_hand=tuple(observation["hole"]) if game.current_player == 0 else None,
        range_source=range_source, use_context="live",
        big_blind=observation["blinds"][1],
    )


def _unsupported(street, reason):
    return SolveResult("unsupported", reason, "not_validated", street=street)


def _action_rows(actions, frequencies, evs, big_blind):
    best = max(evs.values())
    return tuple(ActionEstimate(
        action=action,
        frequency=frequencies[action],
        ev_chips=evs[action],
        ev_bb=evs[action] / big_blind,
        ev_loss_chips=best - evs[action],
        ev_loss_bb=(best - evs[action]) / big_blind,
    ) for action in actions)


def solve(request: SolveRequest) -> SolveResult:
    """Solve a bounded explicit-range postflop game or explain unsupported input.

    ``pot_before_street`` excludes root-street contributions. ``stacks`` are
    effective chip budgets from that street's start, not current stack display.
    Manual ranges must already reflect all observed prior betting. Live advice
    requires separately validated inferred ranges; none are available today.
    """
    started = perf_counter()
    street = request.street
    if street not in ("preflop", "flop", "turn", "river"):
        return _unsupported(street, "Unknown poker street")
    if street == "preflop":
        return _unsupported(street, "Preflop blueprint and full-hand CFR have not passed validation")
    if request.use_context not in ("lab", "live", "experimental_live"):
        return _unsupported(street, "Unknown solver use context")
    if request.use_context == "live" and request.range_source != "validated_inference":
        return _unsupported(street, "Live feedback requires validated betting-history range inference; none is available")
    if (request.use_context == "experimental_live"
            and not request.range_source.startswith("experimental_")):
        return _unsupported(street, "Experimental live review requires an explicitly labeled assumed range source")
    if request.ranges is None:
        return _unsupported(street, "Both player ranges are required; automatic range inference is not validated")
    if request.config.continuation_mode not in ("full_tree", "showdown_cutoff"):
        return _unsupported(street, "Unknown continuation mode")
    cutoff = request.config.continuation_mode == "showdown_cutoff"
    if cutoff and street not in ("flop", "turn"):
        return _unsupported(street, "Showdown cutoff currently supports flop and turn only")
    if cutoff and request.config.chance_sampling:
        return _unsupported(street, "Showdown cutoff already enumerates later runouts exactly; chance sampling is not supported")
    if street == "flop" and not cutoff and not request.config.allow_experimental_flop:
        return _unsupported(street, "Full flop-to-river exact search is experimental and disabled for app feedback")
    if type(request.config.iterations) is not int or not 1 <= request.config.iterations <= 5000:
        return _unsupported(street, "Iteration count must be between 1 and 5000")
    if request.config.variant not in ("cfr", "cfr+", "dcfr"):
        return _unsupported(street, "Unknown CFR update variant")
    if type(request.config.chance_sampling) is not bool:
        return _unsupported(street, "Chance sampling flag must be Boolean")
    if request.config.chance_sampling and request.config.variant != "cfr":
        return _unsupported(street, "Chance sampling currently supports vanilla CFR only")
    if type(request.config.sampling_seed) is not int:
        return _unsupported(street, "Sampling seed must be an integer")
    target = request.config.target_nash_conv_chips
    if target is not None and (type(target) not in (int, float)
                               or not isfinite(target) or target < 0):
        return _unsupported(street, "NashConv target must be a finite nonnegative chip amount")
    budget = request.config.max_time_ms
    if budget is not None and (type(budget) not in (int, float)
                               or not isfinite(budget) or budget <= 0):
        return _unsupported(street, "Time budget must be positive and finite")
    if request.big_blind <= 0:
        return _unsupported(street, "Big blind must be positive")
    if (len(request.stacks_at_street_start) != 2
            or any(type(value) is not int or value < 1
                   for value in request.stacks_at_street_start)):
        return _unsupported(street, "Two positive integer stacks are required")
    if (request.betting.button not in (0, 1)
            or request.betting.first_actor != 1 - request.betting.button
            or request.betting.current_player not in (0, 1)):
        return _unsupported(street, "Postflop first actor must be the big blind; acting seat must be valid")
    if cutoff and request.abstraction != ActionAbstraction():
        return _unsupported(street, "Showdown cutoff currently uses the fixed half-pot/pot/all-in action abstraction")
    if len(request.betting.street_actions) > 5:
        return _unsupported(street, "Betting history exceeds this restricted action tree")
    try:
        board = tuple(request.board)
        if len(board) != {"flop": 3, "turn": 4, "river": 5}[street]:
            return _unsupported(street, f"{street.capitalize()} requires the corresponding public board")
        ranges = tuple(
            WeightedRange(tuple(canonical_hand(hand) for hand in belief.hands),
                          belief.weights)
            for belief in request.ranges
        )
        if cutoff:
            compatible_count = sum(
                1 for hand0, weight0 in zip(ranges[0].hands, ranges[0].weights)
                for hand1, weight1 in zip(ranges[1].hands, ranges[1].weights)
                if weight0 > 0 and weight1 > 0 and not set(hand0) & set(hand1)
                and not set(hand0 + hand1) & set(board)
            )
            cap = min(8, request.config.max_joint_deals)
            if compatible_count > cap:
                return _unsupported(street, f"{compatible_count} compatible deals exceed the {cap}-deal cutoff CPU limit")
            game = OneStreetGame(
                board, tuple(belief.hands for belief in ranges),
                pot=request.pot_before_street,
                stack=min(request.stacks_at_street_start),
                big_blind=request.big_blind,
                range_weights=tuple(belief.weights for belief in ranges),
            )
        else:
            game = MultiStreetGame(
                board, tuple(belief.hands for belief in ranges),
                pot=request.pot_before_street,
                stack=min(request.stacks_at_street_start),
                button=request.betting.button,
                big_blind=request.big_blind,
                range_weights=tuple(belief.weights for belief in ranges),
                bet_fractions=request.abstraction.bet_fractions,
                max_raises=request.abstraction.max_raises,
            )
        if len(game.deals) > request.config.max_joint_deals:
            return _unsupported(street, f"{len(game.deals)} compatible deals exceed the configured {request.config.max_joint_deals}-deal CPU limit")
        if cutoff and len(game.deals) > 8:
            return _unsupported(street, "Current-street cutoff supports at most eight compatible deals")
        if not cutoff and street == "turn" and (len(game.deals) > 4 or request.config.iterations > (1000 if request.config.chance_sampling else 100)):
            return _unsupported(street, "Turn-to-river search currently supports at most four deals and 100 exact or 1000 sampled iterations")
        if not cutoff and street == "flop" and (len(game.deals) > (4 if request.config.chance_sampling else 1) or request.config.iterations > (2000 if request.config.chance_sampling else 3)):
            return _unsupported(street, "Experimental flop search currently supports four deals/2000 sampled iterations or one deal/three exact iterations")
        root = (RiverState(current=request.betting.first_actor,
                           last_raise=request.big_blind) if cutoff else
                game.initial_state(request.betting.first_actor))
        for action in request.betting.street_actions:
            if action not in game.legal(root):
                return _unsupported(street, "Observed action is outside this restricted betting tree")
            root = game.advance(root, action)
        if root.done or getattr(root, "chance", False) or root.current != request.betting.current_player:
            return _unsupported(street, "History does not end at the requested player's decision")
        if request.betting.street_paid is not None and root.paid != request.betting.street_paid:
            return _unsupported(street, "Street contributions disagree with the supplied action history")
        actions = game.legal(root)
        if request.selected_action is not None and request.selected_action not in actions:
            return _unsupported(street, "Selected action is outside this betting abstraction")
        hero = canonical_hand(request.acting_hand) if request.acting_hand else None
        if hero is not None and not any(
            (hand0, hand1)[root.current] == hero for hand0, hand1, _ in game.deals
        ):
            return _unsupported(street, "Selected hand has no compatible opponent range mass")
    except (TypeError, ValueError, KeyError) as exc:
        return _unsupported(street, f"Invalid solver position: {exc}")

    solver = RiverCFR(game, variant=request.config.variant,
                      root_player=root.current, root_state=root,
                      chance_sampling=request.config.chance_sampling,
                      sampling_seed=request.config.sampling_seed)
    checkpoint = max(1, request.config.iterations // 2)
    search_ms = 0.0
    for _ in range(checkpoint):
        step_started = perf_counter()
        solver.step()
        search_ms += (perf_counter() - step_started) * 1000
        if request.config.max_time_ms is not None and (perf_counter() - started) * 1000 >= request.config.max_time_ms:
            break
    halfway = nash_conv(game, solver.average, solver.average, root_state=root)
    trace = [(solver.iterations, halfway)]
    latest_nash_conv = halfway
    check_every = max(1, request.config.iterations // 10)
    next_check = solver.iterations + check_every
    while solver.iterations < request.config.iterations:
        if target is not None and latest_nash_conv <= target:
            break
        if request.config.max_time_ms is not None and (perf_counter() - started) * 1000 >= request.config.max_time_ms:
            break
        step_started = perf_counter()
        solver.step()
        search_ms += (perf_counter() - step_started) * 1000
        if target is not None and solver.iterations >= next_check:
            latest_nash_conv = nash_conv(game, solver.average, solver.average,
                                         root_state=root)
            trace.append((solver.iterations, latest_nash_conv))
            next_check += check_every
    report = solver.analyze_root(player=root.current, private_hand=hero,
                                 chosen_action=request.selected_action)
    if trace[-1][0] != solver.iterations:
        trace.append((solver.iterations, report["nash_conv"]))
    marginal = {}
    for deal in game.deals:
        hand = deal[root.current]
        marginal[hand] = marginal.get(hand, 0.0) + deal[2]
    per_hand = tuple(HandStrategy(
        hand=row["hand"], marginal_probability=marginal[row["hand"]],
        actions=_action_rows(actions, row["frequency"], row["action_evs"],
                             request.big_blind),
    ) for row in report["per_hand"])
    selected = next((row for row in per_hand if row.hand == hero), None)
    aggregate = _action_rows(actions, report["action_frequencies"],
                             report["conditional_action_evs"], request.big_blind)
    loss = report["chosen_action_ev_loss"]
    if loss is None and request.selected_action is not None and selected is None:
        loss = next(row.ev_loss_chips for row in aggregate
                    if row.action == request.selected_action)
    range_assumption = (
        f"{request.range_source}: illustrative priors are not conditioned on earlier public actions"
        if request.use_context == "experimental_live" else
        f"{request.range_source}: supplied priors must already reflect earlier public actions"
    )
    assumptions = (
        range_assumption,
        "Exact best response only inside this small abstract game",
        "No safe re-solving or opponent counterfactual-value constraints",
        f"Bet fractions {request.abstraction.bet_fractions}; at most {request.abstraction.max_raises} raise(s) per street",
        "EV is zero-sum net chips from the root street's start; prior root-street contributions are included",
        ("No betting after the current street; remaining board runouts are enumerated exactly"
         if cutoff else
         "Training samples public-card chance with a fixed seed; final EV and NashConv enumerate exact chance"
         if request.config.chance_sampling else "Training enumerates public-card chance exactly"),
    )
    quality_prefix = ("research_sampled" if request.config.chance_sampling else
                      "experimental_showdown_cutoff" if cutoff else
                      "experimental_assumed_range" if request.use_context == "experimental_live"
                      else "restricted")
    quality = (f"{quality_prefix}_target_met" if target is not None
               and report["nash_conv"] <= target
               else f"{quality_prefix}_residual_measured")
    return SolveResult(
        status="supported_restricted_game",
        reason=("Experimental chance-sampled restricted game; strategy is not full-game GTO"
                if request.config.chance_sampling else
                "Experimental current-street CFR with exact showdown runouts and no later betting; not full-game GTO"
                if cutoff else
                "Experimental assumed-range restricted game; strategy is not full-game GTO"
                if request.use_context == "experimental_live" else
                "Restricted postflop game; strategy is not full-game GTO"),
        validation_status=quality,
        assumptions=assumptions, street=street, actions=actions,
        per_hand=per_hand, selected_hand=selected,
        aggregate_actions=aggregate,
        selected_action_ev_loss_chips=loss,
        selected_action_ev_loss_bb=loss / request.big_blind if loss is not None else None,
        iterations=solver.iterations,
        nash_conv_chips=report["nash_conv"],
        halfway_nash_conv_chips=halfway,
        convergence_trace=tuple(trace),
        target_nash_conv_chips=target,
        search_latency_ms=search_ms,
        latency_ms=(perf_counter() - started) * 1000,
        joint_deals=len(game.deals), root_paid=root.paid,
        root_history=request.betting.street_actions,
        chance_sampling=request.config.chance_sampling,
        sampling_seed=request.config.sampling_seed if request.config.chance_sampling else None,
    )
