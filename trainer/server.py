"""Local HTTP table; all state transitions use poker_cpu.engine.HoldemGame."""

import argparse
import json
import secrets
import time
from dataclasses import replace
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, RLock

from poker_cpu.agents import HeuristicAgent, RandomAgent
from poker_cpu.cards import parse_cards
from poker_cpu.engine import Action, HoldemGame
from poker_cpu.ranges import canonical_hand, preset_catalog
from poker_cpu.river_solver import WeightedRange
from poker_cpu.solve import BettingContext, SolveRequest, SolverConfig, request_from_game, solve
from .demo import DEMO_DESCRIPTION, DEMO_RIVER_REQUEST
from .solve_debug import log_event
from . import offline_training
from .solution_cache import demo_solution, sample_demo_action

STATIC = Path(__file__).with_name("static")
EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
NFSP_MODEL = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "holdem_nfsp"
    / "seed11"
    / "policy_player_1.pt"
)
NFSP_HERO_MODEL = NFSP_MODEL.with_name("policy_player_0.pt")
SESSIONS = {}
LOCK = Lock()


def live_analysis_availability(street, history, game=None):
    names = ("preflop", "flop", "turn", "river")
    if street not in range(4):
        raise ValueError("Unknown Hold'em street")
    shared_request = request_from_game(game) if game is not None and not game.done else SolveRequest(
        board=tuple(game.board) if game is not None else (),
        street=names[street],
        pot_before_street=(game.pot - sum(game.street_paid)) if game is not None else 0,
        stacks_at_street_start=(tuple(game.stacks[i] + game.street_paid[i]
                                      for i in (0, 1)) if game is not None else (1, 1)),
        betting=BettingContext(
            button=game.button if game is not None else 0,
            first_actor=1 - game.button if game is not None else 1,
            current_player=game.current_player if game is not None and not game.done else 0,
            street_actions=tuple(Action(kind, amount) for action_street, _, kind, amount
                                 in history if action_street == street),
            street_paid=tuple(game.street_paid) if game is not None else None,
            public_history=tuple(history),
        ),
        ranges=None, acting_hand=tuple(game.hole[0]) if game is not None else None,
        range_source="unvalidated_inference", use_context="live",
    )
    shared = solve(shared_request)
    log_event("live_availability", street=shared_request.street,
              board=shared_request.board,
              pot_before_street=shared_request.pot_before_street,
              stacks_at_street_start=shared_request.stacks_at_street_start,
              range_source=shared_request.range_source,
              has_ranges=shared_request.ranges is not None,
              status=shared.status, reason=shared.reason)
    if street < 3:
        reason = (
            f"{names[street].capitalize()} solver feedback is unavailable: "
            f"{shared.reason}. Full-street advice has not passed validation."
        )
    elif any(action[0] == 3 for action in history):
        reason = (
            "Live river feedback is unavailable after earlier river actions: "
            f"{shared.reason}. This wager history has not passed range validation. "
            "Solver Lab accepts a small manually conditioned abstract river state."
        )
    else:
        reason = (
            f"Live river feedback is unavailable: {shared.reason}. "
            "Solver Lab accepts small manual ranges."
        )
    return {"status": "unavailable", "street": names[street], "reason": reason,
            "solver_api_status": shared.status}


def analyze_river_position(payload):
    """Analyze a supplied-range postflop decision through the shared solver."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    street = payload.get("street", "river")
    if street not in ("flop", "turn", "river"):
        raise ValueError("Unsupported state: only postflop research solves are available")
    try:
        board = parse_cards(payload.get("board", ""))
        hero_hand = canonical_hand(payload.get("hero_hand", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cards: {exc}") from exc
    if len(board) != {"flop": 3, "turn": 4, "river": 5}[street]:
        raise ValueError(f"Unsupported state: analysis requires a complete {street} board")
    if len(hero_hand) != 2 or set(board) & set(hero_hand):
        raise ValueError("Hero hand must contain two distinct cards outside the board")
    try:
        pot = payload["pot"]
        stacks = payload["stacks"]
        iterations = payload.get("iterations", 500)
    except KeyError as exc:
        raise ValueError(f"Missing required field: {exc.args[0]}") from exc
    if type(pot) is not int or pot < 0:
        raise ValueError("Pot must be a nonnegative integer chip amount")
    if (not isinstance(stacks, list) or len(stacks) != 2
            or any(type(stack) is not int or stack < 1 for stack in stacks)):
        raise ValueError("Stacks must be two positive integer chip amounts")
    maximum = {"river": 5000, "turn": 100, "flop": 3}[street]
    minimum = 20 if street == "river" else 1
    if type(iterations) is not int or not minimum <= iterations <= maximum:
        raise ValueError(f"Iteration budget must be an integer from {minimum} to {maximum} for {street}")
    ranges = payload.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 2:
        raise ValueError("Supply two explicit weighted ranges")
    hands, weights = [], []
    for seat, item in enumerate(ranges):
        if not isinstance(item, dict) or not isinstance(item.get("hands"), list) or not isinstance(item.get("weights"), list):
            raise ValueError(f"Seat {seat + 1} range needs hands and weights arrays")
        if not item["hands"] or len(item["hands"]) > 8 or len(item["hands"]) != len(item["weights"]):
            raise ValueError("Each range must contain 1-8 hands with one weight per hand")
        parsed = []
        for hand in item["hands"]:
            cards = canonical_hand(hand)
            if len(cards) != 2:
                raise ValueError(f"Every seat {seat + 1} range entry must be a two-card hand")
            parsed.append(cards)
        hands.append(tuple(parsed))
        weights.append(tuple(item["weights"]))
    if hero_hand not in hands[0]:
        raise ValueError("Hero hand must appear in seat 1's supplied range")

    first_actor = payload.get("first_actor", 0)
    if type(first_actor) is not int or first_actor not in (0, 1):
        raise ValueError("First river actor must be hero or opponent")
    previous_actions = payload.get("street_actions", payload.get("river_actions", []))
    if not isinstance(previous_actions, list) or len(previous_actions) > 2:
        raise ValueError("Unsupported state: supply at most two prior abstract river actions")
    actions = []
    for previous in previous_actions:
        if not isinstance(previous, dict) or previous.get("kind") not in ("check", "bet", "raise"):
            raise ValueError("Unsupported state: prior river actions must be check, bet, or raise")
        amount = previous.get("amount", 0)
        if type(amount) is not int:
            raise ValueError("Prior river action targets must be integer chips")
        actions.append(Action(previous["kind"], amount))
    try:
        beliefs = tuple(WeightedRange(hand, weight)
                        for hand, weight in zip(hands, weights))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid range: {exc}") from exc
    report = solve(SolveRequest(
        board=board, street=street, pot_before_street=pot,
        stacks_at_street_start=tuple(stacks),
        betting=BettingContext(button=1 - first_actor,
                               first_actor=first_actor, current_player=0,
                               street_actions=tuple(actions)),
        ranges=beliefs, config=SolverConfig(
            iterations=iterations,
            target_nash_conv_chips=payload.get("target_nash_conv_chips"),
            max_time_ms=10000 if street != "river" else None,
            allow_experimental_flop=payload.get("allow_experimental_flop") is True,
        ),
        acting_hand=hero_hand, range_source="manual_assumption",
    ))
    if report.status != "supported_restricted_game":
        if "Selected hand has no compatible" in report.reason:
            raise ValueError("Unsupported state: hero hand has no compatible opponent range mass")
        if "Observed action is outside" in report.reason:
            raise ValueError("Unsupported state: prior river action is outside the restricted tree")
        if "History does not end" in report.reason:
            raise ValueError("Unsupported state: hero must be facing a nonterminal river decision")
        raise ValueError(f"Unsupported state: {report.reason}")
    hero_rows = report.selected_hand.actions
    freq_rows = [{"action": {"kind": row.action.kind, "amount": row.action.amount},
                  "frequency": row.frequency} for row in hero_rows]
    ev_rows = [{"action": {"kind": row.action.kind, "amount": row.action.amount},
                "ev": row.ev_chips, "ev_bb": row.ev_bb,
                "ev_loss": row.ev_loss_chips,
                "ev_loss_bb": row.ev_loss_bb} for row in hero_rows]
    delta = report.nash_conv_chips - report.halfway_nash_conv_chips
    trend = "decreased" if delta < -1e-9 else "increased" if delta > 1e-9 else "unchanged"
    return {
        "status": f"restricted_{street}_analysis",
        "label": f"Restricted {street} game · supplied ranges · CFR+",
        "validation_status": "exact NashConv for this finite restricted game",
        "board": board,
        "street": street,
        "hero_hand": hero_hand,
        "pot": pot,
        "big_blind": 2,
        "effective_stack": min(stacks),
        "first_actor": first_actor,
        "river_actions": [{"kind": action.kind, "amount": action.amount} for action in report.root_history],
        "river_paid": report.root_paid,
        "street_actions": [{"kind": action.kind, "amount": action.amount} for action in report.root_history],
        "street_paid": report.root_paid,
        "iterations": report.iterations,
        "latency_ms": report.latency_ms,
        "search_latency_ms": report.search_latency_ms,
        "joint_deals": report.joint_deals,
        "solver_quality_status": report.validation_status,
        "target_nash_conv_chips": report.target_nash_conv_chips,
        "convergence_trace": [
            {"iterations": iteration, "nash_conv_chips": value}
            for iteration, value in report.convergence_trace
        ],
        "solver_assumptions": report.assumptions,
        "nash_conv": report.nash_conv_chips,
        "nash_conv_halfway": report.halfway_nash_conv_chips,
        "nash_conv_trend": trend,
        "nash_conv_delta": delta,
        "frequencies": freq_rows,
        "action_evs": ev_rows,
        "per_hand": [
            {"hand": row.hand, "marginal_probability": row.marginal_probability,
             "actions": [{"kind": estimate.action.kind,
                          "amount": estimate.action.amount,
                          "frequency": estimate.frequency,
                          "ev": estimate.ev_chips,
                          "ev_bb": estimate.ev_bb,
                          "ev_loss": estimate.ev_loss_chips,
                          "ev_loss_bb": estimate.ev_loss_bb}
                         for estimate in row.actions]}
            for row in report.per_hand
        ],
        "aggregate_strategy": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev": row.ev_chips,
             "ev_bb": row.ev_bb, "ev_loss": row.ev_loss_chips,
             "ev_loss_bb": row.ev_loss_bb}
            for row in report.aggregate_actions
        ],
        "ev_basis": "conditional chip EV for the entered hero hand; opponent hands are weighted by supplied range and blockers; average CFR+ continuation policy",
        "range_note": "Manual ranges are user-supplied beliefs. They are not inferred from betting history.",
        "supported_state": f"{street.capitalize()} decision after the supplied abstract action history; pot is the amount before {street} betting, and supplied ranges must already reflect observed actions. Restricted bet sizes and one raise maximum.",
    }


def analyze_demo_position():
    """Solve the fixed trainer decision through the same explicit-range API."""
    report = analyze_river_position(DEMO_RIVER_REQUEST)
    request = DEMO_RIVER_REQUEST
    return {
        "position": {
            "street": "river",
            "board": report["board"],
            "hero_hand": report["hero_hand"],
            "pot": report["pot"],
            "stacks": request["stacks"],
            "big_blind": report["big_blind"],
            "description": DEMO_DESCRIPTION,
            "range_assumptions": [
                [{"hand": hand,
                  "weight_percent": 100 * weight / sum(seat["weights"])}
                 for hand, weight in zip(seat["hands"], seat["weights"])]
                for seat in request["ranges"]
            ],
        },
        "analysis": report,
    }


def experimental_review_ranges(request):
    """Tiny, disclosed illustrative prior; never conditioned on betting actions."""
    hero = canonical_hand(request.acting_hand)
    blocked = set(request.board) | set(hero)
    candidates = (
        ("Ac", "Ad"), ("Kh", "Kd"), ("Qs", "Js"), ("9h", "9s"),
        ("7h", "6h"), ("4c", "4d"), ("Tc", "9c"), ("3h", "3s"),
    )
    opponent = tuple(canonical_hand(hand) for hand in candidates
                     if not blocked.intersection(hand))[:3]
    if len(opponent) < 2:
        raise ValueError("No illustrative opponent range survives public-card removal")
    return (WeightedRange((hero,), (1.0,)),
            WeightedRange(opponent, tuple(1.0 for _ in opponent)))


def serialize_experimental_review(report, ranges):
    if report.status != "supported_restricted_game":
        return {"status": "unsupported", "reason": report.reason}
    hand = report.selected_hand
    return {
        "status": "experimental_assumed_range",
        "label": (f"Experimental {report.street} review · assumed ranges" +
                  (" · no later betting" if report.street in ("flop", "turn") else "")),
        "range_note": "Hero's observed hand versus three equally weighted illustrative opponent combinations. The opponent range is not inferred from betting history or calibrated to this opponent.",
        "solver_scope_note": report.reason,
        "opponent_range": ranges[1].hands,
        "actions": [
            {"kind": row.action.kind, "amount": row.action.amount,
             "frequency": row.frequency, "ev_chips": row.ev_chips,
             "ev_bb": row.ev_bb, "ev_loss_chips": row.ev_loss_chips,
             "ev_loss_bb": row.ev_loss_bb}
            for row in hand.actions
        ],
        "selected_action_ev_loss_chips": report.selected_action_ev_loss_chips,
        "selected_action_ev_loss_bb": report.selected_action_ev_loss_bb,
        "nash_conv_chips": report.nash_conv_chips,
        "iterations": report.iterations, "latency_ms": report.latency_ms,
        "joint_deals": report.joint_deals,
        "validation_status": report.validation_status,
        "assumptions": report.assumptions,
    }


def experiment_results():
    """Summarize recorded measurements; never synthesize chart values."""
    rows = []
    for path in sorted(EXPERIMENTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        experiment = data.get("experiment")
        if experiment in ("kuhn_synchronous_cfr", "leduc_synchronous_cfr"):
            for point in data["checkpoints"]:
                rows.append(
                    {
                        "source": path.name,
                        "method": experiment,
                        "condition": f"{point['iterations']} iterations",
                        "metric": "NashConv",
                        "value": round(point["nash_conv"], 4),
                    }
                )
        elif experiment == "leduc_nfsp_exact_nash_conv":
            for run in data.get("runs", []):
                if run["snapshots"]:
                    point = run["snapshots"][-1]
                    rows.append(
                        {
                            "source": path.name,
                            "method": "Leduc NFSP",
                            "condition": f"seed {run['seed']}, η={data.get('eta', 0.1)}, reward scale={data.get('reward_scale', 1.0)}, {point['episodes']} episodes",
                            "metric": "NashConv",
                            "value": round(point["nash_conv"], 4),
                        }
                    )
        elif experiment == "heuristic_vs_random":
            rows.append(
                {
                    "source": path.name,
                    "method": "Hold'em heuristic vs random",
                    "condition": f"{data['hands']} hands, seed {data['seed']}",
                    "metric": "chips/hand ± SE",
                    "value": f"{data['mean_chips_per_hand']:.2f} ± {data['standard_error']:.2f}",
                }
            )
        elif experiment in (
            "restricted_holdem_nfsp",
            "restricted_holdem_nfsp_fixed_evaluation",
            "restricted_holdem_nfsp_untrained_evaluation",
        ):
            for run in data.get("runs", []):
                for evaluation in run["evaluation"]:
                    if "hands_per_training_seed" in data:
                        policy_source = (
                            f", {data['hands_per_training_seed']} training hands"
                        )
                    elif "model_path" in run:
                        policy_source = f", {run['model_path']}"
                    else:
                        policy_source = ", initialized policy"
                    rows.append(
                        {
                            "source": path.name,
                            "method": (
                                "Restricted Hold'em NFSP (untrained)"
                                if experiment.endswith("untrained_evaluation")
                                else "Restricted Hold'em NFSP"
                            ),
                            "condition": f"seed {run.get('seed', run.get('training_seed', run.get('initialization_seed')))}, vs {evaluation['opponent']}, {evaluation['hands']} hands{policy_source}",
                            "metric": "chips/hand ± SE",
                            "value": f"{evaluation['mean_chips_per_hand']:+.2f} ± {evaluation['standard_error']:.2f}",
                        }
                    )
        elif experiment == "tiny_range_holdem_river_cfr":
            for point in data["checkpoints"]:
                rows.append(
                    {
                        "source": path.name,
                        "method": "Tiny-range Hold'em river CFR",
                        "condition": f"{point['iterations']} iterations, {data['joint_deals']} deals",
                        "metric": "NashConv",
                        "value": round(point["nash_conv"], 4),
                    }
                )
        elif experiment in (
            "tiny_range_river_value_model",
            "tiny_range_river_value_equity_feature",
        ):
            metrics = data["metrics"]
            rows.append(
                {
                    "source": path.name,
                    "method": "River value network"
                    + (" + equity" if experiment.endswith("equity_feature") else ""),
                    "condition": f"{data['test_positions']} held-out boards",
                    "metric": "MAE chips",
                    "value": round(metrics["held_out_mae_chips"], 4),
                }
            )
    return rows


class TableSession:
    def __init__(self):
        self.lock = RLock()
        self.hand_number = 0
        self.hands = []
        self.opponent = "heuristic"
        self.game = None
        self.agent = None
        self.adviser = None
        self.action_log = []
        self.decision_reviews = []
        self.solver_reviews = []
        self._review_requests = {}

    def new_hand(self, opponent="heuristic"):
        with self.lock:
            return self._new_hand(opponent)

    def _new_hand(self, opponent):
        if opponent not in ("random", "heuristic", "nfsp"):
            raise ValueError("Opponent must be random, heuristic, or nfsp")
        active_model = offline_training.status()["active"] if opponent == "nfsp" else None
        model_pair = (Path(active_model["model_dir"]) / "policy_player_0.pt",
                      Path(active_model["model_dir"]) / "policy_player_1.pt") if active_model else (NFSP_HERO_MODEL, NFSP_MODEL)
        if opponent == "nfsp" and not all(path.is_file() for path in model_pair):
            raise ValueError("Trained NFSP model is not available yet")
        self.opponent = opponent
        retained_hands = {hand["hand"] for hand in self.hands[-10:]}
        self._review_requests = {
            key: request for key, request in self._review_requests.items()
            if key[0] in retained_hands
        }
        self.action_log = []
        self.decision_reviews = []
        self.solver_reviews = []
        stack = 20 if opponent == "nfsp" else 100
        self.game = HoldemGame(
            stacks=(stack, stack),
            button=self.hand_number % 2,
            seed=secrets.randbits(64),
        )
        self.hand_number += 1
        if opponent == "random":
            self.agent = RandomAgent(secrets.randbits(64))
            self.adviser = None
        elif opponent == "heuristic":
            self.agent = HeuristicAgent(secrets.randbits(64), samples=96)
            self.adviser = None
        else:
            import torch
            from poker_cpu.nfsp import HoldemNFSPAgent

            active = offline_training.status()["active"]
            model = Path(active["model_dir"]) / "policy_player_1.pt" if active else NFSP_MODEL
            hero_model = Path(active["model_dir"]) / "policy_player_0.pt" if active else NFSP_HERO_MODEL
            hidden = active["hidden_layers"] if active else (128, 128)
            self.agent = HoldemNFSPAgent(hidden=hidden)
            self.agent.policy.load_state_dict(
                torch.load(model, map_location="cpu", weights_only=True)
            )
            self.agent.policy.eval()
            self.agent.mode = "average"
            self.adviser = None
        if opponent == "nfsp" and hero_model.is_file():
            import torch
            from poker_cpu.nfsp import HoldemNFSPAgent

            self.adviser = HoldemNFSPAgent(hidden=hidden)
            self.adviser.policy.load_state_dict(
                torch.load(hero_model, map_location="cpu", weights_only=True)
            )
            self.adviser.policy.eval()
        self._run_opponent()
        return self.state()

    def _run_opponent(self):
        while not self.game.done and self.game.current_player == 1:
            obs = self.game.observe(1)
            action = (
                self.agent.act(obs, record=False)[1]
                if self.opponent == "nfsp"
                else self.agent.act(obs)
            )
            self._step(action, 1)
        if self.game.done and (
            not self.hands or self.hands[-1]["hand"] != self.hand_number
        ):
            self.hands.append(
                {
                    "hand": self.hand_number,
                    "opponent": self.opponent,
                    "button": self.game.button,
                    "blinds": self.game.blinds,
                    "starting_stacks": self.game.initial_stacks,
                    "history": tuple(self.game.history),
                    "events": tuple(self.game.events),
                    "actions": tuple(self.action_log),
                    "decision_reviews": tuple(self.decision_reviews),
                    "solver_reviews": self.solver_reviews,
                    "board": tuple(self.game.board),
                    "hole": tuple(self.game.hole[0]),
                    "opponent_hole": (
                        tuple(self.game.hole[1]) if self.game.showdown else None
                    ),
                    "rewards": self.game.rewards(),
                }
            )

    def _analysis(self, obs):
        if self.adviser is None:
            return None
        import torch
        from poker_cpu.nfsp import ACTION_COUNT, encode, indexed_actions

        started = time.perf_counter()
        indexed = indexed_actions(obs["legal_actions"])
        mask = torch.tensor([i in indexed for i in range(ACTION_COUNT)])
        with torch.no_grad():
            logits = self.adviser.policy(encode(obs)).masked_fill(~mask, -1e9)
            probabilities = torch.softmax(logits, dim=0)
        by_action = {
            action: probabilities[index].item() for index, action in indexed.items()
        }
        return {
            "source": "NFSP average policy (experimental)",
            "action_probabilities": [
                by_action[action] for action in obs["legal_actions"]
            ],
            "inference_ms": (time.perf_counter() - started) * 1000,
            "action_ev": None,
            "search_runtime_ms": None,
        }

    def _step(self, action, player):
        game = self.game
        obs = game.observe(player)
        if action.kind in ("bet", "raise"):
            paid = action.amount - game.street_paid[player]
            target = action.amount
        elif action.kind == "call":
            paid = min(obs["to_call"], game.stacks[player])
            target = None
        else:
            paid = 0
            target = None
        game.step(action)
        self.action_log.append(
            {
                "street": obs["street"],
                "player": player,
                "kind": action.kind,
                "amount_paid": paid,
                "target": target,
                "pot_after": game.pot,
            }
        )

    def act(self, index=None, action=None):
        with self.lock:
            return self._act(index, action)

    def _act(self, index=None, custom_action=None):
        if self.game is None or self.game.done or self.game.current_player != 0:
            raise ValueError("No human action is pending")
        legal = self.game.legal_actions()
        if custom_action is None:
            if type(index) is not int or not 0 <= index < len(legal):
                raise ValueError("Choose a listed legal action")
            action = legal[index]
        else:
            if not isinstance(custom_action, dict):
                raise ValueError("Custom action must include kind and amount")
            action = Action(custom_action.get("kind"), custom_action.get("amount", 0))
            if not self.game.is_legal_action(action):
                raise ValueError("Custom action is not legal in this betting state")
            index = legal.index(action) if action in legal else None
        obs = self.game.observe(0)
        request = request_from_game(self.game)
        review = {
            "hand": self.hand_number,
            "index": len(self.solver_reviews),
            "street": request.street,
            "board": request.board,
            "hero_hand": request.acting_hand,
            "pot_before_street": request.pot_before_street,
            "stacks_at_street_start": request.stacks_at_street_start,
            "street_actions": tuple({"kind": previous.kind, "amount": previous.amount}
                                    for previous in request.betting.street_actions),
            "chosen_action": {"kind": action.kind, "amount": action.amount},
            "analysis": None,
        }
        self.solver_reviews.append(review)
        self._review_requests[(self.hand_number, review["index"])] = request
        log_event("decision_captured", hand=self.hand_number, index=review["index"],
                  street=request.street, board=request.board,
                  pot_before_street=request.pot_before_street,
                  stacks_at_street_start=request.stacks_at_street_start,
                  street_actions=review["street_actions"],
                  chosen_action=review["chosen_action"], range_source=request.range_source)
        analysis = self._analysis(obs)
        if analysis is not None:
            self.decision_reviews.append(
                {
                    "street": obs["street"],
                    "pot": obs["pot"],
                    "to_call": obs["to_call"],
                    "street_bet": obs["street_paid"][0],
                    "board": tuple(obs["board"]),
                    "hole": tuple(obs["hole"]),
                    "legal_actions": tuple(
                        {"kind": item.kind, "amount": item.amount} for item in legal
                    ),
                    "action_probabilities": tuple(analysis["action_probabilities"]),
                    "chosen_action": {"kind": action.kind, "amount": action.amount},
                    "chosen_probability": (
                        analysis["action_probabilities"][index]
                        if index is not None
                        else None
                    ),
                    "chosen_in_policy_abstraction": index is not None,
                    "inference_ms": analysis["inference_ms"],
                }
            )
        self._step(action, 0)
        self._run_opponent()
        return self.state()

    def review_decision(self, hand_number, index):
        """Opt-in post-action solve; release the session lock during CPU search."""
        with self.lock:
            if type(hand_number) is not int or type(index) is not int:
                raise ValueError("Hand and decision index must be integers")
            request = self._review_requests.get((hand_number, index))
            if request is None:
                raise ValueError("Decision snapshot is unavailable")
            review = next((item for hand in self.hands if hand["hand"] == hand_number
                           for item in hand["solver_reviews"] if item["index"] == index), None)
            if review is None and hand_number == self.hand_number:
                review = next((item for item in self.solver_reviews
                               if item["index"] == index), None)
            if review is None:
                raise ValueError("Decision snapshot is unavailable")
            if review["analysis"] is not None:
                return review["analysis"]
        log_event("review_requested", hand=hand_number, index=index,
                  street=request.street, board=request.board,
                  pot_before_street=request.pot_before_street,
                  stacks_at_street_start=request.stacks_at_street_start,
                  street_actions=[action.__dict__ for action in request.betting.street_actions],
                  chosen_action=review["chosen_action"])
        if request.street == "preflop":
            result = {"status": "unsupported", "reason":
                      "Preflop betting and continuation are not represented by the current postflop CFR tree."}
        elif request.stacks_at_street_start[0] != request.stacks_at_street_start[1]:
            result = {"status": "unsupported", "reason":
                      "Asymmetric street-start stacks are outside this equal-effective-stack solver model."}
        else:
            ranges = experimental_review_ranges(request)
            cutoff = request.street in ("flop", "turn")
            configured = replace(request, ranges=ranges,
                                 range_source="experimental_illustrative_prior",
                                 use_context="experimental_live",
                                 config=SolverConfig(
                                     iterations=500 if cutoff else 1000,
                                     max_joint_deals=3,
                                     continuation_mode="showdown_cutoff" if cutoff else "full_tree"),
                                 selected_action=Action(**review["chosen_action"]))
            log_event("review_solve_start", hand=hand_number, index=index,
                      street=request.street, continuation_mode=configured.config.continuation_mode,
                      iterations=configured.config.iterations,
                      ranges=[{"hands": belief.hands, "weights": belief.weights}
                              for belief in ranges])
            try:
                result = serialize_experimental_review(solve(configured), ranges)
            except Exception as exc:
                log_event("review_solve_error", hand=hand_number, index=index,
                          street=request.street, error_type=type(exc).__name__,
                          error=str(exc))
                raise
        log_event("review_solve_result", hand=hand_number, index=index,
                  street=request.street, status=result["status"],
                  reason=result.get("reason", result.get("solver_scope_note")),
                  iterations=result.get("iterations"),
                  latency_ms=result.get("latency_ms"),
                  nash_conv_chips=result.get("nash_conv_chips"),
                  actions=result.get("actions"))
        with self.lock:
            review["analysis"] = result
        return result

    def state(self):
        with self.lock:
            return self._state()

    def _state(self):
        if self.game is None:
            return {"started": False, "hands": self.hands[-10:]}
        game = self.game
        obs = game.observe(0)
        analysis = (
            self._analysis(obs) if not game.done and game.current_player == 0 else None
        )
        study = None
        if not game.done and game.current_player == 0:
            request = request_from_game(game)
            study = {
                "street": request.street,
                "board": request.board,
                "hero_hand": request.acting_hand,
                "pot_before_street": request.pot_before_street,
                "stacks_at_street_start": request.stacks_at_street_start,
                "first_actor": request.betting.first_actor,
                "street_actions": tuple(
                    {"kind": action.kind, "amount": action.amount}
                    for action in request.betting.street_actions
                ),
                "range_status": "manual ranges required; no validated inference",
            }
        return {
            "started": True,
            "hand": self.hand_number,
            "opponent": self.opponent,
            "starting_stack": game.initial_stacks[0],
            "button": game.button,
            "blinds": game.blinds,
            "hole": obs["hole"],
            "opponent_hole": (
                game.hole[1] if game.done and game.showdown else None
            ),
            "board": obs["board"],
            "pot": game.pot,
            "street_bets": tuple(game.street_paid),
            "stacks": tuple(game.stacks),
            "street": game.street,
            "to_call": obs["to_call"],
            "betting_limits": obs["betting_limits"],
            "current_player": None if game.done else game.current_player,
            "legal_actions": (
                [{"kind": a.kind, "amount": a.amount} for a in game.legal_actions()]
                if not game.done and game.current_player == 0
                else []
            ),
            "history": tuple(game.history),
            "events": tuple(game.events),
            "actions": tuple(self.action_log),
            "decision_reviews": tuple(self.decision_reviews),
            "solver_reviews": tuple(self.solver_reviews),
            "done": game.done,
            "rewards": game.rewards() if game.done else None,
            "hands": self.hands[-10:],
            "analysis": analysis,
            "study_position": study,
            "solver_feedback": live_analysis_availability(game.street, game.history, game),
        }


class Handler(BaseHTTPRequestHandler):
    def _session(self):
        cookie = SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        token = cookie["poker_session"].value if "poker_session" in cookie else None
        with LOCK:
            if token not in SESSIONS:
                token = secrets.token_urlsafe(24)
                SESSIONS[token] = TableSession()
        return token, SESSIONS[token]

    def _json(self, payload, status=HTTPStatus.OK, token=None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if token is not None:
            self.send_header(
                "Set-Cookie",
                f"poker_session={token}; HttpOnly; SameSite=Strict; Path=/",
            )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/training/status":
            self._json(offline_training.status())
        elif self.path == "/api/results":
            self._json({"results": experiment_results()})
        elif self.path == "/api/range-presets":
            self._json({"presets": preset_catalog()})
        elif self.path == "/api/state":
            token, session = self._session()
            self._json(session.state(), token=token)
        elif self.path in ("/", "/index.html"):
            body = (STATIC / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ranges.js":
            body = (STATIC / "ranges.js").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/training.js":
            body = (STATIC / "training.js").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/demo.css":
            body = (STATIC / "demo.css").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path not in ("/api/new", "/api/action", "/api/river-analysis", "/api/solve", "/api/demo-analysis", "/api/cached-demo-action", "/api/review-decision", "/api/training/config", "/api/training/start", "/api/training/stop"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 32768:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            if self.path == "/api/training/config":
                settings = offline_training.validate_config(payload)
                offline_training.atomic_json(offline_training.STORE / "config.json", settings)
                self._json(offline_training.status())
                return
            if self.path == "/api/training/start":
                self._json(offline_training.start())
                return
            if self.path == "/api/training/stop":
                self._json(offline_training.stop())
                return
            if self.path in ("/api/river-analysis", "/api/solve"):
                self._json(analyze_river_position(payload))
                return
            if self.path == "/api/demo-analysis":
                self._json(demo_solution(analyze_demo_position))
                return
            if self.path == "/api/cached-demo-action":
                self._json(sample_demo_action())
                return
            token, session = self._session()
            if self.path == "/api/review-decision":
                self._json({"analysis": session.review_decision(payload.get("hand"), payload.get("index"))}, token=token)
                return
            if self.path == "/api/new":
                state = session.new_hand(payload.get("opponent", "heuristic"))
            else:
                state = session.act(
                    payload.get("index"), action=payload.get("action")
                )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._json(state, token=token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-training", action="store_true",
                        help="Do not start the separate scheduled CPU training worker")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if not args.no_training:
        offline_training.start()
    print(f"Poker trainer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
