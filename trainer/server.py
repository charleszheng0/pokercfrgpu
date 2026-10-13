"""Local HTTP table; all state transitions use poker_cpu.engine.HoldemGame."""

import argparse
import json
import secrets
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, RLock

from poker_cpu.agents import HeuristicAgent, RandomAgent
from poker_cpu.engine import Action, HoldemGame

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

    def new_hand(self, opponent="heuristic"):
        with self.lock:
            return self._new_hand(opponent)

    def _new_hand(self, opponent):
        if opponent not in ("random", "heuristic", "nfsp"):
            raise ValueError("Opponent must be random, heuristic, or nfsp")
        if opponent == "nfsp" and (
            not NFSP_MODEL.is_file() or not NFSP_HERO_MODEL.is_file()
        ):
            raise ValueError("Trained NFSP model is not available yet")
        self.opponent = opponent
        self.action_log = []
        self.decision_reviews = []
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

            self.agent = HoldemNFSPAgent()
            self.agent.policy.load_state_dict(
                torch.load(NFSP_MODEL, map_location="cpu", weights_only=True)
            )
            self.agent.policy.eval()
            self.agent.mode = "average"
            self.adviser = None
        if opponent == "nfsp" and NFSP_HERO_MODEL.is_file():
            import torch
            from poker_cpu.nfsp import HoldemNFSPAgent

            self.adviser = HoldemNFSPAgent()
            self.adviser.policy.load_state_dict(
                torch.load(NFSP_HERO_MODEL, map_location="cpu", weights_only=True)
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
            "done": game.done,
            "rewards": game.rewards() if game.done else None,
            "hands": self.hands[-10:],
            "analysis": analysis,
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
        if self.path == "/api/results":
            self._json({"results": experiment_results()})
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
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path not in ("/api/new", "/api/action"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4096:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            token, session = self._session()
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
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Poker trainer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
