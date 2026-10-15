"""Local, low-priority NFSP training worker. No solver claims are made here.

The HTTP server only reads atomic JSON snapshots. Training lives in another
process, so a request never waits for a gradient step or model evaluation.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.holdem_nfsp import evaluate
from poker_cpu.nfsp import HoldemNFSPAgent, play_self_play_hand

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("POKER_TRAINING_STORE", ROOT / "artifacts" / "offline_training"))
DEFAULT = {
    "learning_rate": 0.001, "batch_size": 64, "hands_per_run": 1000,
    "checkpoint_every": 100, "evaluation_hands": 200,
    "interval_seconds": 3600, "hidden_layers": [128, 128],
    "epsilon": 0.1, "eta": 0.3, "target_update_interval": 200,
    "rl_capacity": 100000, "sl_capacity": 100000, "seed": 11,
    "cfr_iterations": 2000, "validation_fraction": 0.2,
}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def config():
    return read_json(STORE / "config.json", DEFAULT.copy())


def validate_config(payload):
    if set(payload) - set(DEFAULT):
        raise ValueError("Unknown training setting: " + ", ".join(sorted(set(payload) - set(DEFAULT))))
    result = {**config(), **payload}
    ints = {"batch_size": (2, 1024), "hands_per_run": (1, 1000000),
            "checkpoint_every": (1, 100000), "evaluation_hands": (2, 100000),
            "interval_seconds": (10, 86400), "target_update_interval": (1, 100000),
            "rl_capacity": (2, 1000000), "sl_capacity": (2, 1000000),
            "seed": (0, 2147483647), "cfr_iterations": (20, 5000)}
    for key, (low, high) in ints.items():
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    for key, low, high in (("learning_rate", 1e-7, 1.0), ("epsilon", 0, 1),
                           ("eta", 0, 1), ("validation_fraction", 0, 0.8)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    hidden = result["hidden_layers"]
    if not isinstance(hidden, list) or not 1 <= len(hidden) <= 4 or any(
        isinstance(x, bool) or not isinstance(x, int) or not 8 <= x <= 1024 for x in hidden
    ):
        raise ValueError("hidden_layers must contain 1–4 integer widths from 8 to 1024")
    if result["rl_capacity"] < result["batch_size"] or result["sl_capacity"] < result["batch_size"]:
        raise ValueError("Replay capacities must be at least batch_size")
    return result


def status():
    data = read_json(STORE / "status.json", {"state": "idle", "runs_completed": 0})
    pid = data.get("pid")
    if data.get("state") in ("running", "waiting") and pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            data = {**data, "state": "stopped", "error": "Training worker exited"}
    history = read_json(STORE / "history.json", [])
    run_id = data.get("run_id")
    current_run = read_json(STORE / run_id / "progress.json", None) if run_id else None
    return {"status": data, "config": config(), "history": history[-30:],
            "current_run": current_run,
            "active": read_json(STORE / "active.json", None),
            "blueprint": read_json(STORE / "blueprint_active.json", None),
            "blueprint_history": read_json(STORE / "blueprint_history.json", [])[-30:],
            "scope": "NFSP restricted 20-chip self-play and a separate fixed-position restricted river CFR+ cache"}


def start():
    current = status()["status"]
    if current.get("state") in ("running", "waiting"):
        return status()
    STORE.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "trainer.offline_training", "--loop"],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=(STORE / "worker.log").open("a"),
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    atomic_json(STORE / "status.json", {"state": "running", "pid": process.pid,
                                         "started_at": datetime.now(timezone.utc).isoformat(),
                                         "runs_completed": current.get("runs_completed", 0)})
    return status()


def stop():
    (STORE / "stop").touch()
    return status()


def _agent_settings(settings):
    return dict(lr=settings["learning_rate"], eta=settings["eta"],
                epsilon=settings["epsilon"], hidden=settings["hidden_layers"],
                rl_capacity=settings["rl_capacity"], sl_capacity=settings["sl_capacity"],
                target_update_interval=settings["target_update_interval"],
                validation_fraction=settings["validation_fraction"])


def _snapshot(agents, hand, settings, seed, reward_sum, q_losses, sl_losses, curves):
    # A checkpoint includes optimizer and replay state for a true resume. It is
    # local trusted data; never load a checkpoint provided by an HTTP client.
    return {"hand": hand, "seed": seed, "settings": settings,
            "reward_sum": reward_sum, "q_losses": q_losses,
            "sl_losses": sl_losses, "curves": curves,
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "agents": [{"q": a.q.state_dict(), "target": a.target.state_dict(),
                        "policy": a.policy.state_dict(), "q_optimizer": a.q_optimizer.state_dict(),
                        "policy_optimizer": a.policy_optimizer.state_dict(),
                        "rl": list(a.rl), "sl": a.sl.data, "sl_seen": a.sl.seen,
                        "sl_validation": a.sl_validation.data,
                        "sl_validation_seen": a.sl_validation.seen,
                        "q_updates": a.q_updates} for a in agents]}


def _resume(agents, snapshot):
    random.setstate(snapshot["python_rng"])
    torch.set_rng_state(snapshot["torch_rng"])
    for agent, saved in zip(agents, snapshot["agents"]):
        for name in ("q", "target", "policy", "q_optimizer", "policy_optimizer"):
            getattr(agent, name).load_state_dict(saved[name])
        agent.rl.extend(saved["rl"])
        agent.sl.data = saved["sl"]
        agent.sl.seen = saved["sl_seen"]
        agent.sl_validation.data = saved["sl_validation"]
        agent.sl_validation.seen = saved["sl_validation_seen"]
        agent.q_updates = saved["q_updates"]


def improve_demo_blueprint(iterations, output=STORE):
    """Re-solve one explicit-range river spot; promote by exact restricted NashConv.

    This is a cached local strategy, not a full-game preflop blueprint.
    """
    from .demo import DEMO_RIVER_REQUEST
    from .server import analyze_river_position
    from .solution_cache import put

    request = {**DEMO_RIVER_REQUEST, "iterations": iterations}
    started = time.perf_counter()
    report = analyze_river_position(request)
    incumbent = read_json(output / "blueprint_active.json", None)
    row = {"date": datetime.now(timezone.utc).isoformat(), "iterations": iterations,
           "nash_conv_chips": report["nash_conv"],
           "solve_seconds": time.perf_counter() - started,
           "scope": "fixed explicit-range restricted river demo",
           "status": "candidate_only"}
    if report["nash_conv"] <= 0.1 and (incumbent is None or
        report["nash_conv"] < incumbent["nash_conv_chips"] - 1e-6):
        row["status"] = "promoted_restricted_river_cache"
        atomic_json(output / "blueprint_active.json", row)
        put(request, {"position": {"board": report["board"], "hero_hand": report["hero_hand"]},
                      "analysis": report})
    history = read_json(output / "blueprint_history.json", [])
    history.append(row)
    atomic_json(output / "blueprint_history.json", history)
    return row


def run_once(settings=None, *, output=STORE, allow_promotion=True):
    settings = validate_config(settings or {})
    output.mkdir(parents=True, exist_ok=True)
    history = read_json(output / "history.json", [])
    run_id = f"run-{int(time.time() * 1000)}-{os.getpid()}"
    run_dir = output / run_id
    run_dir.mkdir()
    seed = settings["seed"] + len(history) * 1000003
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    agents = [HoldemNFSPAgent(**_agent_settings(settings)) for _ in range(2)]
    latest = output / "latest_checkpoint.pt"
    start_hand = 0
    curves = []
    reward_sum = 0.0
    q_losses, sl_losses = [], []
    if latest.exists():
        # Resume only identical configuration; changing hyperparameters starts a new run.
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["settings"] == settings and saved["hand"] < settings["hands_per_run"]:
            _resume(agents, saved)
            start_hand = saved["hand"]
            seed = saved["seed"]
            curves = saved.get("curves", [])
            reward_sum = saved.get("reward_sum", 0.0)
            q_losses = saved.get("q_losses", [])
            sl_losses = saved.get("sl_losses", [])
    started = time.perf_counter()
    completed = start_hand
    for hand in range(start_hand, settings["hands_per_run"]):
        if (output / "stop").exists():
            break
        reward_sum += play_self_play_hand(agents, seed + hand, button=hand % 2)[0]
        completed = hand + 1
        if (hand + 1) % 4 == 0:
            for agent in agents:
                q, sl = agent.update(settings["batch_size"])
                if q is not None: q_losses.append(q)
                if sl is not None: sl_losses.append(sl)
        if (hand + 1) % settings["checkpoint_every"] == 0 or hand + 1 == settings["hands_per_run"]:
            point = {"hands": hand + 1, "seat0_mean_chips": reward_sum / (hand + 1),
                     "q_loss": sum(q_losses) / len(q_losses) if q_losses else None,
                     "policy_loss": sum(sl_losses) / len(sl_losses) if sl_losses else None,
                     "validation_policy_loss": agents[1].validation_policy_loss(),
                     "validation_examples": len(agents[1].sl_validation.data),
                     "elapsed_seconds": time.perf_counter() - started}
            curves.append(point)
            checkpoint = run_dir / "checkpoint.pt"
            torch.save(_snapshot(agents, hand + 1, settings, seed,
                                 reward_sum, q_losses, sl_losses, curves), checkpoint)
            checkpoint.replace(latest)
            atomic_json(run_dir / "progress.json", {"run_id": run_id, "settings": settings,
                                                     "curves": curves, "state": "training"})
            atomic_json(output / "status.json", {"state": "running", "pid": os.getpid(),
                                                 "run_id": run_id, "progress": point,
                                                 "runs_completed": len(history)})
    if completed < settings["hands_per_run"]:
        return {"state": "stopped", "run_id": run_id, "hands": completed}
    for p, agent in enumerate(agents):
        torch.save(agent.policy.state_dict(), run_dir / f"policy_player_{p}.pt")
    agents[1].mode = "average"
    evaluations = [evaluate(agents[1], opponent, settings["evaluation_hands"], seed + 100000)
                   for opponent in ("random", "heuristic")]
    row = {"run_id": run_id, "date": datetime.now(timezone.utc).isoformat(),
           "seed": seed, "settings": settings, "training_hands": completed,
           "training_seconds": time.perf_counter() - started, "curves": curves,
           "evaluation": evaluations, "model_dir": str(run_dir),
           "sha256": hashlib.sha256((run_dir / "policy_player_1.pt").read_bytes()).hexdigest(),
           "promotion": "candidate_only"}
    # The fixed pre-existing NFSP model can be compared if its architecture matches.
    active = read_json(output / "active.json", None)
    baseline = (Path(active["model_dir"]) / "policy_player_1.pt") if active else ROOT / "artifacts/holdem_nfsp/seed11/policy_player_1.pt"
    if baseline.is_file():
        incumbent = HoldemNFSPAgent(hidden=active["hidden_layers"] if active else (128, 128))
        incumbent.policy.load_state_dict(torch.load(baseline, map_location="cpu", weights_only=True))
        incumbent.mode = "average"
        old_eval = evaluate(incumbent, "heuristic", settings["evaluation_hands"], seed + 100000)
        row["incumbent_heuristic"] = old_eval
        new_eval = evaluations[1]
        margin = 1.96 * math.sqrt((new_eval["standard_error"] or 0) ** 2 +
                                   (old_eval["standard_error"] or 0) ** 2)
        row["heuristic_delta_chips"] = new_eval["mean_chips_per_hand"] - old_eval["mean_chips_per_hand"]
        row["delta_95ci_independent"] = [row["heuristic_delta_chips"] - margin,
                                         row["heuristic_delta_chips"] + margin]
        if allow_promotion and settings["evaluation_hands"] >= 100 and row["delta_95ci_independent"][0] > 0:
            row["promotion"] = "promoted_vs_incumbent_heuristic"
    elif allow_promotion and settings["evaluation_hands"] >= 100:
        row["promotion"] = "initial_checkpoint_no_incumbent"
    if row["promotion"] != "candidate_only":
        atomic_json(output / "active.json", {"run_id": run_id, "model_dir": str(run_dir),
                                                 "hidden_layers": settings["hidden_layers"],
                                                 "promotion": row["promotion"]})
    history.append(row)
    if allow_promotion:
        row["blueprint_job"] = improve_demo_blueprint(settings["cfr_iterations"], output)
    atomic_json(output / "history.json", history)
    atomic_json(run_dir / "progress.json", {**row, "state": "complete"})
    latest.unlink(missing_ok=True)
    return row


def loop():
    try:
        os.nice(10)
    except OSError:
        pass
    (STORE / "stop").unlink(missing_ok=True)
    while not (STORE / "stop").exists():
        try:
            row = run_once()
            atomic_json(STORE / "status.json", {"state": "waiting", "pid": os.getpid(),
                                                 "last_run": row.get("run_id"),
                                                 "runs_completed": len(read_json(STORE / "history.json", []))})
        except Exception as exc:
            atomic_json(STORE / "status.json", {"state": "error", "pid": os.getpid(),
                                                 "error": repr(exc)})
            raise
        due = time.monotonic() + config()["interval_seconds"]
        while time.monotonic() < due and not (STORE / "stop").exists():
            time.sleep(min(2, due - time.monotonic()))
    atomic_json(STORE / "status.json", {"state": "stopped", "pid": os.getpid(),
                                         "runs_completed": len(read_json(STORE / "history.json", []))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.loop: loop()
    elif args.once: print(json.dumps(run_once(), indent=2))
    else: parser.error("Choose --loop or --once")
