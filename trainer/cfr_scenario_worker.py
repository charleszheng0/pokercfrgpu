"""Low-priority offline CFR+ solves for reproducible tiny river scenarios.

Each artifact is a solution to one explicitly specified restricted game. This
worker does not train a generalizing policy or create a full Hold'em blueprint.
Only scenarios meeting the configured exact restricted-game NashConv threshold
are written to the playable-cache directory.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from poker_cpu.cards import DECK
from poker_cpu.ranges import canonical_hand

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("POKER_CFR_SCENARIO_STORE", ROOT / "artifacts" / "cfr_scenario_worker"))
CACHE = ROOT / "artifacts" / "cfr_scenario_cache"
SOLVER_SOURCE_FILES = (
    "poker_cpu/cards.py",
    "poker_cpu/ranges.py",
    "poker_cpu/solve.py",
    "poker_cpu/river_solver.py",
    "poker_cpu/multi_street.py",
    "trainer/server.py",
    "trainer/cfr_scenario_worker.py",
)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def make_request(seed, iterations=1000, combos_per_range=4):
    """Sample a river game with exact weighted combos and no dead sampled hands."""
    if isinstance(combos_per_range, bool) or not isinstance(combos_per_range, int) or not 1 <= combos_per_range <= 8:
        raise ValueError("combos_per_range must be an integer from 1 to 8 for the shared API")
    rng = random.Random(seed)
    cards = list(DECK)
    rng.shuffle(cards)
    board = cards[:5]
    hero_hands = [canonical_hand(cards[5 + 2 * i:7 + 2 * i])
                  for i in range(combos_per_range)]
    opponent_start = 5 + 2 * combos_per_range
    opponent_hands = [canonical_hand(cards[opponent_start + 2 * i:opponent_start + 2 * i + 2])
                      for i in range(combos_per_range)]
    return {
        "street": "river",
        "board": " ".join(board),
        "hero_hand": " ".join(hero_hands[0]),
        "pot": rng.choice((8, 12, 20)),
        "stacks": [rng.choice((10, 20, 40))] * 2,
        # The public endpoint evaluates hero (seat 0) at the root. Randomize
        # boards/ranges/stacks, but keep the root actor aligned with that API.
        "first_actor": 0,
        "iterations": iterations,
        "ranges": [
            {"hands": [list(hand) for hand in hero_hands], "weights": [1.0] * combos_per_range},
            {"hands": [list(hand) for hand in opponent_hands], "weights": [1.0] * combos_per_range},
        ],
        "range_source": "offline_sampled_range",
    }


def solver_fingerprint():
    """Hash the implementation that produced cached strategies."""
    digest = hashlib.sha256()
    for relative_path in SOLVER_SOURCE_FILES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def request_key(request, fingerprint=None):
    """Hash the exact solve request and solver implementation version."""
    fingerprint = fingerprint or solver_fingerprint()
    digest = hashlib.sha256()
    digest.update(json.dumps(request, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(fingerprint.encode("ascii"))
    return digest.hexdigest()


def scenario_key(request):
    """Stable poker-game identity, independent of solver iteration budget."""
    scenario = {key: value for key, value in request.items() if key != "iterations"}
    return hashlib.sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def get_qualified_scenario(request, *, cache=None, max_nash_conv_chips=0.1):
    """Return an exact-current-code cache hit, otherwise ``None``.

    Cache entries without a source fingerprint are deliberately ignored; this
    prevents old strategies from silently surviving solver-code changes.
    """
    if (type(max_nash_conv_chips) not in (int, float)
            or not math.isfinite(max_nash_conv_chips)
            or max_nash_conv_chips < 0):
        raise ValueError("Cache NashConv ceiling must be finite and nonnegative")
    cache = CACHE if cache is None else cache
    fingerprint = solver_fingerprint()
    solve_id = request_key(request, fingerprint)
    try:
        row = json.loads((Path(cache) / f"{solve_id}.json").read_text())
    except (FileNotFoundError, ValueError):
        return None
    nash_conv = row.get("nash_conv_chips")
    if (row.get("status") != "cached_qualified"
            or row.get("solve_id") != solve_id
            or row.get("solver_fingerprint") != fingerprint
            or row.get("request") != request
            or type(nash_conv) not in (int, float)
            or not math.isfinite(nash_conv)
            or not 0 <= nash_conv <= max_nash_conv_chips):
        return None
    if not isinstance(row.get("result"), dict):
        return None
    return row


def solve_one(seed, *, iterations=2000, combos_per_range=4, threshold_chips=0.1,
              store=STORE, cache=CACHE):
    from .server import analyze_river_position

    request = make_request(seed, iterations, combos_per_range)
    started = time.perf_counter()
    report = analyze_river_position(request)
    elapsed = time.perf_counter() - started
    scenario_id = scenario_key(request)
    fingerprint = solver_fingerprint()
    solve_id = request_key(request, fingerprint)
    qualified = report["nash_conv"] <= threshold_chips
    row = {
        "scenario_id": scenario_id,
        "solve_id": solve_id,
        "solver_fingerprint": fingerprint,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "cached_qualified" if qualified else "candidate_only",
        "threshold_nash_conv_chips": threshold_chips,
        "nash_conv_chips": report["nash_conv"],
        "iterations": report["iterations"],
        "latency_seconds": elapsed,
        "joint_deals": report["joint_deals"],
        "scope": f"river; {2 * combos_per_range} supplied private-hand combinations; restricted bets; exact restricted-game NashConv",
        "request": request,
        "result": report,
    }
    store.mkdir(parents=True, exist_ok=True)
    atomic_json(store / f"{solve_id}.json", row)
    if qualified:
        cache.mkdir(parents=True, exist_ok=True)
        atomic_json(cache / f"{solve_id}.json", row)
    return row


def run_loop(*, duration_seconds=3600, pause_seconds=20, max_solves=1000,
             iterations=2000, combos_per_range=4, threshold_chips=0.1, seed=20261020,
             store=STORE, cache=CACHE):
    try:
        os.nice(15)
    except (AttributeError, OSError):
        pass
    store.mkdir(parents=True, exist_ok=True)
    stop_file = store / "stop"
    stop_file.unlink(missing_ok=True)
    started = time.monotonic()
    completed = len(read_json(store / "history.json", []))
    rows = read_json(store / "history.json", [])
    atomic_json(store / "status.json", {
        "state": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": completed, "duration_limit_seconds": duration_seconds,
        "iterations_per_solve": iterations, "combos_per_player_range": combos_per_range,
        "threshold_nash_conv_chips": threshold_chips,
        "cache_scope": "restricted river scenarios only",
    })
    while (time.monotonic() - started < duration_seconds and completed < max_solves
           and not stop_file.exists()):
        try:
            row = solve_one(seed + completed, iterations=iterations,
                            combos_per_range=combos_per_range,
                            threshold_chips=threshold_chips, store=store / "candidates", cache=cache)
        except Exception as exc:
            atomic_json(store / "status.json", {
                "state": "error", "pid": os.getpid(), "completed": completed,
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(exc), "failed_seed": seed + completed,
                "cache_scope": "restricted river scenarios only",
            })
            raise
        rows.append({key: value for key, value in row.items() if key not in ("request", "result")})
        completed += 1
        atomic_json(store / "history.json", rows[-1000:])
        atomic_json(store / "status.json", {
            "state": "running", "pid": os.getpid(), "completed": completed,
            "qualified": sum(item["status"] == "cached_qualified" for item in rows),
            "last_scenario": rows[-1], "elapsed_seconds": time.monotonic() - started,
            "duration_limit_seconds": duration_seconds,
            "iterations_per_solve": iterations, "combos_per_player_range": combos_per_range,
            "threshold_nash_conv_chips": threshold_chips,
            "cache_scope": "restricted river scenarios only",
        })
        end = min(started + duration_seconds, time.monotonic() + pause_seconds)
        while time.monotonic() < end and not stop_file.exists():
            time.sleep(min(0.5, end - time.monotonic()))
    atomic_json(store / "status.json", {
        "state": "stopped", "pid": os.getpid(), "completed": completed,
        "qualified": sum(item["status"] == "cached_qualified" for item in rows),
        "elapsed_seconds": time.monotonic() - started,
        "cache_scope": "restricted river scenarios only",
    })
    return read_json(store / "status.json", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="solve one random river scenario")
    parser.add_argument("--loop", action="store_true", help="run bounded low-priority offline solves")
    parser.add_argument("--stop", action="store_true", help="request a running worker to stop")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--pause-seconds", type=float, default=20.0)
    parser.add_argument("--max-solves", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--combos-per-range", type=int, default=4)
    parser.add_argument("--threshold-chips", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.stop:
        STORE.mkdir(parents=True, exist_ok=True)
        (STORE / "stop").touch()
        print(json.dumps({"state": "stop_requested", "store": str(STORE)}))
    elif args.once:
        print(json.dumps(solve_one(args.seed, iterations=args.iterations,
                                   combos_per_range=args.combos_per_range,
                                   threshold_chips=args.threshold_chips), indent=2))
    elif args.loop:
        print(json.dumps(run_loop(duration_seconds=args.duration_seconds,
                                  pause_seconds=args.pause_seconds,
                                  max_solves=args.max_solves,
                                  iterations=args.iterations,
                                  combos_per_range=args.combos_per_range,
                                  threshold_chips=args.threshold_chips,
                                  seed=args.seed), indent=2))
    else:
        parser.error("Choose --once, --loop, or --stop")


if __name__ == "__main__":
    main()
