"""Exact-position cache of verified restricted river solutions.

Cache keys include all supplied ranges and solver settings. A cached strategy
must never be treated as a policy for an unrelated betting history or board.
"""
import hashlib
import json
import math
import random
from pathlib import Path

from .demo import DEMO_RIVER_REQUEST
from .offline_training import ROOT, atomic_json

CACHE = ROOT / "artifacts" / "solution_cache"
SOLVER_SOURCES = ("poker_cpu/solve.py", "poker_cpu/river_solver.py",
                  "poker_cpu/multi_street.py", "trainer/server.py")


def key(request):
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode())
    for source in SOLVER_SOURCES:
        digest.update((ROOT / source).read_bytes())
    return digest.hexdigest()


def get(request):
    path = CACHE / f"{key(request)}.json"
    try:
        entry = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    if entry.get("key") != key(request):
        return None
    return entry.get("result")


def put(request, result):
    conv = result["analysis"]["nash_conv"]
    if not math.isfinite(conv) or conv < 0 or conv > 0.1:
        return False
    atomic_json(CACHE / f"{key(request)}.json", {"key": key(request),
                                                     "scope": "restricted river demo; explicit ranges",
                                                     "result": result})
    return True


def demo_solution(solve_function):
    saved = get(DEMO_RIVER_REQUEST)
    if saved is not None:
        return {**saved, "cache_status": "cached_verified_demo"}
    result = solve_function()
    put(DEMO_RIVER_REQUEST, result)
    return {**result, "cache_status": "fresh_solve"}


def sample_demo_action():
    result = get(DEMO_RIVER_REQUEST)
    if result is None:
        raise ValueError("No verified cached demo solution yet; run the solver demo first")
    rows = result["analysis"]["frequencies"]
    if not rows or any(not math.isfinite(row["frequency"]) or row["frequency"] < 0 for row in rows) or sum(row["frequency"] for row in rows) <= 0:
        raise ValueError("Cached strategy is invalid")
    action = random.choices(rows, weights=[row["frequency"] for row in rows], k=1)[0]["action"]
    return {"action": action, "source": "cached restricted CFR+ river demo",
            "cache_key": key(DEMO_RIVER_REQUEST)}
