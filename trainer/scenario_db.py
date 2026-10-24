"""SQLite record of exact, versioned Scenario Lab solves.

Every result is retained for audit. Only a result with a measured exact
restricted-game NashConv below the configured ceiling is served as a cache hit.
The key includes the complete request and the implementation fingerprint; it
does not generalize a strategy to other boards, ranges, or action histories.
"""

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .cfr_scenario_worker import request_key, solver_fingerprint


DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / "artifacts" / "scenario_solves.sqlite3"


def database_path():
    return Path(os.environ.get("POKER_SCENARIO_DB", DEFAULT_DATABASE))


def _connect(path=None):
    path = database_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS solves (
            solve_id TEXT PRIMARY KEY,
            solver_fingerprint TEXT NOT NULL,
            request_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            street TEXT NOT NULL,
            nash_conv_chips REAL NOT NULL,
            qualified INTEGER NOT NULL CHECK (qualified IN (0, 1)),
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS solves_street_quality ON solves(street, qualified)")
    return connection


def record(request, result, *, quality_ceiling_chips=0.1, path=None):
    """Persist a measured solve; return whether it qualifies for reuse."""
    conv = result.get("nash_conv")
    if (type(conv) not in (int, float) or not math.isfinite(conv) or conv < 0
            or type(quality_ceiling_chips) not in (int, float)
            or not math.isfinite(quality_ceiling_chips) or quality_ceiling_chips < 0):
        raise ValueError("A finite nonnegative NashConv and quality ceiling are required")
    fingerprint = solver_fingerprint()
    solve_id = request_key(request, fingerprint)
    qualified = conv <= quality_ceiling_chips
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result_json = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with _connect(path) as connection:
        connection.execute("""
            INSERT OR REPLACE INTO solves
                (solve_id, solver_fingerprint, request_json, result_json, street,
                 nash_conv_chips, qualified, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (solve_id, fingerprint, request_json, result_json, request.get("street", "river"),
              conv, int(qualified), datetime.now(timezone.utc).isoformat()))
    return qualified


def lookup(request, *, quality_ceiling_chips=0.1, path=None):
    """Return only an exact, currently qualified result, otherwise None."""
    if (type(quality_ceiling_chips) not in (int, float)
            or not math.isfinite(quality_ceiling_chips) or quality_ceiling_chips < 0):
        raise ValueError("Quality ceiling must be finite and nonnegative")
    fingerprint = solver_fingerprint()
    solve_id = request_key(request, fingerprint)
    with _connect(path) as connection:
        row = connection.execute("""
            SELECT request_json, result_json, nash_conv_chips
            FROM solves WHERE solve_id = ? AND solver_fingerprint = ? AND qualified = 1
        """, (solve_id, fingerprint)).fetchone()
    if row is None:
        return None
    saved_request, saved_result, conv = row
    expected_request = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (saved_request != expected_request or not math.isfinite(conv)
            or conv > quality_ceiling_chips):
        return None
    result = json.loads(saved_result)
    if (not isinstance(result, dict) or result.get("nash_conv") != conv
            or result.get("status") != f"restricted_{request.get('street', 'river')}_analysis"):
        return None
    # Preserve the in-process API's tuple fields. HTTP JSON still serializes
    # these as arrays, but callers should not change types on a cache hit.
    for field in ("board", "hero_hand", "river_paid", "street_paid", "solver_assumptions"):
        if field in result:
            result[field] = tuple(result[field])
    for row in result.get("per_hand", ()):
        row["hand"] = tuple(row["hand"])
    return result


def stats(path=None):
    """Small operational summary; counts are exact requests, not gameplay coverage."""
    with _connect(path) as connection:
        rows = connection.execute("""
            SELECT street, COUNT(*), SUM(qualified) FROM solves GROUP BY street
        """).fetchall()
    return {street: {"recorded": count, "qualified": qualified or 0}
            for street, count, qualified in rows}
