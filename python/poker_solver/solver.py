"""Thin, JSON-friendly wrapper over the pybind11 C++ module."""

from typing import Any, Dict

try:
    from ._poker_solver import cuda_available as _cuda_available
    from ._poker_solver import solve as _solve
except ImportError as exc:  # Clearer than failing later inside an API request.
    _IMPORT_ERROR = exc
    _solve = None
    _cuda_available = None
else:
    _IMPORT_ERROR = None


def cuda_available() -> bool:
    return bool(_cuda_available()) if _cuda_available is not None else False


class Solver:
    def __init__(self, backend: str = "cpu") -> None:
        if backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be 'cpu' or 'cuda'")
        self.backend = backend

    def solve(self, iterations: int = 100_000) -> Dict[str, Any]:
        if not 1 <= iterations <= 10_000_000:
            raise ValueError("iterations must be between 1 and 10,000,000")
        if _solve is None:
            raise RuntimeError(
                "Native solver module is not installed. Build it with the CMake/pybind11 "
                "commands in README.md."
            ) from _IMPORT_ERROR
        return dict(_solve(iterations, self.backend))
