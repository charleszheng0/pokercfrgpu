from pathlib import Path
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from poker_solver import Solver, cuda_available

app = FastAPI(title="Poker GTO Solver API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    iterations: int = Field(default=100_000, ge=1, le=10_000_000)
    backend: str = Field(default="cpu", pattern="^(cpu|cuda)$")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cuda_available": cuda_available()}


@app.post("/solve")
def solve(request: SolveRequest) -> dict:
    if request.backend == "cuda" and not cuda_available():
        raise HTTPException(status_code=400, detail="CUDA backend is not available on this host")
    try:
        return Solver(request.backend).solve(request.iterations)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/benchmark")
def benchmark() -> dict:
    path = Path(os.getenv("POKER_BENCHMARK_PATH", "benchmarks/results.json")).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run ./build/benchmark to create measured results")
    return json.loads(path.read_text())
