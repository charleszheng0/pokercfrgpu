"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";

type Strategy = Record<string, { pass: number; bet: number }>;
type SolveResult = {
  strategy: Strategy;
  iterations: number;
  elapsed_ms: number;
  states_visited: number;
  states_per_second: number;
  iterations_per_second: number;
  game_value_player0: number;
  nash_conv: number;
  exploitability: number;
  backend: string;
};
type Benchmark = {
  cpu?: { iterations_per_second: number };
  cuda?: { iterations_per_second: number } | null;
};

const API = process.env.NEXT_PUBLIC_SOLVER_API ?? "http://localhost:8000";
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const number = (value: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);

export default function SolverLab() {
  const [backend, setBackend] = useState("cpu");
  const [iterations, setIterations] = useState(100000);
  const [result, setResult] = useState<SolveResult | null>(null);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/benchmark`).then((response) => response.ok ? response.json() : null)
      .then(setBenchmark).catch(() => undefined);
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const response = await fetch(`${API}/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ iterations, backend }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Solve failed");
      setResult(payload);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not reach solver API");
    } finally {
      setLoading(false);
    }
  }

  const speedup = benchmark?.cpu && benchmark.cuda
    ? benchmark.cuda.iterations_per_second / benchmark.cpu.iterations_per_second : null;

  return (
    <main className="shell">
      <Link href="/" className="back">← Overview</Link>
      <header className="labHeader">
        <div><p className="eyebrow">KUHN POKER</p><h1>Solver Lab</h1></div>
        <span className="status">Native CFR</span>
      </header>

      <form className="controls" onSubmit={submit}>
        <label>Backend
          <select value={backend} onChange={(event) => setBackend(event.target.value)}>
            <option value="cpu">CPU</option><option value="cuda">CUDA</option>
          </select>
        </label>
        <label>Iterations
          <input type="number" min="1" max="10000000" value={iterations}
            onChange={(event) => setIterations(Number(event.target.value))} />
        </label>
        <button className="primary" disabled={loading}>{loading ? "Solving…" : "Solve"}</button>
      </form>
      {error && <p className="error">{error}</p>}

      {result ? <div className="results">
        <section className="panel strategy">
          <div className="panelTitle"><h2>Average strategy</h2><span>Pass / Bet</span></div>
          <div className="strategyRows">
            {Object.entries(result.strategy).map(([infoset, actions]) => (
              <div className="strategyRow" key={infoset}>
                <code>{infoset}</code>
                <div className="bar"><i style={{ width: percent(actions.pass) }} /></div>
                <strong>{percent(actions.pass)}</strong><strong>{percent(actions.bet)}</strong>
              </div>
            ))}
          </div>
        </section>
        <section className="panel metrics">
          <div className="panelTitle"><h2>Performance</h2><span>Measured</span></div>
          <dl>
            <div><dt>Backend</dt><dd>{result.backend}</dd></div>
            <div><dt>Iterations</dt><dd>{number(result.iterations)}</dd></div>
            <div><dt>Solve time</dt><dd>{result.elapsed_ms.toFixed(2)} ms</dd></div>
            <div><dt>States visited</dt><dd>{number(result.states_visited)}</dd></div>
            <div><dt>States / sec</dt><dd>{number(result.states_per_second)}</dd></div>
            <div><dt>Player 0 EV</dt><dd>{result.game_value_player0.toFixed(5)}</dd></div>
            <div><dt>NashConv</dt><dd>{result.nash_conv.toFixed(5)}</dd></div>
            {speedup !== null && <div><dt>CPU vs CUDA</dt><dd>{speedup.toFixed(2)}×</dd></div>}
          </dl>
        </section>
      </div> : <section className="empty panel">Run the solver to inspect its learned policy and real performance.</section>}
    </main>
  );
}
