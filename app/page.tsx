import Link from "next/link";

export default function Home() {
  return (
    <main className="shell hero">
      <p className="eyebrow">GPU-ORIENTED IMPERFECT-INFORMATION SOLVING</p>
      <h1>Poker GTO Solver</h1>
      <p className="lede">
        A measured, testable CFR prototype for Kuhn Poker, connected from React to
        FastAPI and a native C++ solver.
      </p>
      <Link className="primary button" href="/solver">Open Solver Lab</Link>
    </main>
  );
}
