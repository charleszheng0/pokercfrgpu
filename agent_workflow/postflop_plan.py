"""Bounded implementation queue for the postflop scenario solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Milestone:
    id: str
    title: str
    objective: str
    acceptance: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# This intentionally ends at a postflop scenario lab. Preflop and the playable
# game are separate future goals and must not be pulled into these cycles.
POSTFLOP_MILESTONES = (
    Milestone(
        id="reference-and-contract",
        title="Postflop tree contract and reference fixtures",
        objective=(
            "Audit the existing postflop request and tree. Define the compact per-street "
            "bet/raise menus and a reproducible fixture set. Compare against an independent "
            "compatible reference when available; do not claim equivalence without matching runs."
        ),
        acceptance=(
            "Fixtures explicitly record board, ranges, pot, effective stacks, positions, "
            "betting history, sizing tree, solver variant, and budget.",
            "Tests cover per-street/per-action sizes, minimum raises, all-in deduplication, "
            "card removal, and chip-EV payoff semantics.",
            "Any comparison records real outputs and does not label an unrun comparison as passed.",
        ),
    ),
    Milestone(
        id="chance-and-street-coverage",
        title="Chance-aware flop, turn, and river coverage",
        objective=(
            "Extend the verified restricted game through actual postflop betting and "
            "public-card transitions in measured steps. Scale compatible hand deals, "
            "chance work, and action branches one factor at a time. Keep explicit resource "
            "limits until a measured change justifies raising them."
        ),
        acceptance=(
            "Small exact reference cases cover flop-to-turn-to-river and turn-to-river betting, "
            "including all-in terminals.",
            "Range scaling records compatible deal count, board-removal effects, latency, "
            "memory where available, action EVs, and restricted NashConv.",
            "Unsupported resource budgets fail with an explicit limit instead of returning partial fake advice.",
        ),
    ),
    Milestone(
        id="chance-and-cfr-validation",
        title="CFR convergence and independent validation",
        objective=(
            "Validate the selected CFR/DCFR update and chance handling on tractable "
            "postflop fixtures. Compare conditional action EVs, average strategies, "
            "and restricted-game best responses against independent reference runs."
        ),
        acceptance=(
            "Deterministic tests verify reach weighting, card removal, terminal values, and action EVs.",
            "Convergence is reported as a measured curve with seed, abstraction, and runtime.",
            "Reference differences have documented tolerances and actual before/after values.",
        ),
    ),
    Milestone(
        id="study-spot-selection",
        title="Evidence-based study spot selection",
        objective=(
            "Add useful-scenario filters only from measured solve outputs. Record range reach, "
            "board texture, action mixing, EV gaps, line and stack-to-pot ratio; inspect the "
            "sample distribution before selecting or labeling spots as close/interesting."
        ),
        acceptance=(
            "Candidate metadata is generated from the actual solved request/result and preserves provenance.",
            "Any selection rule is a documented project heuristic validated on a recorded candidate set.",
            "No random scenario is called strategically interesting without supporting metrics.",
        ),
    ),
    Milestone(
        id="scenario-lab-ui",
        title="Minimal postflop Scenario Lab",
        objective=(
            "Make the Scenario Lab the primary experience: compact postflop spot inputs, "
            "range editing, a clear action/EV results table, and concise solve diagnostics. "
            "Keep the gameplay table out of this solver screen and preserve gameplay as a "
            "separate mode for later integration."
        ),
        acceptance=(
            "Users can set a flop, turn, or river scenario and ranges without raw JSON.",
            "The result table displays only actual outputs returned by the shared solver API.",
            "Browser checks verify supported and unsupported states and responsive layout.",
        ),
    ),
    Milestone(
        id="postflop-release-gate",
        title="Postflop Scenario Lab release gate",
        objective=(
            "Run the complete supported-scenario test suite and publish a concise capability "
            "report. Fix blockers that are local and reproducible; otherwise stop with the "
            "smallest next task and evidence."
        ),
        acceptance=(
            "All required tests pass on the final isolated snapshot.",
            "Supported streets, ranges, bet sizes, known limitations, convergence, and latency are documented.",
            "No preflop or full-game claim is made by this postflop milestone.",
        ),
    ),
)


def next_milestone(completed_ids: set[str]) -> Milestone | None:
    """Return the first unfinished milestone in dependency order."""
    return next((item for item in POSTFLOP_MILESTONES if item.id not in completed_ids), None)


def cycle_prompt(milestone: Milestone, source_root: str) -> str:
    checks = "\n".join(f"- {item}" for item in milestone.acceptance)
    return (
        "Project goal: finish a defensible heads-up postflop Hold'em Scenario Lab. "
        "Preflop and the playable game are out of scope for this work queue.\n\n"
        f"Current milestone: {milestone.id} — {milestone.title}\n"
        f"Objective: {milestone.objective}\n\n"
        f"Acceptance criteria:\n{checks}\n\n"
        f"Source checkout: {source_root}. Work only in the supplied isolated snapshot. "
        "Use the editor for one bounded implementation, then have the read-only reviewer "
        "inspect the actual diff. The CLI will independently run the relevant local checks. "
        "Stop if the scope cannot be completed safely in this cycle. Never invent test, "
        "benchmark, reference-solver, or convergence results. End with exactly these two "
        "machine-readable lines, based on the reviewer result and actual checks:\n"
        f"POSTFLOP_MILESTONE: {milestone.id} COMPLETE|BLOCKED\n"
        "POSTFLOP_REVIEW: APPROVE|CHANGES_REQUIRED"
    )


def output_marks_milestone_complete(output: str, milestone_id: str) -> bool:
    """Require explicit manager/reviewer markers before advancing the queue."""
    lines = {line.strip() for line in output.splitlines()}
    return (
        f"POSTFLOP_MILESTONE: {milestone_id} COMPLETE" in lines
        and "POSTFLOP_REVIEW: APPROVE" in lines
    )
