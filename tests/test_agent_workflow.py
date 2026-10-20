import pytest

from agent_workflow.cli import parse_args
from agent_workflow.postflop_plan import (
    POSTFLOP_MILESTONES,
    cycle_prompt,
    next_milestone,
    output_marks_milestone_complete,
)
from agent_workflow.workspace import IsolatedWorkspace


def test_postflop_queue_is_ordered_and_excludes_preflop_and_gameplay():
    assert len(POSTFLOP_MILESTONES) == 6
    assert next_milestone(set()).id == "reference-and-contract"
    completed = {milestone.id for milestone in POSTFLOP_MILESTONES}
    assert next_milestone(completed) is None
    assert all("preflop" not in milestone.id for milestone in POSTFLOP_MILESTONES)
    assert "gameplay" not in " ".join(milestone.id for milestone in POSTFLOP_MILESTONES)


def test_cycle_prompt_contains_scope_acceptance_and_explicit_gate_markers():
    milestone = POSTFLOP_MILESTONES[0]
    prompt = cycle_prompt(milestone, "/repo")
    assert "heads-up postflop Hold'em Scenario Lab" in prompt
    assert "Preflop and the playable game are out of scope" in prompt
    assert "Fixtures explicitly record" in prompt
    assert f"POSTFLOP_MILESTONE: {milestone.id} COMPLETE|BLOCKED" in prompt


def test_advancement_requires_both_milestone_and_reviewer_approval():
    milestone_id = POSTFLOP_MILESTONES[0].id
    assert output_marks_milestone_complete(
        f"POSTFLOP_MILESTONE: {milestone_id} COMPLETE\nPOSTFLOP_REVIEW: APPROVE",
        milestone_id,
    )
    assert not output_marks_milestone_complete(
        f"POSTFLOP_MILESTONE: {milestone_id} COMPLETE\nPOSTFLOP_REVIEW: CHANGES_REQUIRED",
        milestone_id,
    )
    assert not output_marks_milestone_complete(
        f"POSTFLOP_MILESTONE: {milestone_id} BLOCKED\nPOSTFLOP_REVIEW: APPROVE",
        milestone_id,
    )


def test_cli_exposes_bounded_postflop_run_and_dry_run():
    args = parse_args(["--postflop-manager", "--dry-run", "--max-cycles", "3", "--hours", ".5"])
    assert args.postflop_manager is True
    assert args.task is None
    assert args.dry_run is True
    assert args.max_cycles == 3
    assert args.hours == .5


def test_manager_time_budget_can_skip_a_check_without_spawning_process(monkeypatch, tmp_path):
    # Construct only enough state to exercise the budget guard; no repo snapshot
    # or external test process is needed.
    workspace = object.__new__(IsolatedWorkspace)
    monkeypatch.setattr(
        "agent_workflow.workspace.subprocess.run",
        lambda *args, **kwargs: pytest.fail("check should not start after deadline"),
    )
    assert "budget is exhausted" in workspace.run_check("python_tests", 0)
