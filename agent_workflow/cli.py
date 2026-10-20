"""CLI entry point for the isolated manager/editor/reviewer workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .workspace import IsolatedWorkspace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manager/editor/reviewer workflow against an isolated project snapshot."
    )
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="One bounded task for the manager to coordinate.")
    task_group.add_argument(
        "--postflop-manager",
        action="store_true",
        help="Run the ordered postflop Scenario Lab manager/editor/reviewer queue.",
    )
    parser.add_argument("--max-cycles", type=int, default=5,
                        help="Maximum approved milestones in one manager run (default: 5).")
    parser.add_argument("--hours", type=float, default=1.0,
                        help="Wall-clock budget for the postflop manager run (default: 1 hour).")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Git repository to snapshot (defaults to the current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the isolated snapshot and print its inventory without calling the model.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the temporary workspace for inspection after the run.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="Optional model override.")
    return parser.parse_args(argv)


async def run_task(args: argparse.Namespace) -> int:
    if args.max_cycles < 1 or args.max_cycles > 20:
        print("--max-cycles must be between 1 and 20.", file=sys.stderr)
        return 2
    if args.hours <= 0 or args.hours > 12:
        print("--hours must be greater than 0 and at most 12.", file=sys.stderr)
        return 2
    try:
        workspace = IsolatedWorkspace(args.repo, keep=args.keep_workspace)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Could not prepare isolated workspace: {exc}", file=sys.stderr)
        return 2

    try:
        print(f"Source repository: {workspace.source_root}")
        print(f"Agent workspace:   {workspace.workspace_root}")
        print("Source checkout will not be edited by agents.")
        if args.dry_run:
            print("\nDry run: snapshot created successfully.")
            if args.postflop_manager:
                from .postflop_plan import POSTFLOP_MILESTONES
                print("\nPostflop manager queue:")
                for index, item in enumerate(POSTFLOP_MILESTONES, 1):
                    print(f"{index}. {item.id}: {item.title}")
                print(f"\nRun budget: at most {args.max_cycles} milestones and {args.hours:g} hour(s).")
            print("Top-level files:\n" + workspace.list_files("."))
            return 0

        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Set it in this terminal, then rerun.", file=sys.stderr)
            return 2

        try:
            from agents import Runner
            from .agents import make_agents
        except ImportError:
            print(
                "Agents SDK is not installed. Run: python3 -m pip install -r requirements-agents.txt",
                file=sys.stderr,
            )
            return 2

        manager = make_agents(workspace, model=args.model)
        run_error = None
        try:
            if args.postflop_manager:
                run_error = await run_postflop_manager(
                    args, manager, Runner, workspace
                )
            else:
                result = await Runner.run(
                    manager,
                    (
                        f"Repository task:\n{args.task}\n\n"
                        f"Use the supplied isolated snapshot at {workspace.workspace_root}. "
                        "Keep the change narrow. Before accepting it, obtain an independent reviewer "
                        "verdict and run relevant checks."
                    ),
                    max_turns=18,
                )
                print("\nMANAGER REPORT\n")
                print(result.final_output)
        except Exception as exc:
            run_error = exc
            print(f"\nAgent run stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not args.postflop_manager:
            patch_path = workspace.save_patch()
            if patch_path:
                print(f"\nPatch saved for inspection: {patch_path}")
                print("Review it, then apply from the repository root with: git apply <patch-path>")
            else:
                print("\nNo source changes were produced.")
        if args.keep_workspace:
            print(f"\nWorkspace retained at: {workspace.root}")
        return 1 if run_error else 0
    finally:
        workspace.close()


async def run_postflop_manager(args, manager, runner, workspace) -> Exception | None:
    """Run ordered bounded cycles; stop at a failed review/check or time budget."""
    from .postflop_plan import (
        POSTFLOP_MILESTONES,
        cycle_prompt,
        next_milestone,
        output_marks_milestone_complete,
    )

    started = time.monotonic()
    deadline = started + args.hours * 3600
    completed: set[str] = set()
    cycles: list[dict[str, object]] = []
    stop_reason = "milestone queue complete"
    run_error = None
    while len(cycles) < args.max_cycles and time.monotonic() < deadline:
        milestone = next_milestone(completed)
        if milestone is None:
            break
        remaining = max(1.0, deadline - time.monotonic())
        print(f"\nPOSTFLOP CYCLE {len(cycles) + 1}: {milestone.id} — {milestone.title}", flush=True)
        try:
            result = await asyncio.wait_for(
                runner.run(
                    manager,
                    cycle_prompt(milestone, str(workspace.source_root)),
                    max_turns=18,
                ),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            stop_reason = "hour budget reached during agent cycle"
            break
        except Exception as exc:
            run_error = exc
            stop_reason = f"agent cycle errored: {type(exc).__name__}: {exc}"
            break

        output = result.final_output
        print("\nMANAGER REPORT\n" + output, flush=True)
        # The workflow runs these checks itself; acceptance does not rely only on
        # the editor's or manager's description of a test result.
        checks: dict[str, str] = {}
        for check_id in ("python_tests", "browser_demo"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                checks[check_id] = (
                    f"{check_id}: skipped because the manager time budget is exhausted."
                )
                continue
            checks[check_id] = workspace.run_check(
                check_id, timeout_seconds=remaining
            )
        for name, outcome in checks.items():
            print(f"\n{name.upper()}\n{outcome}", flush=True)
        checks_pass = all(
            bool(outcome.splitlines()) and "exit=0" in outcome.splitlines()[0]
            for outcome in checks.values()
        )
        marker_pass = output_marks_milestone_complete(output, milestone.id)
        cycle = {
            "milestone": milestone.id,
            "manager_output": output,
            "checks": checks,
            "checks_passed": checks_pass,
            "approval_markers_present": marker_pass,
        }
        cycles.append(cycle)
        if not checks_pass:
            stop_reason = "independent Python or browser check failed"
            break
        if not marker_pass:
            stop_reason = "manager did not report both milestone completion and reviewer approval"
            break
        completed.add(milestone.id)

    if len(cycles) >= args.max_cycles and len(completed) < len(POSTFLOP_MILESTONES):
        stop_reason = "maximum cycle count reached"
    elif time.monotonic() >= deadline and len(completed) < len(POSTFLOP_MILESTONES):
        stop_reason = "hour budget reached"

    patch_path = workspace.save_patch()
    report_dir = workspace.source_root / "artifacts" / "agent_runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"postflop-manager-{stamp}.json"
    report_path.write_text(json.dumps({
        "goal": "heads-up postflop Scenario Lab",
        "started_at": datetime.fromtimestamp(time.time() - (time.monotonic() - started), timezone.utc).isoformat(),
        "completed_milestones": sorted(completed),
        "cycles": cycles,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "patch": str(patch_path) if patch_path else None,
        "workspace": str(workspace.workspace_root) if args.keep_workspace else None,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nPOSTFLOP MANAGER STOPPED: {stop_reason}")
    print(f"Cycle report: {report_path}")
    if patch_path:
        print(f"Reviewable patch: {patch_path}")
        print("The patch remains unapplied; inspect it before using git apply.")
    else:
        print("No source changes were produced.")
    if args.keep_workspace:
        print(f"Workspace retained at: {workspace.root}")
    if run_error is not None:
        return run_error
    if stop_reason != "milestone queue complete":
        return RuntimeError(stop_reason)
    return None

def main() -> None:
    raise SystemExit(asyncio.run(run_task(parse_args())))


if __name__ == "__main__":
    main()
