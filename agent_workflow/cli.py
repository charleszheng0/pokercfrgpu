"""CLI entry point for the isolated manager/editor/reviewer workflow."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from .workspace import IsolatedWorkspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manager/editor/reviewer workflow against an isolated project snapshot."
    )
    parser.add_argument("--task", required=True, help="One bounded task for the manager to coordinate.")
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
    return parser.parse_args()


async def run_task(args: argparse.Namespace) -> int:
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

def main() -> None:
    raise SystemExit(asyncio.run(run_task(parse_args())))


if __name__ == "__main__":
    main()
