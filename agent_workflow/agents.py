"""Manager, editor, and reviewer definitions for the Agents SDK."""

from __future__ import annotations

from agents import Agent, function_tool

from .workspace import IsolatedWorkspace


def make_agents(workspace: IsolatedWorkspace, model: str | None = None) -> Agent:
    """Create a manager agent with bounded editor and read-only reviewer tools."""

    @function_tool
    def list_project_files(subdirectory: str = ".") -> str:
        """List repository-relative files in the isolated project snapshot."""
        return workspace.list_files(subdirectory)

    @function_tool
    def read_project_file(path: str) -> str:
        """Read one UTF-8 text file by repository-relative path."""
        return workspace.read_file(path)

    @function_tool
    def apply_unified_patch(patch: str) -> str:
        """Apply a unified Git patch to the isolated snapshot, never the original checkout."""
        return workspace.apply_patch(patch)

    @function_tool
    def inspect_working_diff() -> str:
        """Show the current diff from the isolated snapshot's initial state."""
        return workspace.diff()

    @function_tool
    def run_project_check(check_id: str) -> str:
        """Run one allowlisted check: python_tests, browser_demo, or browser_showcase."""
        return workspace.run_check(check_id)

    editor = Agent(
        name="Poker Project Editor",
        model=model,
        instructions=(
            "You implement one bounded repository task in the isolated project snapshot. "
            "First inspect the relevant files and current status. Use apply_unified_patch for "
            "changes; do not claim a change was made unless its tool succeeded. Run only the "
            "checks needed for the task and report their exact outputs. Preserve unrelated "
            "existing changes. Never invent measurements, solver results, or test success. "
            "Return: goal, changed files, patch summary, checks and actual outcomes, limitations."
        ),
        tools=[list_project_files, read_project_file, apply_unified_patch, inspect_working_diff, run_project_check],
    )

    reviewer = Agent(
        name="Poker Project Reviewer",
        model=model,
        instructions=(
            "You are a read-only independent reviewer. Inspect the current snapshot diff and "
            "relevant source files. You cannot apply patches. Check poker rules, card removal, "
            "range semantics, counterfactual versus conditional EV, payoff units, solver claims, "
            "API/UI data flow, and whether supplied check outputs demonstrate the stated behavior. "
            "Do not claim to have run checks; the editor owns check execution. Return "
            "VERDICT: APPROVE or CHANGES REQUIRED; BLOCKERS with smallest fixes; TESTS; CLAIMS."
        ),
        tools=[list_project_files, read_project_file, inspect_working_diff],
    )

    return Agent(
        name="Poker Project Manager",
        model=model,
        instructions=(
            "You are the technical manager for a heads-up postflop Hold'em Scenario Lab. "
            "Preflop solving and the playable game are out of scope unless the user changes "
            "that goal. You own the task and final report. Use the editor and reviewer as bounded tools. "
            "First ask the editor to inspect current code and implement only the requested "
            "bounded task. Then give the reviewer the editor's report and request an independent "
            "review of the actual diff. If the review identifies a blocker, ask the editor to "
            "address it once, then ask for a final review. Do not exceed one correction cycle. "
            "Do not report completion unless the relevant checks actually passed and the final "
            "review has no blockers. If checks fail, review is inconclusive, or the requested "
            "scope is too large, stop and report the blocker. Never claim GTO/equilibrium or "
            "solver quality beyond the supplied evidence. Distinguish measured facts from plans. "
            "For a postflop queue cycle, end with the exact requested POSTFLOP_MILESTONE and "
            "POSTFLOP_REVIEW marker lines. Mark COMPLETE/APPROVE only when the reviewer said "
            "APPROVE and actual check output supports the claim; otherwise use BLOCKED or "
            "CHANGES_REQUIRED."
        ),
        tools=[
            editor.as_tool(
                tool_name="ask_editor_to_implement",
                tool_description="Inspect and implement one bounded change in the isolated checkout.",
            ),
            reviewer.as_tool(
                tool_name="ask_reviewer_to_review",
                tool_description="Independently inspect the current diff and report blockers without editing.",
            ),
        ],
    )
