# Manager, editor, and reviewer workflow

This optional prototype uses the OpenAI Agents SDK to coordinate three roles:

- **Manager/orchestrator:** owns the task, calls the editor and reviewer, limits the correction cycle, and writes the final report.
- **Editor:** inspects the repository snapshot, proposes/applies a unified diff there, and runs allowlisted checks.
- **Reviewer:** receives the editor's report and can inspect files, diffs, and checks. Its tools are read-only.

The SDK supports this manager pattern by exposing specialist agents as tools. The manager retains control and passes a bounded request to each specialist. This is different from a handoff, where the specialist takes over the conversation. See the [OpenAI orchestration guide](https://developers.openai.com/api/docs/guides/agents/orchestration) and [Python SDK quickstart](https://openai.github.io/openai-agents-python/quickstart/).

## Isolation and approval boundary

Agents work in a temporary copy of the repository, including the current working-tree files. Their code edits do not touch the source checkout. The CLI saves the resulting unified patch under `artifacts/agent_runs/`; a person can inspect it and apply it with `git apply`. This avoids accidental source changes while learning the workflow and makes the reviewer inspect the actual diff.

This is an isolation boundary for ordinary coding mistakes, not a hardened security sandbox: checks run local project code in the current Python environment, and an existing `node_modules` directory is linked read-only by convention for browser test resolution. Do not use it with untrusted repositories or tests. The reviewer has no patch tool, but a language model's verdict is not a formal proof; required checks and human review remain important.

The manager is prompted to perform one edit, one review, and at most one correction/re-review cycle. The CLI also caps manager turns. It does not auto-merge, deploy, or apply patches to the source checkout.

## Install and run

Use Python 3.10 or newer and install the optional SDK dependency:

```bash
python -m pip install -r requirements-agents.txt
export OPENAI_API_KEY="your-api-key"
```

Start with a local dry run. It creates an isolated snapshot but does not call the API:

```bash
python3 -m agent_workflow.cli --repo . --task "Inspect the current river solver path" --dry-run
```

Then run a narrowly scoped task:

```bash
python3 -m agent_workflow.cli \
  --repo . \
  --task "Trace one supported river request through the solver and add a focused regression fix if a concrete bug is found. Do not expand to flop or turn."
```

The CLI supports `python_tests`, `browser_demo`, and `browser_showcase` as allowlisted checks. The agents can request checks by ID, not arbitrary shell commands. The resulting patch is **not** applied automatically. Review the manager report and patch, then apply only if you agree:

```bash
git apply artifacts/agent_runs/agent-run-<timestamp>.patch
```

Use `--keep-workspace` to retain the temporary copy for inspection. Do not put the API key in a prompt or project file; the SDK reads it from the environment.

## Poker-project guardrails

- Treat `docs/CURRENT_STATE.md` and saved experiment artifacts as dated evidence; verify them against code before relying on them.
- Keep game correctness, small-game validation, and restricted Hold'em evidence separate.
- Never present assumed ranges as observed betting-history ranges.
- Do not call restricted-game NashConv a full Hold'em exploitability measure.
- Keep NFSP separate from CFR. Do not use the previous failed continuation model without new held-out decision-quality evidence.
- Do not begin CUDA work until CPU correctness, relevant solver quality, and measured bottlenecks support it.
- Update experiment and benchmark documents only with measurements actually produced by the workflow.

## How information moves

The editor and reviewer do not share a spontaneous side conversation. The SDK manager sends each a task input, receives the tool result, and decides the next call. The reviewer sees the diff through read-only workspace tools; if it finds a blocker, the manager can pass that feedback to the editor for one revision. The final answer is still produced by the manager.

The first useful learning exercise is a dry run, followed by one small task with `--keep-workspace`. Inspect the generated patch and trace before increasing autonomy or widening the available tools.

## Postflop solver manager

The repository now has an ordered manager queue dedicated to the heads-up postflop Scenario Lab. It deliberately excludes preflop and gameplay work. The queue covers: a compact street/action-specific tree and references, chance-aware flop/turn/river coverage, convergence and independent validation, measured study-spot selection, the minimal Scenario Lab UI, and a final release gate. It keeps range, chance, and tree limits until measured profiling supports increasing them. See `agent_workflow/postflop_plan.py` for the exact objectives and acceptance criteria.

Run a no-API dry run first:

```bash
python -m agent_workflow.cli --repo . --postflop-manager --dry-run
```

After installing `requirements-agents.txt` and setting `OPENAI_API_KEY` in the launching terminal, start a bounded run:

```bash
python -m agent_workflow.cli --repo . --postflop-manager --max-cycles 6 --hours 2 --keep-workspace
```

Each cycle uses the manager to direct the editor and independent read-only reviewer. The CLI then runs Python and browser checks itself. It advances only when both checks pass and the manager reports explicit completion and reviewer approval markers. A failed check, missing approval, API error, cycle limit, or time limit stops the run and writes a report under `artifacts/agent_runs/`. Proposed source edits stay in the isolated workspace and are exported as one reviewable patch; the manager never applies them to the user's checkout. This is software-development orchestration, not CFR/NFSP training, and it cannot run while this computer is asleep.

The model/API key was not present when this workflow was implemented, so only its offline dry-run and local tests can be verified here. A live manager/editor/reviewer cycle requires the user to install the optional SDK and provide the key in their own terminal.
