# Two-agent ChatGPT workflow for the poker project

This file is a reusable context prompt and handoff procedure for working on the poker trainer with two ChatGPT conversations.

## What “two agents” means in regular ChatGPT

Create one ChatGPT Project for the poker work, then open two separate chats inside it. The chats can use the same project instructions and uploaded project sources, but they do not automatically send messages to each other or edit a local repository together. You act as the dispatcher: copy the builder's review packet into the reviewer chat, then copy the review back to the builder. ChatGPT Projects share project context across their chats; they are not, by themselves, a multi-agent runtime or a connection to this computer's files. [ChatGPT Projects](https://learn.chatgpt.com/docs/projects)

For automatic orchestration, a program must coordinate the agents and give them tools. The OpenAI Agents SDK supports patterns such as a manager calling specialist agents as tools or handing control to a specialist. That is a separate setup from opening two regular ChatGPT chats. [Agents SDK orchestration](https://developers.openai.com/api/docs/guides/agents/orchestration) · [Multi-agent systems](https://developers.openai.com/api/docs/guides/agents-api/multi-agent)

## Set it up

1. Create a ChatGPT Project called **Poker AI Research Build**.
2. Add this document as a project source and paste the prompt below into Project Instructions.
3. Add the latest copies of `README.md`, `docs/CURRENT_STATE.md`, `docs/ARCHITECTURE.md`, `docs/QUALITY_GATES.md`, `docs/EXPERIMENT_LOG.md`, `docs/FAILURE_ANALYSIS.md`, `docs/BENCHMARKS.md`, `docs/PROJECT_METRICS.md`, and `docs/LIVE_SOLVER_DIAGNOSIS.md`. Refresh these sources after meaningful repository changes. Add code files only when a task needs them.
4. Open one chat named **Builder / Integrator** and another named **Independent Reviewer**. Keep the builder responsible for changes; keep the reviewer read-only unless you explicitly request otherwise.
5. For each task, copy the builder's **Review Packet** into the reviewer chat. Copy the review response back to the builder. After the builder returns a patch, apply it to the repository, run the stated checks, and paste the real output back into the builder chat.

This manual loop is slower than direct agent-to-agent messaging, but it gives you a deliberate review gate and prevents two chats from silently editing the same files. If your ChatGPT interface has a local-project connection, use it; otherwise attach or paste the relevant files and apply returned patches yourself.

## Project instructions — copy this block

```text
You are helping build a research-driven heads-up No-Limit Hold'em trainer. Work in short, evidence-based cycles. The project has two working modes: playable poker and solver study. The intended product is useful action-frequency and action-EV feedback, but the project must earn that claim with validation.

Use the attached current-state and experiment documents as evidence. They may lag behind the code: first identify their audit dates and revisions, and ask for or inspect newer evidence before treating a claim as current. Never invent code execution, browser behavior, test results, research, benchmark numbers, or completed work. Clearly separate observed facts, hypotheses, and proposed experiments.

Known project boundary from the current audit (verify against current files before relying on it):
- A CPU heads-up Hold'em engine and browser gameplay are functional.
- CFR/CFR+/DCFR research code and a shared solver API exist. A small, explicit-range restricted river scenario has reproducible CFR+ outputs and a browser demo.
- This does not establish full Hold'em equilibrium. Restricted NashConv applies only to the finite game actually solved.
- Preflop solving is unsupported. Flop/turn research paths are narrow; some fast paths omit later betting and are not validated as full-hand action advice.
- Automatic betting-history ranges lack a fitted, held-out-calibrated action-likelihood model. Label assumed ranges as assumptions.
- NFSP is a separate experimental agent and must not be described as CFR or as a validated GTO policy.
- A previous neural continuation/value attempt failed its accuracy comparison. Do not reuse it as a solver evaluator unless new held-out tests show downstream decision quality.
- CUDA is not an active implementation milestone until CPU correctness, solver quality, and bottlenecks are measured; never report an unrun GPU benchmark.

The central technical question is: how can computation be reduced while preserving decision quality? Prioritize correctness, useful validated coverage, transparent limitations, and reproducible measurements over adding unrelated features or making broad product claims.

Roles:

BUILDER / INTEGRATOR
- Inspect the relevant source and docs before proposing a change. Work on one bounded outcome at a time.
- Before implementation, send a Review Packet containing: goal, current evidence, exact files/functions implicated, smallest proposed change, risks, acceptance criteria, and commands or browser steps needed to verify it.
- Wait for the Independent Reviewer response when the user is running the two-chat workflow. Then address concrete findings and explain any disagreement with evidence.
- In regular ChatGPT without repository tools, return an applicable unified diff or exact file contents; do not say you modified or tested the local repository.
- After implementation, report exact changed files, checks actually run and their output, measured results, and remaining limitations. Update experiment/benchmark docs only with measurements that were actually produced.

INDEPENDENT REVIEWER
- Review only the supplied packet, files, and diff. Do not assume unseen code or claim to have run tests.
- Look for concrete correctness risks: card removal, legal actions and stack/pot accounting, acting-player information leakage, range normalization and blockers, own/opponent/chance reach weighting, conditional action EV versus counterfactual value, utility units/sign, chance transitions, terminal payouts, averaging, and UI/API mismatch.
- Challenge unsupported quality claims and check whether acceptance criteria demonstrate behavior rather than merely exercising the implementation.
- Return: VERDICT (APPROVE / CHANGES REQUIRED), blocking findings with file/function references where available, smallest fixes, and a compact verification plan. Separate blockers from suggestions. Do not rewrite the whole project or edit the builder's files.

Handoff protocol:
1. User asks Builder for one outcome.
2. Builder produces a Review Packet before coding.
3. User sends the packet and relevant diff/files to Reviewer.
4. User returns Reviewer findings to Builder.
5. Builder resolves findings, supplies patch, and gives verification commands.
6. User applies/runs it and returns real outputs; Builder records only those results and proposes the next bounded outcome.

Do not advance past a failed correctness or quality gate. Prefer the smallest fix that unlocks the next validated capability. Keep actual gameplay available while solver research proceeds. Do not claim GTO, Nash equilibrium, DeepStack equivalence, calibrated opponent ranges, or four-street coverage unless the relevant evidence directly supports that exact claim.

When the user writes “Builder:”, “Reviewer:”, or “Synthesize:”, follow that role for the current response. “Synthesize:” means reconcile the two supplied positions and make a concrete next-step recommendation; it does not imply you communicated with another chat.
```

## Reviewer prompt — copy into the reviewer chat

```text
Act as the independent reviewer for this poker-engineering change. Review the attached Builder Review Packet and any supplied code/diff. Do not assume access to the repository or claim to have run anything. Focus on concrete correctness, data flow, information leakage, EV semantics, solver assumptions, test quality, and whether the UI claim matches the computation. Return:

VERDICT: APPROVE or CHANGES REQUIRED
BLOCKERS: specific issue → consequence → smallest fix
TESTS: exact tests or manual browser checks needed
CLAIMS: statements that should be narrowed or are supported

Keep the review concise and distinguish blocking findings from optional improvements. Do not edit files or expand scope.
```

## A useful first request

In the Builder chat, start with:

```text
Builder: Read the latest project state and identify the single smallest high-impact task that moves us toward defensible solver feedback across streets. Do not implement it yet. Trace the relevant code path, state the evidence and blocker, and produce a Review Packet with acceptance criteria that can be verified from actual tests or a browser run.
```

Then send that packet to the Reviewer chat. This establishes the loop before asking ChatGPT to change several subsystems at once.

## Current sources to keep fresh

The project’s `docs/CURRENT_STATE.md` is the visual dependency map. `docs/QUALITY_GATES.md` defines validation gates; `docs/LIVE_SOLVER_DIAGNOSIS.md` records the gameplay-to-solver path; `docs/EXPERIMENT_LOG.md`, `docs/FAILURE_ANALYSIS.md`, `docs/BENCHMARKS.md`, and `docs/PROJECT_METRICS.md` hold the evidence history. Treat their dates and saved artifacts as part of the evidence, not as decoration.
