# Agentic Software Delivery System

**Version 1.0** This package specifies a delivery process. It does not install tooling, configure GitHub, or certify repository readiness.

> You decide what to build. Separate agents write tests, implement, and review. GitHub runs the checks. Lessons improve the next task.

## The whole scheme

**One monorepo, four skills, ordinary CI, and a learning log. No custom delivery platform.**

First, inventory existing failures, measured coverage, environment problems, and unresolved behavior. Separate installing delivery tooling from repairing the product, including their cost and required decisions. Reuse approved architecture or clarify the missing decisions with the owner. Keep the 100% statement and branch coverage requirement; installation alone does not make the product ready.

For a large request, intake proposes small end-to-end slices; the owner approves the plan and all contracts in one read-back. A request that fits one slice needs only its task document. Each task has one checkpoint lineage and, by default, at most five distinct contract scenarios. Independent tasks may run in separate worktrees; dependent slices wait for completed, integrated prerequisites. A legacy baseline that makes a slice unable to reach full verification and global 100% coverage is an adoption decision to expose up front, not an exception to the gate.

For each approved slice:

```text
Approved task or slice
    → A writes tests → B reviews tests and their decision logic
    → establish valid baseline evidence; save the test checkpoint
    → C implements or confirms no product change is needed
    → full local verification passes on the exact candidate
    → open/update the PR and run CI; D reviews the passing candidate
    → normal GitHub merge after green CI and D's review
```

A–D are fresh root sessions with narrow packets and concrete completion conditions. Tests focus on actual user behavior: browser journeys, commands, or public APIs. Expected results follow approved behavior, applicable primary references, or mathematical invariants. Existing-code tests may pass initially; no artificial red result or product mutation is required. A claimed bug still needs the intended failure.

Full verification gates opening or reopening a PR, including a draft, and pushes updating an open PR. Before a PR exists, pushes may back up failing checkpoints. Backup is not submission or readiness; closing a PR or moving a branch does not bypass the submission gate.

Classify failures before repair. After two B reviews without acceptance, B diagnoses contract ambiguity or an oversized slice before another rewrite. Keep A/B rounds separate from C repairs and total budget. One compact line in Current state records scenario count, A/B rounds, C repair use, and spend when known; renaming or splitting work does not erase history or consumed resources.

## What you do

Describe the outcome. Answer the consequential questions. Approve the architecture, task, budget, and any material change. Review the final summary and merge normally. You are not expected to write code or pretend to perform expert code review.

The agents are told not to change reviewed tests or workflows to make their work pass. **That restriction is a prompt, not a technical barrier.** GitHub CI still runs the configured tests, but an agent could change those rules. This is an intentional simplicity tradeoff, not a guarantee of bug-free software.

## How the system learns

During setup, create root `LESSONS.md` for useful, evidence-linked discoveries. Only that log requires append-only treatment; Git preserves superseded task status.

Once installed, run `delivery doctor` when experience reveals a gap; before installation, use [DOCTOR-PROMPT.md](DOCTOR-PROMPT.md). It distinguishes instruction gaps from execution errors and proposes edits to the existing specification/instructions. You review the diff before it is committed and merged.

## Start here

| File | Use it for |
|---|---|
| [SETUP-PROMPT.md](SETUP-PROMPT.md) | Give a coding tool this prompt and the package to set up the repository. |
| [DELIVERY-SYSTEM-SPEC.md](DELIVERY-SYSTEM-SPEC.md) | The single authoritative implementation specification. |
| [INTAKE-PROMPT.md](INTAKE-PROMPT.md) | Clarify an architecture, feature, or bug request. |
| [DOCTOR-PROMPT.md](DOCTOR-PROMPT.md) | Improve the specification from observed gaps, even before the command exists. |
| [GITHUB-SETUP.md](GITHUB-SETUP.md) | Configure ordinary CI and branch protections. |
| [WORKFLOW-EXAMPLES.md](WORKFLOW-EXAMPLES.md) | See an empty-project start, a feature, and a doctor update. |

The repository `.agents/skills/` directory contains the four role procedures. Setup adapts them to the selected harness and creates or reconciles repository `AGENTS.md` as the single operational home for shared instructions. `templates/` contains project/task starters and repository instructions. [REFERENCES.md](REFERENCES.md) explains sources and deliberate omissions.

Do not overwrite an existing product README by copying this bundle into its root. Setup should preserve existing docs, install one canonical delivery spec, and update links. No older draft package is needed.

**Vocabulary:** the delivery system is this whole arrangement. A coding harness runs agents. A skill tells a role how to work. Continuous integration (CI) executes checks. No custom controller, external control repository, or tracing tool is part of this design.
