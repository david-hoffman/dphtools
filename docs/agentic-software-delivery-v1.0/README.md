# Agentic Software Delivery System

**Version 1.0** This package specifies a delivery process. It does not install tooling, configure GitHub, or certify repository readiness.

> You decide what to build. An author and a fresh independent reviewer handle routine work; specialists handle scientific or safety risk. GitHub runs the required checks. Lessons improve the next task.

## The whole scheme

**One monorepo, four skills, ordinary CI, and a learning log. No custom delivery platform.**

First, inventory existing failures, measured coverage, environment problems, and unresolved behavior. Separate installing delivery tooling from repairing the product, including their cost and required decisions. Reuse approved architecture or clarify the missing decisions with the owner. Keep the 100% statement and branch coverage requirement; installation alone does not make the product ready.

Routine maintenance reuses established contracts and its existing task/PR record; no new document or specialist is required for every assertion. For new behavior or a large request, intake records necessary contracts/slices and obtains approval only for material decisions not already authorized. New contracts default to at most five distinct scenarios. Independent tasks may use separate worktrees; dependent slices wait for integrated prerequisites. A baseline preventing complete verification and exact 100% coverage is an adoption decision to expose up front, not an exception to the merge gate.

For each approved slice:

```text
Authorized routine task under established contracts
    → author changes code/tests/docs within scope
    → meaningful focused checks and cheap verification pass
    → open/update PR; one fresh independent reviewer checks the candidate
    → full required platform CI, exact owned coverage, verified protection
    → normal owner-authorized merge of the unchanged reviewed candidate

Genuine scientific or safety risk
    → A writes tests → B reviews the contract/oracle and checkpoint
    → C implements → approved risk checks, normally full local reference
    → fresh D review + the same complete protected platform merge gate
```

Classify by changed behavior and lost proof, not patch size. New scientific contracts/custom correctness oracles, release/publication safety, or other material behavior/security risk use fresh specialist A–D sessions. All roles receive narrow packets and concrete completion conditions; reviewers never inherit the author's conversation or approve their own repair. Tests focus on actual user behavior: browser journeys, commands, or public APIs. Expected results follow approved behavior, primary references, or invariants. Existing-code tests may pass initially; no artificial red result or product mutation is required. A claimed bug still needs the intended failure.

Routine PR opening/reopening (drafts included) and updates require focused and cheap checks. Full local verification remains useful as a reference, for diagnosis, and when the risk/check plan requires it. Known failures remain blockers; pending CI is not success. Merge needs complete Linux/macOS/Windows verification and exact 100% owned statement/branch coverage. Rely on `ci-required` as authoritative only after actual native protection on `codex-main` is read back and verified; missing protection is a reported blocker. Preserve production `main` and separate frozen-bundle publication approval. Before a PR exists, pushes may back up incomplete checkpoints without establishing readiness. Preserve fast generic hooks.

Classify failures before repair. On the specialist route, after two B nonacceptances, B diagnoses ambiguity or excessive scope before another rewrite. Keep specialist A/B rounds separate from C repairs and total budget. One Current state section records route, approvals, candidate, checks/review, blockers, and next action, with concise scenario/launch/round/repair/spend metrics. Renaming/splitting never resets used resources. Keep full logs outside Git; host-required progress updates remain mandatory and unavailable event-wait tools remain unavailable.

## What you do

Describe the outcome. Answer the consequential questions. Approve the architecture, task, budget, and any material change. Review the final summary and merge normally. You are not expected to write code or pretend to perform expert code review.

The agents are told not to change reviewed tests or workflows to make their work pass. **That restriction is a prompt, not a technical barrier.** GitHub CI still runs the configured tests, but an agent could change those rules. This is an intentional simplicity tradeoff, not a guarantee of bug-free software.

## How the system learns

During setup, create root `LESSONS/` with a `README.md` format guide. Record each useful, evidence-linked discovery in its own timestamped Markdown file. Add corrections as new files referencing earlier entries; do not update a shared index. Only lesson entries require append-only treatment; Git preserves superseded task status.

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
