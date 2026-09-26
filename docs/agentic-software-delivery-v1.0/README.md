# Agentic Software Delivery System

**Version 1.0** Repository setup is in progress. See [the setup task](../tasks/SETUP-001.md) for actual verification and remaining gaps; this document does not certify readiness.

> You decide what to build. Separate agents write tests, implement, and review. GitHub runs the checks. Lessons improve the next task.

## The whole scheme

**One monorepo, four skills, ordinary CI, and a learning log. No custom delivery platform.**

First, reuse the approved architecture. If there isn't one, the intake conversation helps create it. For an existing application, it documents what is there and resolves what is intended. For an empty repository, it starts with your goals. You approve the resulting project record.

For each change:

```text
Clarify the request → you approve the task
    → A writes tests → B reviews tests
    → tests fail for the intended reason; save the test checkpoint
    → C implements → automated checks → D reviews
    → normal GitHub merge
```

A–D are fresh sessions, not four names in the same conversation. They may use the same model. The intake conversation is additional. Tests focus on actual user behavior: browser journeys, commands, or public APIs. Smaller tests fill genuine gaps, rather than duplicating everything.

## What you do

Describe the outcome. Answer the consequential questions. Approve the architecture, task, budget, and any material change. Review the final summary and merge normally. You are not expected to write code or pretend to perform expert code review.

The agents are told not to change reviewed tests or workflows to make their work pass. **That restriction is a prompt, not a technical barrier.** GitHub CI still runs the configured tests, but an agent could change those rules. This is an intentional simplicity tradeoff, not a guarantee of bug-free software.

## How the system learns

Agents append useful surprises and gotchas to [LESSONS.md](../../LESSONS.md), with evidence. They do not dump conversations there.

Run `delivery doctor` when experience reveals a gap. It checks the evidence and edits the existing specification/instructions on a documentation branch. You review the diff before it is committed and merged. Git records revisions. It does not repair a failing feature by rewriting the rules.

## Start here

| File | Use it for |
|---|---|
| [SETUP-PROMPT.md](SETUP-PROMPT.md) | Give a coding tool this prompt and the package to set up the repository. |
| [DELIVERY-SYSTEM-SPEC.md](DELIVERY-SYSTEM-SPEC.md) | The single authoritative implementation specification. |
| [INTAKE-PROMPT.md](INTAKE-PROMPT.md) | Clarify an architecture, feature, or bug request. |
| [DOCTOR-PROMPT.md](DOCTOR-PROMPT.md) | Improve the specification from observed gaps, even before the command exists. |
| [GITHUB-SETUP.md](GITHUB-SETUP.md) | Configure ordinary CI and branch protections. |
| [WORKFLOW-EXAMPLES.md](WORKFLOW-EXAMPLES.md) | See an empty-project start, a feature, and a doctor update. |

The repository `.agents/skills/` directory contains the four canonical procedures; Codex reads them natively. `templates/` contains only project/task starters and repository instructions. [REFERENCES.md](REFERENCES.md) explains sources and deliberate omissions.

Do not overwrite an existing product README by copying this bundle into its root. Setup should preserve existing docs, install one canonical delivery spec, and update links. No older draft package is needed.

**Vocabulary:** the delivery system is this whole arrangement. A coding harness runs agents. A skill tells a role how to work. Continuous integration (CI) executes checks. No custom controller, external control repository, or tracing tool is part of this design.
