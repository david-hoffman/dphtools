# Agent Harness Workflow

## Phase 0 Status

Phase 0 is enabled. It covers repository intake, local harness docs, references, templates, validation scripts, and CI scaffolding.

Phase 2 clean-context metadata enforcement is documented but not yet blocking.

## Normal Issue State Flow

```text
agent-ready
  -> needs-red-tests
  -> red-tests-ready
  -> implementation-ready
  -> needs-adversarial-review
  -> needs-numerics-review, if applicable
  -> ready-for-ci
  -> ready-for-human-review, if applicable
  -> done
```

Blocked states:

```text
agent-blocked
needs-human-decision
needs-ci-triage
```

Phase 0 harness-only pull requests may skip red-test states when they make no product behavior changes. The pull request must state that explicitly.

## Required Labels

```text
agent-ready
agent-blocked
agent-running
needs-red-tests
red-tests-ready
implementation-ready
needs-adversarial-review
needs-numerics-review
needs-ci-triage
needs-human-decision
ready-for-ci
ready-for-human-review
done
risk:low
risk:medium
risk:high
area:ci
area:harness
area:docs
area:tests
area:packaging
area:numerics
area:api
area:release
area:downstream
area:hardware
```

## Required Local Commands

- `make bootstrap`
- `make test-fast`
- `make coverage`
- `make harness-check`
- `make package`
- `make check`

## Pull Request Rules

- Link an issue unless the maintainer explicitly approves a no-issue administrative change.
- State the phase.
- Paste exact commands and concise results.
- State product behavior impact.
- State public API impact.
- State scientific, hardware, or numerical impact as facts, assumptions, and guesses.
- Do not mark a pull request ready if required harness evidence is missing.

## Risk Defaults

- Harness, CI, docs, and test-only work: usually `risk:medium`.
- Product bug fixes with narrow behavior and tests: usually `risk:low` or `risk:medium`.
- Public API, release, security, branch governance, hardware, or numerical tolerance changes: `risk:high` unless a human maintainer says otherwise.
