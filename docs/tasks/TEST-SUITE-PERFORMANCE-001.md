# Full-suite performance

**Version 1.0** Owner request: branch from PR #18 and reduce the full test suite from the reported 24 minutes to five minutes or less.

## Contract

Routine test infrastructure maintenance. Preserve the numerical library, every existing test scenario and assertion, real clean-install checks, and exact 100% owned statement and branch coverage. The owner request authorizes the implementation and review; prior specialist spectrum-fitting repair allowances do not apply to this separate task.

| Scenario | Required behavior |
| --- | --- |
| P1 | The complete original pytest/doctest collection executes exactly once. |
| P2 | Bounded workers own their temporary paths, logs, receipts and fresh parent/child coverage. Failed, skipped, missing or duplicate execution fails verification and retains diagnostics. |
| P3 | Runtime fingerprints read fresh bytes on every invocation and preserve the original identity, alias and startup-hook rules. |
| P4 | Instrumentation covers owned modules at every depth and copied CLI helpers without tracing unrelated virtual-environment dependencies. No handwritten coverage exclusions are added. |
| P5 | The canonical full verification command completes in at most 300 seconds on the measured host, with its existing quality, audit, build/install and exact coverage gates. Host measurements do not establish all-platform performance. |

## Evidence

PR #18 head is `576602a6497ce1b893c2adce61c8a0a56136a592`. Its [retained CI run](https://github.com/david-hoffman/dphtools/actions/runs/37424307829) passed 2,128 tests; Linux shard receipts measured approximately 1,008 and 1,069 seconds. Clean-install checks and runtime-fingerprint regression tests dominated those durations.

On this managed Linux host, one original runtime fingerprint read approximately 3 GB across more than 62,000 files and took approximately ten seconds. A regression test also demonstrated that the original `*/dphtools/**/*.py` include pattern instrumented third-party dependencies inside a repository-local environment when the checkout directory was named `dphtools`.

The first local full attempt stopped at an audit-cache write to a read-only home directory. Writable temporary caches resolved that environment failure. The subsequent serial profiling run was interrupted during release tests; its incomplete test/coverage results are retained as diagnostics and are not passing baseline evidence. Complete receipts and logs remain outside Git under `reports/verification/`; no old full-suite evidence is reused.

## Current state

- Authorization and route: owner-authorized routine infrastructure work from PR #18; original product behavior and scientific assertions preserved.
- Candidate pointer: this branch; final candidate identity and measured results will be recorded outside its tracked tree.
- Checks: targeted owned-depth/foreign-runtime instrumentation regression passes. Worker and fingerprint regression checks pass, including resealed foreign-worker receipt rejection. Full candidate verification is pending; a fresh independent reviewer is examining the candidate.
- Next action: integrate the measured fingerprint and worker changes, complete full verification, then obtain independent candidate review.
- Metrics: five scenarios; author/helper launches 3; independent reviewer launches 1; no specialist rounds or repairs; no fixed owner budget or reliable dollar metering supplied.
