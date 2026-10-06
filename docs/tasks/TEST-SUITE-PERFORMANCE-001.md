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

The profiled PR #18 runtime candidate is `576602a6497ce1b893c2adce61c8a0a56136a592`. Its [retained CI run](https://github.com/david-hoffman/dphtools/actions/runs/37424307829) passed 2,128 tests; Linux shard receipts measured approximately 1,008 and 1,069 seconds. Clean-install checks and runtime-fingerprint regression tests dominated those durations. The branch also preserves PR #18's subsequent documentation update at `0634499296e143bab6a16daba3f504b5c6e4391d`.

On this managed Linux host, one original runtime fingerprint read approximately 3 GB across more than 62,000 files and took approximately ten seconds. A regression test also demonstrated that the original `*/dphtools/**/*.py` include pattern instrumented third-party dependencies inside a repository-local environment when the checkout directory was named `dphtools`.

The first local full attempt stopped at an audit-cache write to a read-only home directory. Writable temporary caches resolved that environment failure. The subsequent serial profiling run was interrupted during release tests; its incomplete test/coverage results are retained as diagnostics and are not passing baseline evidence. Complete receipts and logs remain outside Git under `reports/verification/`; no old full-suite evidence is reused.

The first integrated parallel attempt exceeded the target and exposed a worker scratch directory inside the checkout, violating an existing installed-probe assertion. It was interrupted after that confirmed failure; its partial coverage is diagnostic evidence. The corrected workers keep scratch files outside the checkout. Final measurements use a dedicated CPython installation with the unchanged hashed dependency lock rather than the host's ambient base installation.

## Current state

- Authorization and route: owner-authorized routine infrastructure work from PR #18; original product behavior and scientific assertions preserved.
- Candidate pointer: this branch; final candidate identity and measured results will be recorded outside its tracked tree.
- Checks: targeted instrumentation, worker and fingerprint regressions pass. Independent review identified worker artifact retention and serial lifecycle validation gaps; both corrections and the outside-checkout scratch repair are being verified. Final full verification and review are recorded outside the tracked candidate.
- Next action: verify the final candidate against the five-minute target and obtain independent acceptance; keep hosted platform results explicit.
- Metrics: five scenarios; author/helper launches 3; independent reviewer launches 1; no specialist rounds or repairs; no fixed owner budget or reliable dollar metering supplied.
