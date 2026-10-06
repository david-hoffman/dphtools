# Full-suite performance

**Version 1.0** Owner request: branch from PR #18 and reduce the full test suite from the reported 24 minutes to five minutes or less.

## Contract

Routine test infrastructure maintenance. Preserve the numerical library, every existing test scenario and assertion, real clean-install checks, and exact 100% owned statement and branch coverage. The owner request authorizes the implementation and review; prior specialist spectrum-fitting repair allowances do not apply to this separate task.

| Scenario | Required behavior |
| --- | --- |
| P1 | Every original pytest/doctest node executes exactly once in each platform's full suite. |
| P2 | Bounded workers own their temporary paths, logs, receipts and fresh parent/child coverage. Failed, skipped, missing or duplicate execution fails verification and retains diagnostics. |
| P3 | Runtime fingerprints read fresh bytes on every invocation and preserve the original identity, alias and startup-hook rules. |
| P4 | Instrumentation covers owned modules at every depth and copied CLI helpers without tracing unrelated virtual-environment dependencies. No handwritten coverage exclusions are added. |
| P5 | The complete hosted full-suite execution window is at most 300 seconds, with quality, audit, build/install and exact coverage gates passing. Report local command time and total hosted workflow, setup, collection and aggregation time separately; runner limits and queueing must remain explicit. |

## Evidence

The profiled PR #18 runtime candidate is `576602a6497ce1b893c2adce61c8a0a56136a592`. Its [retained CI run](https://github.com/david-hoffman/dphtools/actions/runs/37424307829) passed 2,128 tests; Linux shard receipts measured approximately 1,008 and 1,069 seconds. Clean-install checks and runtime-fingerprint regression tests dominated those durations. The branch also preserves PR #18's subsequent documentation update at `0634499296e143bab6a16daba3f504b5c6e4391d`.

On this managed Linux host, one original runtime fingerprint read approximately 3 GB across more than 62,000 files and took approximately ten seconds. A regression test also demonstrated that the original `*/dphtools/**/*.py` include pattern instrumented third-party dependencies inside a repository-local environment when the checkout directory was named `dphtools`.

The first local full attempt stopped at an audit-cache write to a read-only home directory. Writable temporary caches resolved that environment failure. The subsequent serial profiling run was interrupted during release tests; its incomplete test/coverage results are retained as diagnostics and are not passing baseline evidence. Complete receipts and logs remain outside Git under `reports/verification/`; no old full-suite evidence is reused.

The first integrated parallel attempt exceeded the target and exposed a worker scratch directory inside the checkout, violating an existing installed-probe assertion. It was interrupted after that confirmed failure; its partial coverage is diagnostic evidence. The corrected workers keep scratch files outside the checkout. Final measurements use a dedicated CPython installation with the unchanged hashed dependency lock rather than the host's ambient base installation.

The next complete local attempt took about 676 seconds and failed: an outer environment's locked tools were unavailable to real nested environments that inherit the base interpreter's packages, a new fixture changed duration inputs after freezing its identity, and serial validation added a blocked receipt step that violated an original hook expectation. Correct the environment setup and fixture ordering, and validate serial execution after successful tests while preserving the original hook assertion. This four-CPU host did not meet five minutes. The owner's request concerns the full suite; the earlier requirement that this specific host's entire verification command finish within 300 seconds was an author assumption. The hosted trial uses four machines per platform with bounded local workers and retains all gates; it needs actual timing evidence before any performance claim.

The corrected eight-worker local run executed all 2,201 nodes and measured every owned statement and branch. It took about 843 seconds and failed one real coverage integration test at its harness's 30-second deadline under CPU contention. That case now allows 90 seconds for real copied-environment provisioning and coverage reporting; its original failure-status, missing-source and branch-measurement assertions remain intact. Other fixture deadlines remain unchanged. The failed run is diagnostic evidence, and hosted acceptance still requires a fresh complete passing run within 300 seconds.

## Current state

- Authorization and route: owner-authorized routine infrastructure work from PR #18; original product behavior and scientific assertions preserved.
- Candidate pointer: this branch; final candidate identity and measured results will be recorded outside its tracked tree.
- Checks: targeted instrumentation, worker and fingerprint regressions pass. Earlier integrated full results remain failed diagnostics, with exact complete owned coverage in the latest run. Its single harness timeout repair is being verified. Hosted distribution preserves the two-shard CLI default and adds strictly sealed counts; final full results and review are recorded outside the tracked candidate.
- Next action: verify the timeout repair and cheap checks on the exact candidate, run the hosted complete suite, measure its execution window against five minutes, and obtain independent acceptance.
- Metrics: five scenarios; author/helper launches 4; independent reviewer launches 1; no specialist rounds or repairs; no fixed owner budget or reliable dollar metering supplied.
