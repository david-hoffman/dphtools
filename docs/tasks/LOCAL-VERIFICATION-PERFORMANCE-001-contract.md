# Local verification performance contract

Version 1.0. Owner-authorized in the task request on 2026-10-06.

## Scope and acceptance

Optimize the complete local `python tools/verification.py full` command on fixed
compute. Required elapsed time is at most 300 seconds; plan for 200 seconds.
The measured baseline already meets both limits. Prefer a small reduction in
redundant test setup over changes to numerical behavior or the gate.

Preserve the [local verification contract](LOCAL-VERIFICATION-CONTRACT.md), all
existing scientific behaviors, numerical assertions/tolerances, real command
boundaries, clean-install checks, and every quality/audit/build/install/test/report
gate. Require exact 100% measured statements and branches globally and per owned
package, including subprocesses, copied helpers, and never-imported files.
No workers, threads, cores, exclusions, skips, weakened checks, longer deadlines,
remote execution, pushes, pull requests, or publication may be added.

The authorized test-only slice may consolidate demonstrably redundant fixture
preparation. Keep all existing test cases and assertions unless a separately
reviewed purpose-to-retained-test mapping establishes equivalence. Do not change
runtime code, workflows, dependency versions, coverage/discovery settings, skills,
delivery instructions, or numerical data. No new product behavior is approved.

| ID | Input/context | Observable outcome |
| --- | --- | --- |
| P1 | Fresh controlled tool caches; complete command | All gates pass; elapsed time at most 300 s, planning target 200 s |
| P2 | Warm controlled tool caches; complete command | All gates pass under the same resource settings; elapsed time at most 300 s, planning target 200 s |

All existing contract scenarios remain mandatory regression requirements. P1/P2
add timing conditions; they do not consolidate the existing behavioral scenarios.

## Measurement and review

Freeze baseline `e557c7d5822be5d8413d039454782eb7772c9074` and the full command
before edits. Time externally from preparation/launch to final process exit.
Run locally and sequentially with Python 3.13.12 and the unchanged hashed lock,
one test worker, one Black worker, and one numerical-library thread. Use the same
10-core Mac16,12 host with 16 GiB RAM and unchanged CPU allocation. Clear only
declared tool/repository caches for cold runs; do not claim cold OS caches.
Report cold and warm separately. Retain every attempt and failure. Obtain one
cold plus three warm complete candidate measurements and sufficient matched
baseline measurements. Alternate later baseline/candidate warm measurements.
Required cache preparation remains inside the timer.

A uses `design-tests` for test/fixture consolidation without reading runtime
implementation or history. B uses `review-work` tests mode in a fresh root
session. Commit the accepted checkpoint. Fresh C uses `implement-task` and may
make zero product edits. D uses `review-work` candidate mode in a fresh root
session after complete passing evidence. Disable optional memory and delegation
inside roles. Use source-free warnings/tracebacks for A/B checks and disclose any
exposure. Independence is procedural, not an access-control guarantee.

Default allowances: two A/B reviews per window; one C repair after initial C.
No dollar cap was specified and reliable dollar metering is unavailable. Record
elapsed preparation, execution, reviews, and failed attempts; naming this task
does not replenish exhausted scientific repair allowances. Complete with local
changes, matched measurements, retained-case/setup rationale, and independent
review, or a specific classified blocker.
