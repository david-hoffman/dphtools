# Local verification performance

Version 1.0. The owner explicitly authorized local investigation and optimization
on 2026-10-06. The [contract](LOCAL-VERIFICATION-PERFORMANCE-001-contract.md)
transcribes that request and its retained gates. No second approval is needed for
the authorized fixture optimization.

## Baseline and evidence

Baseline: `e557c7d5822be5d8413d039454782eb7772c9074`, unchanged tracked tree.
Canonical command: `/private/tmp/dphtools-verification-env/bin/python tools/verification.py full`.
Fresh locked environment prepared locally with Python 3.13.12 and
`python -m pip install --require-hashes -r requirements-dev.lock`. Environment
provisioning is a command precondition; no required full-command step is moved
outside its timer. Hardware: Mac16,12, 10 physical/logical cores, 16 GiB RAM.
Worker/thread settings and all installed versions are recorded in the external
[measurement directory](/private/tmp/dphtools-verification-measurements).

Initial cold-tool-cache profile: 133.938941875 s, exit 0, 1,232 passed, 185 warnings,
zero failed/errored/skipped tests, exact 2,094/2,094 statements and 544/544 branches.
The 13 scheduled tool steps plus report validation passed. Tests consumed
107.6 s of external elapsed time; audit 14.9 s; types 7.1 s. JUnit module totals:
verifier tests 40.8 s; hook tests 15.5 s; rolling-ball tests 10.2 s; doctor tests
6.9 s. The baseline already meets 200 s; optimize only justified redundant setup.
Raw reports: `reports/verification/full-zz2e8v_p/`. External monotonic timing,
stdout, stage boundaries, hashes, resource settings and tree-state checks are in
`/private/tmp/dphtools-verification-measurements/baseline-cold.{json,log}`.

Cold-tool-cache means this arm's controlled XDG/pip/Matplotlib caches and the
checkout's mypy/pytest/build/egg-info/bytecode caches are removed inside the timer.
Shared platform-managed Black/audit HTTP caches and OS caches are not claimed
cold. Warm runs retain each arm's caches. The same environment/interpreter and
dependency versions serve both arms sequentially; each full run freshly builds
and reinstalls its own wheel before tests. No extra worker/CPU allocation is used.

## Reviewed test preparation

Fresh root A consolidated the invariant Black stand-in preflight from 48 launches
to one per module. Every test retains a new private command fixture. No test
case, assertion, numerical contract, timeout, hook, runtime file, configuration,
or gate changed. The [purpose mapping](LOCAL-VERIFICATION-PERFORMANCE-001-test-rationale.md)
documents the retained proof and limits. A's 48-case baseline/candidate checks
both passed; that single subset pair is not full-command performance evidence.

Fresh root B accepted round 1, identifying `tests/test_verification.py` SHA256
`804eeebd89e7f52ebe879079326e1c490fee241de64a74cc0ead555f018aba95`.
Its fresh 48 verifier and 24 hook cases passed with zero failures/errors/skips.
The subprocess launch observer confirmed one preflight, all 48 cases retained
private fixtures, and preserved test/helper syntax-tree hashes matched. B found
no defect or implementation exposure. Raw evidence: `reports/performance-A/`
and `reports/performance-B/review.md`, `fresh-runs.json`, and
`verification-probe-receipt.json`. The coordinator commits this accepted checkpoint;
fresh C requires no product edits and runs the complete candidate gate.

Final results will distinguish the initial profile at the candidate worktree path
from the additional cold priming run at the immutable baseline worktree path.
Both are retained. Later warm baseline/candidate runs alternate sequentially.
The timing harness additionally verifies the lock and installed dependency
versions (apart from the distribution under test) against the frozen environment.

## Current state

Approvals: owner request; P1/P2 and all inherited regression contracts retained.
Candidate/checkpoint/reviews/check results, final measurements, blockers, next
action, and metrics will be maintained in the originating task conversation and
its linked local evidence after the reviewed checkpoint is committed. This is
the single Current state pointer, prepared before final candidate verification;
the candidate's own hash/results are recorded outside its tracked tree.
