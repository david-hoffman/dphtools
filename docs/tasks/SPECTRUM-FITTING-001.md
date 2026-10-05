# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Contract revision R3 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This scientific task is separate from the routine-maintenance ten-task pilot.

## Current state

- Approval/scope: owner approved R3 and 33 scenarios as one slice on 2026-10-05:
  height amplitudes, positive amplitudes/widths, free centers, actual
  Levenberg-Marquardt default and physical custom optimizer. Preserve existing
  interfaces. Commit/publish codex/spectrum-fitting is authorized; merge/release
  is not. Runtime baseline a835a490373835fa01e4460ce2159365150d1946.
- Final evidence pointer: exact final candidate identity, commands/results,
  D/platform outcomes and disposition belong in this task's conversation,
  outside its tracked tree. Prepare this pointer before verification and do not
  edit a verified candidate to record its own hash/results.
- Published historical candidate: C2 commit
  820e17dc31d9121b7b1ff2f5c839c0e77d8ad70f. Canonical full passed 2116 tests with
  186 retained warnings, 3333/3333 owned statements and 856/856 branches across 20
  owned files; zero failures/errors/skips/exclusions. Normal pre-push fast passed
  and GitHub readback matched. This passing suite missed the following defect.
- D1 rejected 820e17d: a finite constant background of 1e12 data units silently
  understates Gaussian sigma variance about15.3-fold. Analytic physical Jacobian
  conditioning is 3.257; default LM and a real custom adapter reproduce it. The
  coordinator's public reproduction also fails. This is a product covariance
  defect within R3/S06/S09/S31, not a new requirement or environment failure.
  Independent report/reproduction receipts remain outside Git in the conversation.
- CI run 37370536644 attempt 1 was cancelled after D rejection. macOS and Windows
  collection passed; Linux was cancelled while queued. Actual available logs and
  artifacts are retained; no complete platform success or readiness is claimed.
- Reviewed baseline: b62450a206b6f3592c6b8aab712d34fb42f7f08f,285cases. All reviewed
  test/oracle/tolerance AST and62168 test-prefix bytes remain exact SHA-256
  50bb168312355b5c3035f534195260f4a0081a6f171cdd6a6de52235774837dd.
  [Historical report](SPECTRUM-FITTING-001-tests.md) prefix136921 bytes remains
  exact SHA-25673bf5276ddf3a315544f37627701615ab9651b9f967f9aebe96c8bb0b528c55c.
- Accepted A11/A12 revision: 287cases;285passed/2failed at the intended full-covariance
  check for default LM/custom TRF; Black99 passed, source exposure none reported.
  Corrected test SHA-25633501511bff9dcf676821606341516790eb30ffec1319eec27dbb730835e617d;
  report SHA-256e44db5ca4e2ff95d6cdc49e8873267f2b03999f7e3b6853b9b9fc776b6aa386b.
  B8 round1/2 rejected only custom-test exception bookkeeping: a permitted
  numerical RuntimeError is mistaken for a failed protocol assertion. Covariance
  math/tolerance and both intended finite-covariance failures were independently
  confirmed. Fresh A12 corrected the marker with 82 independent outcomes. B9 accepted
  this exact revision at round 2/2 after 86 independent outcome checks. The window
  is accepted/closed; both finite-covariance failures are valid product-red.
- Review lineage: original B2/2, formatting1/2, coverage1/2, numerical1/2 and
  prior post-repair2/2 accepted/closed. The later independently evidenced missing
  covariance case opens this new window under AGENTS.md; no prior rounds or costs
  reset. All previous attempts/proofs remain in the historical report/receipts.
- Allowance: initial C and both authorized repairs are used 2/2. The second was
  explicitly approved for custom numerical-exception normalization and completed
  in 820e17d. One further C repair for the covariance defect is requested and
  awaiting owner extension. No product repair proceeds before that approval.
- Environment/limits: disposable non-Conda CPython 3.12.14 with 85 applicable hashed
  pins and real nested tools/import/child measurement. Earlier failed/interrupted
  full attempts are retained, not successes. Native wider-than-float64 execution
  and Python 3.8 remain untested; shell-hook coverage is unsupported.
- Metrics: scenarios 33; reviewed cases 287; A/B launches 12/9 complete;
  closed B windows2/2,1/2,1/2,1/2,2/2; covariance window 2/2 accepted/closed;
  initial C 1, repairs 2/2, D1 rejected; one D CLI launch failed before a session
  due incompatible flags and is retained. No fixed execution budget cap supplied.
  Start observation 2026-10-05 16:56:58 UTC. Raw CLI usage/receipts stay outside Git;
  cached input is part of input. Parent/investigator and exact billing unavailable.
- Incomplete backup pointer: freeze the accepted 287-test checkpoint and this task
  state on codex/spectrum-fitting with no open PR. Its exact commit/publication
  result belongs in the conversation. Runtime remains the rejected C2 source;
  two valid regression failures remain. This backup is not submission/readiness.
- Next: obtain the requested repair extension; then fresh C against the frozen
  reviewed checkpoint, exact immutable canonical full, fresh D and required
  Linux/macOS/Windows CI before publishing a ready candidate. Known failures
  block submission. Exact 100% owned statements/branches globally and per file
  and zero failures/errors/skips/exclusions remain required. No PR is open;
  the current branch is incomplete.
