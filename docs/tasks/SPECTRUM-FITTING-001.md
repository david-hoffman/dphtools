# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Public contract R4 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This scientific task is separate from the routine-maintenance ten-task pilot.

## Current state

- Approval: owner approved R3 and the 33-scenario slice on 2026-10-05. R4 records
  the instruction to ground numerical acceptance in physical plausibility.
  Heights, positive amplitudes/widths, free centers, actual Levenberg–Marquardt
  (LM) default and physical custom-optimizer protocol remain approved. Preserve
  existing interfaces. Commit/publish `codex/spectrum-fitting` is authorized;
  merge/release is not.
- Route/pointer: specialist A/B/C/D; baseline
  `a835a490373835fa01e4460ce2159365150d1946`. Prepare this task pointer before
  final verification. The final candidate's own identity and actual local,
  independent-review and platform results belong outside its tracked tree.
- Published candidate: `f63ebf2de3fc65c047b00a3631f84b8bf0c68437`, with unchanged
  C2 runtime `820e17dc31d9121b7b1ff2f5c839c0e77d8ad70f`. Normal commit/push fast
  checks passed and GitHub readback matched. Its exact canonical full passed
  2118 tests, 186 retained warnings, 3333/3333 statements and 856/856 branches
  across 20 owned files, with zero failures/errors/skips/exclusions. Actual
  receipt seals, reports, inputs, logs and unchanged candidate were checked.
- D2 rejected that candidate. The same representative R4 voltage spectrum
  expressed in metres makes default LM report success at unfinished parameters:
  residual sum of squares is 1.19075 V² against the feasible generating model's
  0.0172294 V². Nanometre/default and metre/real-custom fits reach 0.0170722 V².
  The default centre derivative step is 14.9 nm, larger than the 0.85 nm initial
  width. This is an in-scope physical-unit optimization defect, not a covariance
  formula error. Lorentz/Voigt and modest linear backgrounds also fail in metres.
  Existing interfaces and simplicity passed review.
- Reviewed checkpoint: fresh A14/A15 and B11 accepted 296 cases across the
  unchanged 33 scenarios, round 1/2; the physical-unit window is closed.
  Tests SHA-256
  `b3fe64dc5da2bdb40e4af19832f39684507b77957ac7361dc789431af09069d0`;
  report SHA-256
  `f345641c2d7697ff064ac73c59ec0478f216183b469243331e0791211db6ee0d`.
  All 287 previously accepted cases and their report prefix remain byte-exact.
  Nine new nm↔m cases cover the three profiles, actual default/real custom
  optimizers, constant/linear backgrounds, independent physical fit quality,
  full covariance and unit conversion. Baseline: 290 passed, six valid
  default-metre product failures, exit 1, no skips/expected failures. Five fits
  return unfinished parameters; Gaussian with a line raises RuntimeError.
  Mathematical controls and Black 99 pass. Failed metre cases leave later
  covariance/invariance/storage outcomes unproved. This revision is an
  explicitly incomplete backup checkpoint, not a passing product candidate.
- CI run 37381489582 attempt 1 for the published candidate completed/cancelled
  after D2. Three collections succeeded; six shards cancelled. Aggregate and
  `ci-required` failures followed cancellation. All 12 partial artifacts and
  actual logs/metadata are retained. No platform success is claimed. Earlier
  cancelled run 37370536644 remains historical evidence.
- Historical scope: the artificial 1e12-offset covariance defect remains
  documented. The owner excluded it from R4 acceptance; that repair request is
  superseded and no such repair occurred. Ordinary physical-unit behavior is
  still required. Neither scope correction erased attempts or spending.
- Allowances: initial C and both authorized repairs are complete; repairs 2/2
  used. One additional physical-coordinate-scaling repair is requested/pending.
  No third C is authorized or performed. Prior B windows 2/2, 1/2, 1/2, 1/2,
  2/2, 2/2, 1/2, 1/2 remain closed and counted. Changed tests need fresh B acceptance
  and an immutable revised checkpoint before C.
- Next: obtain the pending physical-unit repair extension; then fresh C against
  this accepted checkpoint, exact unchanged canonical full, fresh D and complete
  Linux/macOS/Windows
  Python 3.10 verification. Require exact 100% owned statements/branches globally
  and per file and zero failures/errors/skips/exclusions. The published branch
  is incomplete despite its earlier passing suite. No PR is open.
- Environment/limits: disposable non-Conda CPython 3.12.14 with applicable hashed
  pins and real nested tools/imports/child measurement. Python 3.8 and native
  wider-than-float64 remain unverified; shell-hook coverage is unsupported.
  Earlier failed/interrupted/rejected attempts and actual evidence are retained.
- Metrics: 33 scenarios; 296 accepted test cases; baseline 290 pass / six product
  failures. A/B completed launches 15/11; physical-unit window accepted/closed
  at 1/2; initial C one; repairs 2/2; D1/D2 rejected. No fixed execution budget
  cap. Start observation 2026-10-05 16:56:58 UTC. Actual role usage/logs remain
  outside Git; cached input is part of input and all prior spending is retained.
