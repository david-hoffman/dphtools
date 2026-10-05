# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Contract revision R3 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This is a new independent product task, not an extension of prior scientific
repair allowances. The existing project architecture and dependencies apply.

## Current state

- Approval/scope: owner approved R3 and33 scenarios as one scientific slice on
  2026-10-05: height amplitudes, positive amplitudes/widths, free centers, actual
  Levenberg-Marquardt default and physical custom optimizer. Preserve existing
  interfaces. Committing/publishing codex/spectrum-fitting is authorized;
  merge/release is not. Runtime baseline a835a490373835fa01e4460ce2159365150d1946.
  This scientific task is separate from the routine-maintenance ten-task pilot.
- Final evidence pointer: exact final candidate identity, check commands/results,
  D/platform outcomes and disposition are recorded in this task's conversation,
  outside its tracked tree. This pointer is prepared before verification; no
  verified commit will be edited to record its own results.
- Reviewed checkpoint: b62450a206b6f3592c6b8aab712d34fb42f7f08f,285cases. Tests
  SHA-256 50bb168312355b5c3035f534195260f4a0081a6f171cdd6a6de52235774837dd;
  historical report SHA-256
  73bf5276ddf3a315544f37627701615ab9651b9f967f9aebe96c8bb0b528c55c.
  All276 previously accepted cases/oracles/tolerances remain preserved. The
  [historical report](SPECTRUM-FITTING-001-tests.md) retains mappings, mathematical
  proofs, failed attempts, source-exposure disclosures and spending.
- Review lineage: original B2/2, format1/2, coverage1/2, numerical1/2 and final
  post-repair2/2 windows accepted/closed. B6 rejected undocumented stacked-SVD
  and output-dtype restrictions; fresh A9 corrected them. An independently
  reproduced complex-SVD observer restriction prompted supplementary A10 before
  B7. B7 accepted exact revisions after46 independent checks. No rounds reset.
- Product/evidence: initial C dd31c2c237ac119d1ba1a3008f6b16e7e48b41f8;
  first repair8ae5438c09fa80559882ba0e3834f13756015851 fixes covariance units and
  representability failures. Its348 focused cases and fast passed. Latest285-case
  spectrum result is284passed/1failed; B independently confirms a custom optimizer
  numerical LinAlgError escaping instead of RuntimeError as valid product-red.
  A357-case diagnostic is356passed/1failed; helper178/179statements and78/78branches,
  zero exclusions. This partial diagnostic is not canonical full success.
- Allowance: initial C and original repair were used. On2026-10-05 the owner
  explicitly approved ONE additional C repair for custom numerical-exception
  normalization. Repairs used1/2 before fresh C2; prior attempts/spending retained.
  Further product repair after C2 needs another explicit extension.
- Environment/evidence limits: selected disposable non-Conda CPython3.12.14 is
  /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3. All85
  applicable hashed pins, real nested tools/wheel import and child instrumentation
  passed; all59 verification-reuse cases passed. Original full2076passed/18failed
  was classified as environment/tooling with incomplete coverage. Later clean
  full on c7ee4099e5276c9218a2d5bd90891ba7f017c6c2 was interrupted after numerical
  defects were discovered, exit130; neither establishes readiness. ARM lacks a
  wider-than-float64 dtype here; higher-range observer acceptance has mathematical
  and dtype-instrumentation evidence, not native wider-range execution.
- Metrics: scenarios33; cases285; A/B launches10/7; B windows2/2,1/2,1/2,1/2,2/2
  accepted/closed; initial C1, repair1/2 before C2, D0; coordinator/reference/
  environment-investigation launches1/1/1; no fixed execution budget cap supplied.
  Task-start observation2026-10-05 16:56:58 UTC. CLI usage/raw receipts remain
  outside Git; cached input is a subset of input. Parent and billing usage are
  unavailable. Final role/check counts are maintained in the conversation.
- Next: fresh C2 repairs only the validated custom numerical-exception boundary
  against the frozen tests, then coordinator canonical full on its immutable
  commit, fresh D and required Linux/macOS/Windows CI, and authorized publication.
  Exact100% global/per-file owned statements/branches and no failures/skips/
  exclusions remain required. No submission occurs with known failed checks.
