# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Contract revision R3 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This is a new independent product task, not an extension of prior scientific
repair allowances. The existing project architecture and dependencies apply.

## Current state

- Status: owner approved R3 and all 33 scenarios. B accepted the original
  behavioral tests, the later Black-only correction, and the later public
  coverage correction. Final candidate identity, role/check outcomes, blockers,
  and completion evidence are maintained in this task's conversation outside the
  tracked candidate. No verified commit will be edited to record its own results.
- Authorization: approval on 2026-10-05 covers component peak-height amplitudes,
  positive amplitudes/widths, free centers, default Levenberg-Marquardt, and a
  physical custom-optimizer protocol. New helpers may supplement deficient
  existing fitters; existing public interfaces stay intact. Committing/publishing
  this branch is authorized; merge/release is not.
- Scope/route: one new scientific interface, specialist A/B/C/D. This task remains
  separate from DELIVERY-EFFICIENCY-001's routine-maintenance pilot. Runtime
  baseline a835a490373835fa01e4460ce2159365150d1946; isolated spectrum-fitting
  worktree, branch codex/spectrum-fitting.
- Reviewed checkpoint: 9e3962dcfaa2fc60beb67803446e642b5b87ff17. Tests SHA-256
  1ca6c122fa10d628af4e4c00f6c19b3aef948bbff38bcabec147397061bac764;
  mapping/report SHA-256
  a6dd1d690176ddc159dbd2479f8a024fcd4c86eda78602dfaf2bfc53f341aec3.
  All original test/oracle/tolerance AST nodes and historical report bytes are
  preserved. The [historical A report](SPECTRUM-FITTING-001-tests.md) retains all
  attempts, mappings, tolerance evidence, and post-implementation correction.
- Review lineage: B1 rejected S10/S13/S23 for unrelated-rejection paths; fresh A
  corrected them and B2 accepted. A later fast format failure opened an evidenced
  correction window; fresh A3 applied Black only and B3 accepted exact AST
  equality at checkpoint 1f9050ccd1d7e6b0ae86fd9eabc3600fcf29a268. Initial full
  measurement then exposed untested approved boundaries; fresh A4 added ten
  meaningful public cases and B4 independently accepted conditioning, tolerance
  discrimination, singular uncertainty, and both allowed numerical-boundary
  outcomes. Original=2/2 accepted/closed; format=1/2 accepted/closed;
  coverage=1/2 accepted/closed. No allowance or spending was reset.
- Implementation/evidence: initial C committed only the additive fitter/export
  in dd31c2c237ac119d1ba1a3008f6b16e7e48b41f8. Its 299 focused cases and fast
  passed. Canonical full first encountered audit DNS failure; the hashed audit
  passed with network access. The unchanged full rerun returned 2,076 passed and
  18 failed, with incomplete coverage. Twelve failures inherited Black 23.3.0
  rather than locked 26.5.1; six failed runtime/cache eligibility. These were
  environment/tooling defects, not product-red evidence. The new fitter's four
  unmeasured statements/four branches supplied the later test-correction reason.
- Baseline: original 263 cases returned 1 failure/262 errors because spectrum_fit
  was absent. This was feature-absence evidence only. Later added tests are
  explicitly post-implementation evidence; no red result was manufactured.
- Repaired environment: disposable non-Conda CPython 3.12.14 at
  /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3. All 85
  applicable hashed-lock pins match. Real nested system-site inheritance and
  wheel installation/import passed; actual child instrumentation works. All 59
  verification-reuse tests passed, measuring that module's 78/78 statements and
  46/46 branches. Original environment/receipts remain preserved.
- Focused revised evidence: 273 spectrum tests and ten independent B probes
  passed on clean Python 3.12.14. A separate 309-case diagnostic passed all 273
  spectrum and 36 old-fitter cases, measuring the new helper's 155/155 statements
  and 66/66 branches with zero exclusions. Its intentionally partial global
  diagnostic failed the unchanged 100% threshold; it is not full success. All
  A/B sessions reported no source exposure. The nonblind environment investigator
  disclosed one source line from a warning; it was not provided to A/B.
- Metrics: scenarios=33; cases=273; A/B windows=prior2/2, format1/2,
  coverage1/2 accepted/closed; A/B launches=4/4; initial C=1, D=0;
  C repairs=0/1; coordinator/reference/environment-investigation launches=1/1/1;
  execution budget=no fixed cap supplied; task-start observation=2026-10-05
  16:56:58 UTC. CLI usage/raw receipts are retained outside Git; parent/billing
  usage is unavailable. Prior attempts/allowances remain preserved.
- Blockers/next: canonical full on the immutable revised candidate with the
  repaired environment, then fresh D and required Linux/macOS/Windows CI. Exact
  candidate, commands, check/role outcomes, remaining allowances, and disposition
  are recorded in this conversation. This pointer is prepared before final
  verification, rather than editing the verified commit afterward.
