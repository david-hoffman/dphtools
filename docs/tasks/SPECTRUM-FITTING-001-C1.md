# SPECTRUM-FITTING-001 — C1 repair packet

Role: fresh root specialist C, .agents/skills/implement-task/SKILL.md.
Owner approved contract R3 and S01-S33 on 2026-10-05, including 33-scenario
exception, height amplitudes, positive amplitudes/widths and freely moving centers.
Read the contract, root AGENTS.md, docs/PROJECT.md, approved public tests/report.
No previous role transcripts or private reasoning. No Superpowers, optional
memory or delegation. Do not use the historical initial-C packet as current.

Accepted checkpoint: f70661a9ed1813312e5bc409fae8a288c8c3d025.
Tests SHA-256: c5e51bab70912a674d7242541591d1844939dca981c5b78cbce9de70617accca.
Report SHA-256: 2bd7819d6c6112d2a6660fd03cd5f71685613565f11081fa8b340f3647a8ef3c.
B5 independently accepted the latest three numerical regressions in round1/2;
all 273 previously accepted cases and historical report bytes are unchanged.
Original, format, coverage and numerical B windows are accepted/closed.
Initial product: dd31c2c237ac119d1ba1a3008f6b16e7e48b41f8. Baseline runtime:
a835a490373835fa01e4460ce2159365150d1946. Current task pointer adds no product.

Valid product-red: current spectrum file has 273 passed, 3 failed, no errors/skips.
Two S29 cases leak LinAlgError for the smallest finite positive Gauss/Lorentz
widths; numerical failure must be RuntimeError or legitimate validated success.
One S09/S31 case proves an identifiable stationary minimizer and independently
checks local physical covariance and unit transformation C'=D C D. Its covariance
fails after converting x/centers/widths by 1e-6. Repair numerical covariance/failure
handling generally. Do not change the contract or impose arbitrary lower-width
cutoffs. A separate nine-sample maximum-finite-height exact-minimizer custom
probe stalled after a covariance overflow warning and was terminated. This is
incomplete diagnostic evidence for the same numerical boundary, not an extra
approved timeout oracle. Keep diagnostics; avoid feeding nonfinite numerical
work into routines that cannot handle it. No fixture-specific implementation.

Allowed edits: additive spectrum fitter/helper/export and docstrings only,
under dphtools/utils. Preserve all older public signatures/behavior. Never edit
reviewed tests/fixtures/snapshots, workflows, measurement/discovery/type settings,
dependencies, skills, task records or delivery rules. No test-only seams. No
push/PR/merge/release. This is the ONE remaining C repair: usage becomes1/1.
No new budget cap was supplied; preserve all attempts/spending and report usage.

Clean selected Python: /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3
CPython3.12.14, all85 applicable hashed pins. Real nested system-site inheritance,
wheel import, subprocess coverage, and all59 verification-reuse tests passed.
Set MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig and
PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache.
Never install into the shared primary .venv. Preserve the repaired environment.

Run meaningful focused spectrum plus existing fitter regressions, then canonical
fast. Commit only authorized product files with normal hooks using
DPHTOOLS_PYTHON=<clean selected Python>. Return promptly after the immutable
commit and focused/fast evidence. The COORDINATOR will run canonical full on
that exact commit, retain full receipts, and arrange fresh D/platform CI. This
allocation does not waive full verification or the exact100% global/per-file
owned statement/branch gate before submission. Do not repeatedly run expensive
full while a focused known failure blocks it. Do not claim full/D/CI success.

Classify any failed checks before routing. Test defects/coverage gaps go to the
coordinator; do not edit them. No further C product repair is authorized after
this one without owner extension. Final report: candidate commit, changed paths,
exact commands/results, unchanged test/report hashes, classified blockers and
remaining full/D/CI work. You cannot approve your own repair.
