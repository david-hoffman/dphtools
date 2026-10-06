# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Public contract R4 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This scientific task is separate from the routine-maintenance ten-task pilot.

## Current state

- Approval: R3 and the 33-scenario exception remain approved. R4 records physical
  plausibility: resolved spectra, plausible noise and reasonable backgrounds.
  Peak heights, positive heights/widths, free centers, actual Levenberg–Marquardt
  (LM) default, physical custom-optimizer protocol and preserved existing public
  interfaces remain required. The owner requested saving plots in docs and then
  opening a PR, following the pending one-additional-test-review-round request.
  That authorizes the proposed one-round extension and PR work. Publish
  `codex/spectrum-fitting` and open the PR against `codex-main` after required
  checks pass. Merge/release remains unauthorized.
- Candidate pointer: prepare this record before exact final verification. C3
  runtime is `f3a655709920f6c7f13a392c5c6265ae8d2f045e`; only the additive
  spectrum helper changed. Private center/width coordinates and line slope use
  the observed x span. Intensities and physical outputs/covariance stay unchanged.
  Final candidate identity and actual full/D/platform/PR results belong outside
  its tracked tree, in the conversation or linked PR. No successful final full,
  D3 or platform result is asserted by this preparation record.
- Reviewed checkpoint: fresh A18/B14 accepted 297 cases across unchanged
  33 scenarios, round 3/3 of the SAME owner-extended correction window; closed.
  Tests SHA-256 `22083617a37363f398116749087c4b0a8f080cc3cb27e4ebcaaeb0093342bf35`;
  report SHA-256 `b376f8eb13184564ec3836af0c818b65c011b3f5bf385a9f587cfa41f4d8e5ea`.
  The entire accepted 296-case test prefix (84331 bytes, 91 AST nodes) and report
  prefix (199983 bytes, hash-only) remain exact. A16/B12 and A17/B13 rejected
  rounds remain retained. B14 accepted the corrected warning-preservation
  checks and independently assessed the actual covariance-inability RuntimeError
  as informative. Class/nonempty error text alone is not informative acceptance;
  D must independently assess the actual numerical context too.
- Checks: corrected case, all 297 spectrum cases, Black 99, 54 author and
  19 independent reviewer controls passed. Focused spectrum-helper measurement
  is 186/186 statements and 80/80 branches, zero exclusions. This is focused
  evidence only. C3's earlier exact full passed all 2127 tests with 186 warnings
  but FAILED coverage at 3338/3339 statements and 857/858 branches. The one
  missing covariance inability is now exercised; global/full success must still
  be verified on the final candidate. Actual failed receipt seals, logs, raw
  coverage, JUnit, lifecycle records, artifacts and input identities are retained.
- Human plots: [saved examples](../examples/spectrum-fitting/README.md) contain
  eight scientific PNGs, a portable HTML page, compressed simulated/fitted data
  and a hash/provenance manifest, 12 files total. All image/data bytes match the
  original after-repair review; 21 relative links and 108 arrays were checked.
  The 18 original real fits returned with zero numerical warnings; residual RMS
  was 0.00915723–0.00968070 V for 0.01 V read noise. Default nm/m fitted-signal
  difference was at most 4.71755e-10 V. Images and exact table cells were
  inspected; browser CSS rendering remains unverified. These figures preserve
  the C3 runtime results from preparation 1c72cad, before tests/docs-only changes;
  they do not replace numerical acceptance or claim execution on later commits.
- History: published backup `8351a7dddfa9e593a189de0ad251229de7655287`
  remains incomplete. D2 rejected the earlier default metre fits; C3 passes the
  accepted physical-unit regression cases. Cancelled CI runs 37381489582 and
  37370536644, partial artifacts/logs, the initially misrouted covariance witness,
  rejected test appendices and reviewer-harness failures remain retained.
  The artificial 1e12-offset accuracy example stays outside R4; no such repair
  occurred. The ten-task maintenance pilot is separate and remains incomplete.
- Next: commit the accepted tests/plots/record, run canonical exact full,
  obtain fresh D3, publish the exact passing candidate and open the requested PR.
  Require exact 100% owned statements/branches globally and per file, with zero
  failures/errors/skips/exclusions. Complete the actual PR's Python 3.10
  Linux/macOS/Windows matrix and retain all platform evidence. GitHub readiness
  confirms unchanged `codex-main` base a835a490 and required strict `ci-required`.
  No PR was open at preparation; no merge/release is authorized.
- Environment/limits: disposable non-Conda CPython 3.12.14 with applicable hashed
  pins and real nested tools/imports/child measurement. Python 3.8 and native
  wider-than-float64 remain unverified; shell-hook coverage is unsupported.
  Role reports, failed attempts, usage and final results remain outside Git.
- Metrics through B14: 33 scenarios; 297 accepted cases; A/B launches 18/14;
  current correction window accepted/closed 3/3. Earlier closed windows
  2/2, 1/2, 1/2, 1/2, 2/2, 2/2, 1/2, 1/2 stay counted. Initial C one;
  repairs 3/3 spent, C3 completed; D1/D2 rejected. No fixed execution cap.
  Start observation 2026-10-05 16:56:58 UTC. Completed CLI roles consumed
  46,836,797 input tokens, including 43,884,416 cached;
  735,905 output, including 280,724 reasoning.
  Parent/helper and exact billing metering is unavailable. No allowance or
  spending resets; no further test rewrite or product repair is authorized.
