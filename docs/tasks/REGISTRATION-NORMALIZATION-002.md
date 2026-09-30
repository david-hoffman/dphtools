# REGISTRATION-NORMALIZATION-002: resolve issue #2

**Version 1.0**

## Contract

The [behavioral contract](REGISTRATION-NORMALIZATION-002-contract.md) records five
scenarios within the owner's direct request: “Branching off PR13, solve
https://github.com/david-hoffman/dphtools/issues/2”. This request authorizes the
bug fix and necessary regression checks; no separate approval of this subsequently
written document is claimed. Existing project architecture and public interfaces
remain applicable. Branch `codex/fix-registration-normalization` starts at PR #13's
head, `42e893a410c7fab978da678edc63cf09bde94b3d`.

Allowed changes are issue-specific regression tests, a demonstrated normalization
fix if necessary, relevant API documentation, this task's evidence, and concise
new lesson files. Preserve existing tests and all verification/delivery rules.
The issue reports faulty anisotropic scaling with rotations; root cause and
current-baseline reproduction must be established independently.

## Investigation

The issue was filed against `84ec9f04`, which already used a common isotropic
scale for two-dimensional similarity registration. Later commit `acb291f`
generalized that scale to `self.D` and removed rigid registration's anisotropic
override. Those changes are present in the PR #13 baseline; they do not by
themselves establish resolution of the issue.

For row vectors `x = y B.T + t`, normalized coordinates are
`x_n = (x - tx) Sx` and `y_n = (y - ty) Sy`, with diagonal scale matrices.
Substitution gives `B_n = Sx B Sy^-1` and
`t_n = (ty B.T + t - tx) Sx`. The inverse is
`B = Sx^-1 B_n Sy` and `t = -ty B.T + t_n Sx^-1 + tx`.
The current equations agree with these identities, including noncommuting
rotation and diagonal scale. This is an algebra check, not convergence evidence.

The existing registration suite checks similarity scaling without rotation,
rotation through the correspondence estimator, one affine map, and normalized
rigid rotation. It does not directly check the normalized intermediate mapping.

## Reviewed checkpoint

- Authorization: owner's direct request above; no unresolved behavioral question.
- Reviewed test commit: `7d9a81c087584f77b5625470e0c109fcf520f940`, based on PR #13.
  Test SHA256: `9d00f35e7ab4df29b872ff806f809252a9a6772a0af80a0ed7eb5fc269e6aefb`.
- Fresh root roles: A1 `01a0f2d1-adac-7ea1-a416-ac1ede072336`,
  B1 `01a0f2db-e6ea-78c1-b098-2664552616fa`,
  A2 `01a0f2e0-4d6d-7901-8049-ed33e4e75bfb`,
  B2 `01a0f2e6-6481-7502-affd-4d8efc7f87a5`.
  B1 rejected the unsupported forced-variance convergence expectation; B2 accepted
  the corrected digest and all five scenario mappings. The window is closed.
- Accepted baseline: 12 passed, zero failures/errors/skips, Python 3.13.12.
  B2 independently reran the suite with source-free diagnostics. Black 26.5.1
  passes, and the checkpoint commit's canonical fast hook passes. Existing
  registration/public-boundary tests also passed: 97 tests, 6 existing warnings.
  This is passing regression evidence, not a reproduced current product defect.
- Evidence: [A's methods, mappings, and retained attempts](REGISTRATION-NORMALIZATION-002-test-evidence.md).
  B1/B2 temporary reports are `/private/tmp/dphtools-issue2/B1-summary.txt` and
  `/private/tmp/dphtools-issue2/B2-summary.txt`. Each role reported no accidental
  source exposure. Independence is procedural, not engineered isolation.
- At handoff: scenarios=5; A/B rounds=2/2, accepted; C repairs=0/1.
  No explicit task budget was supplied. A/B CLI usage: 1,289,755 input tokens
  (1,148,416 cached), 37,044 output tokens. Coordinator and dollar usage are
  unmetered. Failed launch/test attempts remain in the evidence; allowances were
  not reset by the correction.

## Current state

The final candidate's own commit, C/D session results, canonical verification
reports and exact coverage, hosted CI results, remaining blockers, next action,
and final metrics are recorded in the delivery conversation or linked pull
request, outside the tracked tree. This pointer is prepared before final
verification so recording its outcome does not change the verified candidate.
