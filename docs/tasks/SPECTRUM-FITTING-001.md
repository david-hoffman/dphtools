# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Contract revision R3 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This is a new independent product task, not an extension of prior scientific
repair allowances. The existing project architecture and dependencies apply.

## Current state

- Status: R3 and all 33 scenarios approved; B2 accepted the behavioral tests,
  and B3 accepted the later formatting-only correction with identical test AST.
  Original window=2/2 accepted/closed; correction window=1/2 accepted/closed.
  All prior attempts/spending remain retained. Final candidate identity,
  role/check outcomes, remaining blockers, and completion evidence are maintained
  in this task's conversation outside the tracked candidate; no verified commit
  will be edited to record its own hash/results.
- Authorization: owner approved R3 and the 33-scenario exception on 2026-10-05,
  including component peak-height amplitudes, positive amplitudes/widths, free
  centers, default Levenberg-Marquardt, and a custom optimizer. New helpers may
  supplement deficient existing fitters; existing public interfaces stay intact.
  Committing/publishing this branch is authorized; merge/release is not.
- Scope/route: one new scientific interface, specialist A/B/C/D. This task is
  recorded separately from DELIVERY-EFFICIENCY-001's routine-maintenance pilot.
- Baseline: runtime a835a490373835fa01e4460ce2159365150d1946; isolated worktree
  spectrum-fitting, branch codex/spectrum-fitting. Existing project architecture
  and locked NumPy/SciPy dependencies apply.
- Reviewed checkpoint: tests SHA-256
  4f762faf63839e4301038a91d02abee9b2f6c8748dc671d19a48aec1ffb674d2;
  mapping/tolerance report SHA-256
  0f9e2b90b7f6d96a016073bc2dd0807c3888a9d257f781c568d097f221d23674.
  B1 rejected S10/S13/S23 for unrelated-rejection paths; fresh A corrected them,
  and B2 independently accepted all scenarios, oracles, and tolerances. The
  subsequent fast format failure was classified as a test formatting defect;
  fresh A3 applied Black only and B3 independently proved identical AST and
  accepted the revised checkpoint. No contract, oracle, or tolerance changed. The
  [historical A report](SPECTRUM-FITTING-001-tests.md) preserves attempts/evidence.
  No product implementation was read in A/B; all roles reported no source exposure.
- Baseline evidence: 263 collected cases, 1 failure/262 errors, exit 1, because
  spectrum_fit is absent. This is feature-absence evidence only; numerical and
  validation assertions have not yet run against an implementation. Classification:
  product defect against the new approved API, not a reproduced existing bug.
- Environment/check plan: isolated copy of the locked Python 3.13.12 environment
  at /private/tmp/dphtools-spectrum-delivery/venv; preflight passed including the
  nested installation probe. C runs focused checks, fast, and canonical full.
  Complete required platform CI and fresh D remain prerequisites to acceptance.
  Exact candidate/report/check evidence belongs in the conversation.
- Metrics: scenarios=33; A/B rounds=prior 2/2 accepted, format 1/2 accepted; A/B launches=3/3;
  coordinator/reference/environment-investigation launches=1/1/1; initial C and D
  pending; C repairs=0/1 after initial C; execution budget=no fixed cap supplied;
  task-start observation=2026-10-05 16:56:58 UTC; CLI token usage retained outside
  Git, parent/billing usage unavailable. Prior attempts/allowances are preserved.
- Blockers/next: no owner or test-review blocker remains. Freeze the revised
  reviewed checkpoint, start fresh C with one later repair allowance, then fresh
  D and required platform checks. The conversation records final disposition.
