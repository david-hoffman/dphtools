# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Contract revision R2 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This is a new independent product task, not an extension of prior scientific
repair allowances. The existing project architecture and dependencies apply.

## Current state

- Status: intake; draft R2 pending approval of remaining numerical conventions,
  output schema, failure behavior, and the 29-scenario size exception.
- Authorization: owner requested spectrum fitting on 2026-10-05 and confirmed
  positive peaks with optional constant/linear background, both center/full
  guesses, optional x, joint least squares, and the requested result contents.
  Amplitude means component peak height, not integrated area, per the owner's
  correction. The owner authorized committing and publishing this contract
  branch. These settled requirements need no further approval round.
- Scope/route: one additive spectrum-fitting interface; specialist A/B/C/D for
  the new scientific contract. The spectrum task does not count as a routine
  maintenance sample in DELIVERY-EFFICIENCY-001.
- Baseline: a835a490373835fa01e4460ce2159365150d1946; isolated managed worktree
  spectrum-fitting, branch codex/spectrum-fitting. Prior task worktrees are
  preserved. Post-merge CI run 37329170472 is complete on all three platforms;
  retained baseline evidence is 1,831 passing tests per platform with exact
  3,152 statements/778 branches and zero failures/errors/skips/exclusions.
- Candidate/checkpoint: no implementation or acceptance tests written. Future
  exact candidate/check evidence belongs in the conversation or linked PR.
- Metrics: scenarios=29 proposed; coordinator/reference-investigation launches=1/1;
  A/B rounds=0/2; C repairs=0/1 after initial C; execution budget=not specified;
  task-start observation=2026-10-05 16:56:58 UTC; model tokens/billing=unavailable.
- Blocker: owner decision on draft R2's remaining choices and size exception.
- Next: once R2 is approved, issue narrow fresh-root A then B packets, record the
  reviewed checkpoint, and route C and D sequentially. Record wall time, launches,
  receipts, required CI runner usage, and detected defects for this scientific
  task without relabeling it as routine work.
