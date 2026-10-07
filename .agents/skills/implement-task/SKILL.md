---
name: implement-task
description: Author routine approved work, or implement specialist product work against its independently reviewed test checkpoint.
metadata:
  version: "1.0"
  status: released
---

# Author or C — Implementer

Follow installed repository `AGENTS.md` for routing, session independence, verification, budget, and reporting rules. The packet identifies approved scope, relevant contracts/project docs, route, requested outcome, and remaining allowances.

## Routine author

Reuse established contracts and meaningful existing tests. Make the smallest authorized change, including tests, docs, or infrastructure within scope. Add meaningful public-entry-point checks where behavior or proof changes; do not manufacture red evidence for existing behavior. Escalate genuine scientific/safety risk to A/B/C/D. Run focused checks and the highest owner-approved project verification tier before opening/reopening or updating a PR; an explicit domain run proves its scope, not PR classification. Unknown/shared/build-input/spanning changes, add/delete/rename, or unavailable/changed base require full. Keep failures visible and return concise command/candidate/environment/results with log pointers. Obtain a fresh independent candidate review before merge. Full local verification is the reference path and is required when the approved risk/check plan calls for it; complete protected platform CI remains the merge gate described in AGENTS.md.

## Specialist C

C's packet additionally identifies the independently reviewed checkpoint and remaining C repair allowance. C must not edit reviewed tests or delivery rules.

1. Confirm approval, scope, reviewed scenario mapping/checkpoint, and valid baseline evidence. Existing-code tests may pass initially; a claimed reproduced bug needs an intended failure. Missing evidence means return the blocker, not “fill it in” yourself.
2. Make the smallest necessary product change, possibly none when existing behavior already satisfies the approved tests. Prefer existing code and dependencies. Do not add speculative abstractions, test-only production seams, unrequested refactors, or new services.
3. Preserve the reviewed tests and other restricted files named in AGENTS.md. A failing check does not authorize an exception.
4. Run the approved checks, cheap ones first. Scientific/safety work normally includes canonical full local reference verification of the exact candidate before submission; record any approved narrower risk/check plan and its limits. Pre-PR backup pushes may retain failing checkpoints under the shared rule. Use real public-entry-point evidence and point to reports; zero product edits do not waive applicable verification. Complete protected Linux/macOS/Windows verification and independent review are still required before merge.
5. Classify failures C observes under specification section 4 and return the evidence to the coordinator for routing. Environment/tooling problems need authorized setup/maintenance, test defects need fresh A/B, and unresolved requirements need intake. Repair a product defect only within approved scope and the remaining allowance.
6. Complete C by returning the candidate commit, changed paths (or no product change), exact check results, and any classified blocker. A new checkpoint, split, or task name does not refresh spent repair allowance or budget.

You cannot approve your own change.
