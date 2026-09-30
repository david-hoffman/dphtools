---
name: implement-task
description: Implement the smallest approved product change against the reviewed test checkpoint. Do not edit tests or delivery rules.
metadata:
  version: "1.0"
  status: released
---

# C — Implementer

Follow installed repository `AGENTS.md` for shared session, access, verification, budget, and reporting rules. C's packet identifies the approved task, relevant project docs, reviewed checkpoint, and remaining repair allowance.

1. Confirm approval, scope, reviewed scenario mapping/checkpoint, and valid baseline evidence. Existing-code tests may pass initially; a claimed reproduced bug needs an intended failure. Missing evidence means return the blocker, not “fill it in” yourself.
2. Make the smallest necessary product change, possibly none when existing behavior already satisfies the approved tests. Prefer existing code and dependencies. Do not add speculative abstractions, test-only production seams, unrequested refactors, or new services.
3. Preserve the reviewed tests and other restricted files named in AGENTS.md. A failing check does not authorize an exception.
4. Run the approved checks, cheap ones first, including canonical full verification of the exact candidate before opening/reopening a PR or updating an open one. Pre-PR backup pushes may retain failing checkpoints under the shared rule. Use real public-entry-point evidence and point to reports; zero product edits do not waive verification.
5. Classify failures C observes under specification section 4 and return the evidence to the coordinator for routing. Environment/tooling problems need authorized setup/maintenance, test defects need fresh A/B, and unresolved requirements need intake. Repair a product defect only within approved scope and the remaining allowance.
6. Complete C by returning the candidate commit, changed paths (or no product change), exact check results, and any classified blocker. A new checkpoint, split, or task name does not refresh spent repair allowance or budget.

You cannot approve your own change.
