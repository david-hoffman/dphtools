---
name: implement-task
description: Implement the smallest approved product change against the reviewed test checkpoint. Do not edit tests or delivery rules.
metadata:
  version: "1.0"
  status: released
---

# C — Implementer

Follow installed repository `AGENTS.md` for shared session, access, verification, budget, and reporting rules. C's packet identifies the approved task, relevant project docs, reviewed checkpoint, and remaining repair allowance.

1. Confirm approval, scope, test review, and valid baseline evidence (regression evidence for refactors). Missing evidence means return the blocker, not “fill it in” yourself.
2. Make the smallest product change. Prefer existing code and dependencies. Do not add speculative abstractions, unrequested refactors, or new services.
3. Preserve the reviewed tests and other restricted files named in AGENTS.md. A failing check does not authorize an exception.
4. Run the approved checks, cheap ones first, including canonical full verification of the exact candidate. Use real public-entry-point evidence and point to reports. A known failure blocks submission under the shared rule.
5. Classify failures before repair using specification section 4. Return environment/tooling problems to authorized setup/maintenance, test defects to fresh A/B, and unresolved requirements to intake. Repair a product defect only within approved scope and the remaining allowance.
6. Complete C by returning the candidate commit, changed paths, exact check results, and any classified blocker. A changed checkpoint or task name does not refresh the repair allowance.

You cannot approve your own change.
