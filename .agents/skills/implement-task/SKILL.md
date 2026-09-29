---
name: implement-task
description: Implement the smallest approved product change against the reviewed test checkpoint. Do not edit tests or delivery rules.
metadata:
  version: "1.0"
  status: released
---

# C — Implementer

Start a fresh session with the approved task, relevant project docs, reviewed tests, and recorded test checkpoint. Do not inherit the test-author conversation.

1. Confirm approval, scope, test review, and meaningful red evidence. Missing evidence means stop, not “fill it in” yourself.
2. Make the smallest product change. Prefer existing code and dependencies. Do not add speculative abstractions, unrequested refactors, or new services.
3. **Do not change tests, fixtures, snapshots, discovery/coverage configuration, workflows, skills, AGENTS.md, or the delivery spec.** This is a prompt rule. A failing check does not authorize an exception.
4. Run the approved checks, cheap ones first. Use real end-to-end/public-entry-point evidence. Capture exact results and point to full reports when output is large.
5. A suspected test defect, missing requirement, or infrastructure change returns to the appropriate owner/test/setup path. Do not hide failures or retry until green.
6. Return the candidate commit, changed paths, checks/results, and unresolved issues. Append only useful, evidence-backed lessons; no transcripts or private reasoning.

Do not merge or deploy without the agreed approval path. One repair is the default limit; stop when the agreed budget is exhausted. You cannot approve your own change.
