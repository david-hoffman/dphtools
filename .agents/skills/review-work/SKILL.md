---
name: review-work
description: Independently review a routine candidate, specialist tests (B), specialist candidate (D), or delivery docs (doctor).
metadata:
  version: "1.0"
  status: released
---

# Review work

Follow installed repository `AGENTS.md` for routing, shared session, access, verification, budget, and reporting rules. Use one mode per narrow packet. Routine candidate review needs one fresh independent session; A/B/C/D are reserved for genuine scientific or safety risk.

## Tests — B

Use A's approved behavior/public-interface inputs and proposed tests.

Check contract clarity, expected-result sources, and the scenario-to-test mapping in the task's Contract table. Review custom oracle logic, including parsers, observers, units, and allowed outcomes. Check A's tolerance rationale in both directions: legitimate numerical variation must pass and plausible wrong values must fail. Do not normalize malformed output into validity or impose an undocumented choice.

Reject tests that merely execute code to raise coverage, negative cases that pass through an unrelated rejection, fixtures that supply the purported production result, and names claiming unexercised behavior. Require a meaningful correctness check, not a particular assertion syntax. Behavior-preserving refactors should not break tests unless the supposed refactor violates an independently approved interface/structural contract.

Complete B with acceptance of an identified test revision/mapping or concrete defects/contract gaps. After the second nonacceptance in a review window, diagnose the blocker: send a specific ambiguity and recommended clarification to intake, or propose genuine end-to-end split points for excessive scope. Confirm the diagnosed problem is resolved before restarting rounds under specification section 9; otherwise report the blocker. Do not add another reviewer or model-escalation path.

Accepted tests need valid baseline evidence and a recorded checkpoint before C. Existing-code tests may pass initially without mutation testing or manufactured failure. Classify observed failures under specification section 4; a setup failure is not product-red evidence.

## Candidate — routine reviewer or specialist D

Read approved scope/established contracts, the exact candidate, and actual applicable check evidence, not the author's conversation. Specialist D also receives the reviewed test checkpoint. Form findings before consulting current-task implementation lessons.

Inspect behavior, security, public boundaries, and test/workflow changes. Compare the tested commit with the proposed merge. Missing/skipped/incomplete results are not success. Check the real product result, not only a summary or compilation. Flag production seams created solely for tests; investigate test-only local callers without treating externally used public APIs as dead code.

Include simplification in this review: unnecessary wrappers, duplication, dependencies, and speculative features. Stay within the task. No extra simplification agent. Complete review with acceptance of the exact candidate under the applicable checks or concrete findings and evidence; identify pending platform CI rather than calling it passed. Do not fix the candidate and then approve it. Route findings under specification section 4. Routine corrections return to the authorized author and need renewed independent review; specialist corrections use fresh C within the recorded allowance and fresh D. Reuse applicable evidence; rerun when changed inputs or unresolved findings justify it. Confirm the selected plan's exact base/candidate, conservative tier, complete whole-owned-domain/test closure, and Linux/macOS/Windows results before merge; scoped results establish only declared domains, while full retains fresh global/per-package 100% coverage. Confirm verified `ci-required` protection; workflow text or local success cannot establish that gate.

## Doctor

Use [DOCTOR-PROMPT.md](../../../docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md) and specification section 8. This is the explicit documentation-editing mode, not a product reviewer secretly changing the rules.

Inspect recent lessons, task metrics, and evidence. Complete with an evidenced patch to the existing instructions or a report of no supported gap; `--check` writes nothing. Scenario counts and outcomes may support an owner-approved change to default slice size, never weaker acceptance thresholds. Follow the dedicated prompt's edit/approval limits and keep Version 1.0.
