---
name: test-audit
description: Audit test value when requested, and apply a behavior-first authoring gate to test changes. Preserve meaningful contracts, coverage, and the task's editing limits.
metadata:
  version: "1.0"
  status: released
---

# Test audit

Use the installed delivery roles and owner-approved task scope. This supporting
skill adapts [OpenClaw's test audit](https://github.com/openclaw/openclaw/blob/e09f3ecd491db23f4fdc6e79b4349808d010f394/.agents/skills/test-audit/SKILL.md)
to dphtools. It adds neither permission nor another role. Use `implement-task`
for routine authors and `review-work` for independent reviewers. Specialist A/B
keep their blind inputs: they apply the authoring/value criteria without reading
implementation, source history, or other roles' conversations.

## Authoring gate

Before changing a test, identify its observable contract, a credible regression,
the gap in existing proof, and the real public or process boundary it exercises.
Extend an existing case table when it covers the same contract. Another test
layer needs a distinct risk, such as transport, lifecycle, or platform behavior.
Expected values must come from approved behavior or independent references and
invariants. Never add a production interface solely to make a test possible.

A claimed bug regression needs evidence that valid input failed at the intended
behavior before repair. Existing-behavior and supplemental tests may pass on the
baseline; label them honestly rather than manufacturing red evidence.

## Read-only audit

For an authorized routine audit, inspect each complete candidate test, its real
owner/entry point, callers/callees, overlapping tests, routing, and relevant
history. Inspect applicable dependency source/types when the claim depends on
them. Parallelize independent discovery where useful, then select one coherent
owner-boundary batch. Broad campaigns need their own approved scope.

Look for repeated proof, fixture self-comparisons, implementation-shaped greps,
private-helper replays, coverage-only execution, expectations calculated by the
subject, mocks/fixtures producing the promised result, negative cases rejected
by the wrong guard, and names overstating the exercised behavior.

These patterns are evidence to investigate, not automatic deletion rules.
Retain independent public, platform, structural, packaging, release, and security
contracts; observable ordering; credible regressions; and hard-to-reach numerical
regimes. An externally used library API is not dead because only tests call it
in this repository. An approved exception or successful completion can be the
observable result. Static checks and slow tests can protect meaningful contracts.

Before any deletion or consolidation, record:
- exact test location and the failure it can detect;
- non-test callers and the reason the test/support seam exists in history;
- stronger retained proof, or the approved reason no proof is needed;
- test/support edits enabled, risk, and focused validation command.

Missing evidence leaves the candidate unchanged. A valid failing test remains
visible: classify it under specification section 4. A tests-only task reports a
product defect; it cannot repair production or delete the test to obtain green.

## Repair, review, and completion

Keep repairs within the owner's scope. Remove only demonstrated redundancy,
move retained regressions to their owning boundary, and simplify repeated test
setup without weakening checks or freezing implementation details. Production,
tooling, workflows, test discovery, and coverage policy remain untouched in a
tests-only audit. Instruction edits require their own owner authorization.

Do not edit files while checks of that candidate are running. Run the focused
owner/sibling tests, targeted formatting, and `git diff --check`. Then classify
and verify the exact committed candidate with
`python tools/verification.py pr --base BASE`; retain the selected complete-domain
coverage requirements. A focused run is diagnostic, not full coverage evidence.

Use a fresh independent reviewer after repairs. When the owner authorizes a
review/repair loop, return concrete findings to the author and renew review after
each changed candidate until acceptance or a real blocker. Respect inherited
budgets and specialist allowances; a new session does not reset them. Never let
the repair author approve their own changes.

Report removed categories, retained false positives, focused/required evidence,
test versus production/tooling line changes, unresolved findings, and PR state.
Commit, push, PR creation, merge, and publication still need their applicable
authorization. Use dphtools' commands; OpenClaw's Node runners, linked campaign
and landing skills, and production-deletion preference are not installed here.
