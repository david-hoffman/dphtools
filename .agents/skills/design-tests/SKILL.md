---
name: design-tests
description: For scientific or safety risk, write independent end-to-end-first tests from an approved contract without inspecting implementation.
metadata:
  version: "1.0"
  status: released
---

# A — Test author

Use this specialist role for genuine scientific or safety risk under installed repository `AGENTS.md`; routine authors reuse established contracts and write authorized tests without a separate A session. Follow AGENTS.md for shared session, access, verification, budget, and reporting rules. A's packet contains only approved behavior, public interfaces, approved fixtures, and test conventions.

1. Map the slice's approved scenarios to tests, supplying references for the task's Contract scenario table. Keep one reviewed checkpoint lineage per task; material new behavior returns to intake.
2. Prefer the real product entry point: browser journey, executable, service endpoint, or library API. Exercise owned components together; isolate test data and uncontrollable external services. Do not design production exports/hooks used only by tests.
3. Trace each expected result to approved behavior, an applicable primary reference, or a mathematical invariant. Unspecified units, conventions, and estimators return to intake. Choose and explain absolute/relative numerical tolerances from permitted algorithm descriptions, references, conditioning, and precision; never from implementation reads or tuning to candidate output. B reviews the rationale and discriminating examples.
4. Use smaller integration/unit tests only for a meaningful gap. Do not duplicate E2E coverage just to create more tests. Keep helpers simple.
5. Run applicable checks and classify observed failures under specification section 4. Return tests, scenario/expectation-source mapping, tolerance rationale, baseline evidence, and contract limits. Initially passing existing-code tests are valid; do not fabricate red evidence or mutate production code. This handoff or a specific contract blocker completes A.
6. Return later test defects/coverage gaps through fresh A/B work within the recorded allowances. After two B nonacceptances, wait for B's diagnosis and the documented disposition before another rewrite. Do not call a post-implementation test original test-first evidence.

You may write authorized tests, not product logic or workflows. Do not weaken existing behavior to fit a candidate. A supported invariant can validate shape or ratios while absolute units remain unresolved; disclose that limit.
