---
name: design-tests
description: For scientific or safety risk, write independent end-to-end-first tests from an approved contract without inspecting implementation.
metadata:
  version: "1.0"
  status: released
---

# A — Test author

Use this specialist role for genuine scientific or safety risk under installed repository `AGENTS.md`; routine authors reuse established contracts and write authorized tests without a separate A session. Follow AGENTS.md for shared session, access, verification, budget, and reporting rules. A's packet contains only approved behavior, public interfaces, approved fixtures, test conventions, and the approved project coverage policy with relevant risks. A missing coverage choice returns to intake before test design; do not silently impose or remove a numerical gate.

1. Map the slice's approved scenarios to tests, supplying references for the task's Contract scenario table. Identify relevant high-risk paths and how the tests detect plausible wrong outcomes; record the short risk mapping for B's independent omission audit. Keep one reviewed checkpoint lineage per task; material new behavior returns to intake.
2. Prefer the real product entry point: browser journey, executable, service endpoint, or library API. Exercise owned components together; isolate test data and uncontrollable external services. Do not design production exports/hooks used only by tests.
3. Trace each expected result to approved behavior, an applicable primary reference, or a mathematical invariant. Unspecified units, conventions, and estimators return to intake. Choose and explain absolute/relative numerical tolerances from permitted algorithm descriptions, references, conditioning, and precision; never from implementation reads or tuning to candidate output. B reviews the rationale and discriminating examples.
4. Use smaller integration/unit tests only for a meaningful gap, including reachable failures through suitable boundaries. Do not duplicate E2E assertions, force impossible internal states, or invent rejection rules solely for a percentage. Keep helpers simple. Behavior coverage and risk review always apply; only explicitly selected metrics add numerical gates. Blind authorship cannot certify every branch of a future implementation.
5. Run applicable checks and classify observed failures under specification sections 4 and 5.2. Return tests, scenario/expectation-source and risk mapping, tolerance rationale, baseline evidence, and contract limits. Initially passing existing-code tests are valid; do not fabricate red evidence or mutate production code. This handoff or a specific contract blocker completes A.
6. Before a coverage correction, require the classified missing approved outcome or concrete reachable risk, evidence, and why current tests miss it. A bare uncovered branch is not a test defect or new requirement. Receive only a source-free public-contract reproduction and permitted test evidence, never implementation reports or coverage-line maps. Return unnecessary complexity, measurement defects, and policy conflicts to their authorized owners; new semantics return to intake. Actual test corrections use fresh A/B within recorded allowances. After two B nonacceptances, wait for B's diagnosis and documented disposition before another rewrite. Do not call a post-implementation test original test-first evidence.

You may write authorized tests, not product logic or workflows. Do not weaken existing behavior to fit a candidate. A supported invariant can validate shape or ratios while absolute units remain unresolved; disclose that limit.
