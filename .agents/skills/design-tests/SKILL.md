---
name: design-tests
description: Write independent end-to-end-first tests from an approved behavioral contract, without inspecting the implementation.
metadata:
  version: "1.0"
  status: released
---

# A — Test author

Follow installed repository `AGENTS.md` for shared session, access, verification, budget, and reporting rules. A's packet contains approved behavior, public interfaces, approved fixtures, and test conventions.

1. Map the packet's approved requirement group to observable success, relevant errors, and boundaries. Group related cases into one coherent checkpoint.
2. Prefer the real product entry point: browser journey, executable, service endpoint, or library API. Exercise owned components together; isolate test data and uncontrollable external services.
3. Trace each expected result to approved behavior, an applicable primary reference, or a mathematical invariant. Unspecified units, conventions, and estimators return to intake before dependent assertions. Coverage supplies no missing contract. Do not compute expectations with the code under test or mock away required behavior.
4. Use smaller integration/unit tests only for a meaningful gap. Do not duplicate E2E coverage just to create more tests. Keep helpers simple.
5. Run applicable checks and classify failures under specification section 4. Return tests, a concise requirement/expectation-source mapping, baseline evidence, and any contract limits. This handoff or a specific contract blocker completes A; B reviews before the checkpoint and C.
6. A suspected later test defect or coverage gap comes back through fresh A/B work. Do not call a post-implementation test original test-first evidence.

You may write authorized tests, not product logic or workflows. Do not weaken existing behavior to fit a candidate. A supported invariant can validate shape or ratios while absolute units remain unresolved; disclose that limit.
