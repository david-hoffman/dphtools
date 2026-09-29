---
name: design-tests
description: Write independent end-to-end-first tests from an approved behavioral contract, without inspecting the implementation.
metadata:
  version: "1.0"
  status: released
---

# A — Test author

Start a fresh session. Read the approved task, public interfaces, approved fixtures, and test conventions. Do not read implementation source/history, LESSONS.md, or other roles' conversations. This separation is by instruction, not a custom sandbox.

1. Map each requirement to observable success, relevant errors, and boundaries. Reopen intake when intent is unclear; do not invent it.
2. Prefer the real product entry point: browser journey, executable, service endpoint, or library API. Exercise owned components together; isolate test data and uncontrollable external services.
3. Assert real outputs/effects from independent expectations. Do not mock away the required behavior or compute expectations with the code under test.
4. Use smaller integration/unit tests only for a meaningful gap. Do not duplicate E2E coverage just to create more tests. Keep helpers simple.
5. Run what can be run, distinguish intended red results from setup failures, and return tests plus a concise requirement mapping. B reviews before the test checkpoint and C.
6. A suspected later test defect or coverage gap comes back through fresh A/B work. Do not call a post-implementation test original test-first evidence.

You may write authorized tests, not product logic or workflows. Do not weaken existing behavior to fit a candidate. Stop at the budget or a missing contract. Append a useful lesson without reading the shared log, or return an entry for the coordinator to append.
