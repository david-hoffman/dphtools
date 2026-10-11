# Isolate the guard a negative test claims to exercise

- ID: 20261011T225144Z-TEST-AUDIT-002-coordinator-isolate-negative-guards
- Date: 2026-10-11T22:51:44Z
- Task/role: TEST-AUDIT-002 / coordinator
- Status: confirmed
- Observation: Invalid test inputs can trigger an earlier integrity or conservative-fallback guard, leaving the named semantic guard untested.
- Evidence: Base b995f584; test_verification_shards.py foreign run/platform cases retained stale seals; its index-duplicate case retained foreign executions; test_verification_scope.py invalid inventory also changed the inventory path, and unchanged-prefix proof added an unsupported call. Read-only owner tracing is retained in /private/tmp/dphtools-test-audit-002-boundaries.md. These are test-value findings, not mutation or product-red evidence.
- Lesson: Keep unrelated guards valid, isolate the candidate delta, and observe the intended rejection reason. Preserve separate byte-integrity/fallback controls.
