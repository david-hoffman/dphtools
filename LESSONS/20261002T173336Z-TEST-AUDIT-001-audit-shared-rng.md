# Removing generated test cases can change other tests' random inputs

- ID: 20261002T173336Z-TEST-AUDIT-001-audit-shared-rng
- Date: 2026-10-02T17:33:36Z
- Task/role: TEST-AUDIT-001 / audit
- Status: confirmed
- Observation: The obsolete split-image parameter table reinitialized and advanced the same module RNG used by unrelated utility tests before any test ran.
- Evidence: `3f7253c:tests/test_utils.py:37,253-258`; both assignments bind `rng`, and the table consumes its draws at import. `docs/tasks/TEST-AUDIT-001.md` records the cleanup and validation plan.
- Lesson: When deleting generated fixtures, inspect shared state and validate affected sibling tests. Remove obsolete draws rather than retaining unexplained state advancement.
