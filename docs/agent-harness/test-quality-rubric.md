# Test Quality Rubric

A valid test must:

- Assert desired public behavior.
- Fail on the base commit if it is a regression or feature test.
- Use deterministic data.
- Avoid network dependency unless marked integration and isolated.
- Avoid local machine state.
- State units and coordinate conventions when relevant.
- Use tolerances that catch real regressions without creating platform flakes.
- Avoid implementation details unless the issue targets internal behavior.

Weak tests include:

- Import-only tests.
- Execution without assertions.
- Broad snapshots without semantic checks.
- Exact floating-point assertions without an oracle.
- Expected values recomputed by the same code path under test.
- Random data without a seed.
- Platform skips without an issue.
