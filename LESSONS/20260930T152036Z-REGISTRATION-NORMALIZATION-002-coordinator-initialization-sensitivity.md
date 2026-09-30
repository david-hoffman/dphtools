# Separate normalization invariants from initialization-sensitive convergence

- ID: 20260930T152036Z-REGISTRATION-NORMALIZATION-002-coordinator-initialization-sensitivity
- Date: 2026-09-30T15:20:36Z
- Task/role: REGISTRATION-NORMALIZATION-002 / coordinator
- Status: confirmed
- Observation: A small affine example failed with normalization enabled and a forced initial variance of 0.01, but the same points, expected transform, and tolerances passed with the public default initialization. Direct coordinate-change and round-trip invariants also passed. The unsupported convergence expectation was a test defect, not confirmed evidence against normalization.
- Evidence: B1's independent comparison and disposition, retained in `docs/tasks/REGISTRATION-NORMALIZATION-002-test-evidence.md`: maximum coordinate error 1.4082 distance units with the forced variance, versus 4.44e-16 with the default. A2 retained the fixture and tolerances and passed all 12 tests.
- Lesson: Check the coordinate-change identity separately from the iterative fit, and establish the initialization contract before classifying a local solver's mismatch as product-red. Retain failed test assumptions and their correction history.
