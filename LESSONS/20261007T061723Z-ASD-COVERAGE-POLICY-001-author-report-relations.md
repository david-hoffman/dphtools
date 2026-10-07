# Coverage-detail relationships after relaxing completeness

- ID: 20261007T061723Z-ASD-COVERAGE-POLICY-001-author-report-relations
- Date: 2026-10-07T06:17:23Z
- Task/role: ASD-COVERAGE-POLICY-001 / author
- Status: confirmed
- Observation: Type and list-length validation still admitted duplicate missing entries and contradictory executed/missing entries after removing the completeness gate.
- Evidence: A fresh read-only root reviewer reproduced duplicate missing lines, duplicate missing branches, and overlapping line details: the candidate returned 0 and the baseline returned 1. Four public full-command regression cases, including branch overlap, then failed against the candidate because corrupt reports returned 0. Their log is `/private/tmp/dphtools-asd-behavior-tests-integrity-before.log`; the finding and correction are recorded in [the adoption task](../docs/tasks/ASD-COVERAGE-POLICY-001.md).
- Lesson: Check the uniqueness and disjointness of coverage-detail sets separately from their types, counters, and numerical coverage target. Preserve valid uncovered statements and negative branch-exit destinations.
