# Report schema after removing a coverage threshold

- ID: 20261007T060835Z-ASD-COVERAGE-POLICY-001-author-report-schema
- Date: 2026-10-07T06:08:35Z
- Task/role: ASD-COVERAGE-POLICY-001 / author
- Status: confirmed
- Observation: Removing the completeness gate exposed malformed nonempty coverage details that the old zero-missing requirement had rejected incidentally. Count/list-length consistency alone accepted a string as a missing line and null as a missing branch.
- Evidence: The independent behavior_application_review reproduced both cases through the public full command: exit 0 and a passing receipt despite otherwise consistent JSON/XML counts. The finding and correction are recorded in [the adoption task](../docs/tasks/ASD-COVERAGE-POLICY-001.md). Regression cases use test_full_rejects_invalid_reports_even_when_all_tools_pass; legitimate incomplete reports use test_full_accepts_honest_incomplete_coverage_and_retains_diagnostics.
- Lesson: Keep report-schema validation separate from a numerical target. Test malformed details with honest counts and preserve valid incomplete-report alternatives when relaxing the target. This is a measurement defect, not a reason to restore a percentage gate or commission speculative behavior tests.
