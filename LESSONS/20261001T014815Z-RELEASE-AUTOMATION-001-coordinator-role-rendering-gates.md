# Keep blind-role rendering requirements in the role runner

- ID: 20261001T014815Z-RELEASE-AUTOMATION-001-coordinator-role-rendering-gates
- Date: 2026-10-01T01:48:15Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: Two autouse test fixtures enforced the blind A/B traceback mode for every caller. All focused blind-role checks passed, but unchanged ordinary canonical pytest failed 11 cases at setup before reaching their behavior assertions.
- Evidence: Candidate 2db547ada5628576a56211430ac59510e9bbf8f3, reports/verification/full-710vafdy/: 1,604 passes and 11 setup errors. Fresh A7 removed only the two fixture gates; all remaining test syntax stayed identical. Its 11 affected cases passed through the source-free runner and ordinary default pytest under parent coverage; reports/release-automation/roles/A7-report.md.
- Lesson: Configure source-free warnings and tracebacks in blind-role launchers. Keep behavior tests compatible with the installed canonical caller, and check that caller mode when changing diagnostic fixtures.
