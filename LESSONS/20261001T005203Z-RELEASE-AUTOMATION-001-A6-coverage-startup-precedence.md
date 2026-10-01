# Explicit child coverage settings must precede automatic startup

- ID: 20261001T005203Z-RELEASE-AUTOMATION-001-A6-coverage-startup-precedence
- Date: 2026-10-01T00:52:03Z
- Task/role: RELEASE-AUTOMATION-001 / A6
- Status: confirmed
- Observation: With pinned coverage 7.16.1 parent subprocess measurement, inherited serialized configuration outranked explicit file/data settings, and its early startup hook ran before the test fixture hook. A missing-config control consequently exited zero. Parallel data used a pid-prefixed token rather than the fixture's bare numeric token.
- Evidence: reports/release-automation/roles/A6-report.md attempts 1–2 reproduced fixture failures; attempts 6–7 passed the corrected actual wheel/source helper measurement and valid-startup/failure controls in both modes. Retained helper data matched actual process IDs; ordinary combination kept raw files.
- Lesson: Run disposable test instrumentation before automatic coverage startup. Override inherited serialized settings only for explicit child file configuration; test inherited-only measurement as well. Associate raw helper data with real process IDs without requiring an entire filename schema. Nonempty records do not establish complete coverage.
