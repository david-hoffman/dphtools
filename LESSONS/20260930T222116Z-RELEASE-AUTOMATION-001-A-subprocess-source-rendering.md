# Source-free diagnostics include subprocess operands

- ID: 20260930T222116Z-RELEASE-AUTOMATION-001-A-subprocess-source-rendering
- Date: 2026-09-30T22:21:16Z
- Task/role: RELEASE-AUTOMATION-001 / A, recorded by coordinator
- Status: confirmed
- Observation: Source-free traceback rendering also needs to prevent pytest from displaying raw subprocess result operands.
- Evidence: A2-report.md attempt 2 exposed a child probe line; attempt 9 with `require_success` retained nonzero statuses without displaying that line. The exposure remains disclosed; later A/B probes kept `--tb=no` effective.
- Lesson: Keep the source-free option effective and render numeric status plus sanitized diagnostics. Do not assert directly against a subprocess object that can display command source on failure.
