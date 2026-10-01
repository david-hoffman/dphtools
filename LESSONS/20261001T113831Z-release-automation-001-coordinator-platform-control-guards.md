# Passing host tests can omit a platform control

- ID: 20261001T113831Z-release-automation-001-coordinator-platform-control-guards
- Date: 2026-10-01T11:38:31Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: All A9 host tests passed, but its new alternate executable-name control was guarded off on Windows even though the interpreter oracle ran on every platform.
- Evidence: B9-report.md SHA-256 `82aca23c0dbf678689d74cd4871b0f66864f8f277cb2ede504f306962b6e7113`; proposed A9 checkpoint `f6001704294bfb37fdf3691021657cce446ef084ce9875490990b7eb1657ce35`; B9 nonaccepted the missing required Windows control.
- Lesson: Inspect each required platform control separately from the oracle and host pass count. Use a real operation supported on that platform; keep unexecuted platform branches explicit until actual CI runs.
