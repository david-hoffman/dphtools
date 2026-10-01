# Existing timeout settings do not establish platform capacity

- ID: 20261001T074822Z-release-automation-001-coordinator-unmeasured-windows-budget
- Date: 2026-10-01T07:48:22Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: Matching CI's earlier30-minute limit to the release workflow's existing45-minute limit still cancelled Windows before tests returned. Linux/macOS finished; Windows had no final reports after43min9.14s in the test phase.
- Evidence: [Windows job](https://github.com/david-hoffman/dphtools/actions/runs/36826950918/job/110254644623); ignored hosted-windows-cancellation-annotations-36826950918.json explicitly states45m0s. tools/verification.py buffers subprocess output until return, so missing tests.log/checks/JUnit/coverage cannot distinguish slow execution, a stalled child or network waiting. Earlier lesson20261001T053442Z-release-automation-001-coordinator-ci-timeout already requires completed reports rather than treating a larger limit as proof.
- Lesson: An existing workflow limit is not a measured completion time. Keep CI and release capacity consistent, preserve the full gate and require genuine final platform reports. The next90-minute bound is provisioning, not certified success; if it also expires, collect live progress/process evidence before another increase. No product-red or product editing authority derives from a deadline cancellation.
