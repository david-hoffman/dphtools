# Calibrate full verification job limits across workflows

- ID: 20261001T053442Z-release-automation-001-coordinator-ci-timeout
- Date: 2026-10-01T05:34:42Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: Hosted Windows CI hit its 30-minute job limit while tests were still running. It retained early logs/build files but no final checks, JUnit or coverage report. macOS and Linux completed the same canonical full; release verification already has a 45-minute job limit.
- Evidence: [Windows job](https://github.com/david-hoffman/dphtools/actions/runs/36817510212/job/110225683204); retained reports/release-automation/roles/hosted-windows-cancellation-annotations.json explicitly reports the 30-minute maximum; hosted macOS/Linux evidence and workflow timeout fields corroborate the mismatch.
- Lesson: Route a confirmed deadline cancellation through scoped infrastructure maintenance. Preserve the full gate and require completed platform reports afterward. Early step success or retained partial artifacts do not establish passing tests or complete coverage; a longer limit is not proof of readiness.
