# Use the required network environment for canonical verification

- ID: 20261001T061913Z-release-automation-001-coordinator-full-network-mode
- Date: 2026-10-01T06:19:13Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: A full run launched in the default command environment became slow in pip installation subprocesses. A read-only PyPI HEAD probe there failed DNS resolution; the identical probe with approved network access returned HTTP200 in 0.167 s. The incomplete full run was interrupted and retained; it produced no final test, coverage or checks report.
- Evidence: reports/release-automation/roles/network-probe-default.json; network-probe-approved.json; full-q68a7j33-activity-061009.json; full-interrupted-environment-candidate-4cb2ade8db85.json (exit143 and retained report/log pointer).
- Lesson: Select the required approved network environment for canonical dependency/audit and real artifact-installation verification. Passing early stages do not establish later package-index access. Diagnose this execution mismatch without changing tests/product code or treating an incomplete run as product-red; retain the attempt and rerun the complete gate.
