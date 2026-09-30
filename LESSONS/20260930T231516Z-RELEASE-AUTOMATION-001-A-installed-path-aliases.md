# Normalize both paths in installed-origin checks

- ID: 20260930T231516Z-RELEASE-AUTOMATION-001-A-installed-path-aliases
- Date: 2026-09-30T23:15:16Z
- Task/role: RELEASE-AUTOMATION-001 / A (verbatim lesson handoff recorded by coordinator)
- Status: confirmed
- Observation: Real install/check/probe and independent observer statuses were successful, but a new installed-origin comparison failed because macOS reported `/private/var` against an environment path beneath `/var`.
- Evidence: `tests/test_release_completion.py::test_real_installs_and_probes_strip_dummy_inherited_credentials`; A4-report.md retains two failing path-comparison attempts, followed by seven final passing cases after Path.resolve normalization. No product change occurred.
- Lesson: Independent installed-origin assertions must normalize both paths before ancestry comparison on macOS; this session's successful subprocess/observer statuses and `/var` versus `/private/var` alias diagnosis establish a test defect rather than an installation defect.
