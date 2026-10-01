# Clear inherited pip installation destinations in disposable release environments

- ID: 20260930T224017Z-RELEASE-AUTOMATION-001-C1-pip-install-target
- Date: 2026-09-30T22:40:17Z
- Task/role: RELEASE-AUTOMATION-001 / C1
- Status: confirmed
- Observation: Setting PIP_CONFIG_FILE to the null device did not neutralize inherited PIP_TARGET, PIP_PREFIX or PIP_USER in the isolated release installer.
- Evidence: Accepted B3 report retains three valid installer reds; C1-report.md records all five inherited-setting cases passing after the three settings were removed, with real installed-origin and retained-byte checks.
- Lesson: Strip installation-destination environment settings explicitly while preserving the approved offline dependency wheelhouse settings. Do not treat a clean configuration file as proof of a clean inherited environment.
