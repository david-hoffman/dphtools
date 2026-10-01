# Parent dependency installation does not provision clean child installs

- ID: 20261001T090917Z-release-automation-001-coordinator-parent-lock-is-not-child-input
- Date: 2026-10-01T09:09:17Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: macOS CI parent locked install succeeded, yet fresh real installation tests failed reading files.pythonhosted.org; one deliberately disablespipcache.
- Evidence: run36835101659 job110280518359;2explicitReadTimeoutErrors; CI-maintenance-3-report.md stages85hashcheckednativehostwheels andoffline dryrunPASS.
- Lesson: Provision complete hashchecked platform/interpreter dependency files before canonical tests; parent site-packages and pipcache do not guarantee childinputs. Acquisitionfailures remain fail. Three otherMac causes remain unresolved; hostedrerunneeded.
