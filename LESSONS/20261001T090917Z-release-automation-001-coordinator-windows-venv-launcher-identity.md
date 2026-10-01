# Windows virtual-environment launcher PID differs from the measured interpreter

- ID: 20261001T090917Z-release-automation-001-coordinator-windows-venv-launcher-identity
- Date: 2026-10-01T09:09:17Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: A test assumed subprocess.Popen.pid appeared in coveragefiles; Windowsvenvredirector spawns actualPythonprocess, so filenameusesdifferent os.getpid.
- Evidence: run36835101659job110280518822 completes1614passes/1fixtureassertfailure;CPython3.10.11venv/launcher/subprocess andlockedcoverage7.16.1 publicsource receipts A8-public-tooling/source-receipts.json.
- Lesson: Observe executinginterpreter identity portably andbind trace to each successfulownedhelperinvocation. Keep parent/startup/negativecontrol traces from satisfying it. Aggregate100% doesnotprove that association. FreshA/B required forfixturecorrection.
