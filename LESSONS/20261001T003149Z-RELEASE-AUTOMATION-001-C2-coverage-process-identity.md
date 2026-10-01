# Coverage child identity and startup configuration

- ID: 20261001T003149Z-RELEASE-AUTOMATION-001-C2-coverage-process-identity
- Date: 2026-10-01T00:31:49Z
- Task/role: RELEASE-AUTOMATION-001 / C2
- Status: confirmed
- Observation: Locked coverage 7.16.1 writes parallel process suffixes containing `.pidNNN.`, and inherited COVERAGE_PROCESS_CONFIG takes precedence over COVERAGE_PROCESS_START and the fixture's per-child data setting. A nonempty production trace can coexist with a failing test lookup and a missing-config control that unexpectedly succeeds.
- Evidence: reports/release-automation/roles/C2-child-measurement.json records both actual successful helper PIDs with ordinary 12/12 statement and 2/2 branch coverage. Instrumented tests/test_release_probe_measurement.py fails its actual-PID lookup and missing-config startup control; the installed coverage control.py process_startup explicitly selects serialized config first.
- Lesson: Validate child instrumentation under the actual parent subprocess patch, including ordinary filename identity and configuration precedence. Route fixture corrections through independent A/B; do not change production logic or weaken measurement to satisfy a faulty observer.
