# Coverage data filenames can change when saved

- ID: 20261001T095118Z-release-automation-001-coordinator-coverage-file-finalizes-on-save
- Date: 2026-10-01T09:51:18Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: A8 initially recorded the public coverage filename before saving. Locked coverage 7.16.1 renamed the ordinary data file on write, so that receipt pointed at a nonexistent file.
- Evidence: A8-report.md records two test-support failures and their correction. B8-report.md independently accepted checkpoint 70cc2a8fb7185995ef0f6a234595876311489a9ec87f0bfb1e24ab0187cf44b4, with both successful helper files retained and control files excluded.
- Lesson: Obtain the finalized filename after ordinary saving through public tooling APIs. Preserve the exact successful invocation's data; do not infer filenames from a launcher PID or accept an arbitrary union of traces.
