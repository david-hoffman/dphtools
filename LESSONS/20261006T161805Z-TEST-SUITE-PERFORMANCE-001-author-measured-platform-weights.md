# New tests need measured scheduling costs

- ID: 20261006T161805Z-TEST-SUITE-PERFORMANCE-001-author-measured-platform-weights
- Date: 2026-10-06T16:18:05Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: Equal one-second weights for new infrastructure cases substantially understated their complete setup/call/teardown costs. The hosted trial exceeded 300 seconds despite balanced historical weights.
- Evidence: [Run 37491893447](https://github.com/david-hoffman/dphtools/actions/runs/37491893447) measured new distributed cases around 14–38 seconds on Linux; complete test windows were 337 seconds on macOS and 525 seconds on Windows. All original assertions were preserved.
- Lesson: Refresh advisory scheduling weights from actual platform phase receipts and use the same selected weights at both partition levels. Weights never substitute for fresh test or coverage evidence.
