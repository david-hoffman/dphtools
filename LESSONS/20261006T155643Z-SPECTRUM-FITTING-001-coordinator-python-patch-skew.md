# Floating Python patches can split one CI workflow

- ID: 20261006T155643Z-SPECTRUM-FITTING-001-coordinator-python-patch-skew
- Date: 2026-10-06T15:56:43Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: Separate Linux jobs requesting Python `3.10` resolved to 3.10.21 and 3.10.22 in the same workflow, causing exact manifest validation to reject a worker before tests ran.
- Evidence: [Run 37483079692](https://github.com/david-hoffman/dphtools/actions/runs/37483079692), candidate `0634499296e143bab6a16daba3f504b5c6e4391d`: collection manifest and failed worker/aggregation receipts differ only at `environment/version`. `tools/verification_shards.py` also seals the workflow run and attempt.
- Lesson: Pin one supported exact interpreter patch per platform consistently across collection, workers and aggregation. Preserve identity checks. Run a fresh complete workflow after repair; failed-job-only retries cannot reuse a collection sealed to an earlier attempt.
