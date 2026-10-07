# Floating Python versions across dependent CI jobs

- ID: 20261007T200851Z-ASD-COVERAGE-POLICY-001-author-ci-python-drift
- Date: 2026-10-07T20:08:51Z
- Task/role: ASD-COVERAGE-POLICY-001 / author
- Status: confirmed
- Observation: Independent jobs requesting Python `3.10` resolved different patch versions within one CI run, so an exact manifest identity correctly blocked aggregation even though all test shards passed.
- Evidence: [PR 20 run 37581500018](https://github.com/david-hoffman/dphtools/actions/runs/37581500018) retained Python 3.10.21 in Linux collection and a successful shard; failed Linux aggregation recorded 3.10.22. All other recorded portable fields matched. Artifacts are retained in `/private/tmp/dphtools-pr20-ci-linux/`. macOS ARM64 and Windows passed on 3.10.11; the verified Actions version manifest supports the approved per-platform pins.
- Lesson: Use one exact available interpreter pin per platform across dependent collection, shard, and aggregation jobs. Check architecture availability before pinning a new patch; preserve identity validation when repairing environment drift.
