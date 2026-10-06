# Serial groups retain independent fixture costs

- ID: 20261006T174920Z-TEST-SUITE-PERFORMANCE-001-author-serial-group-costs
- Date: 2026-10-06T17:49:20Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: The two real distributed roundtrip parameters use separate function-scoped proofs. Their measured Windows setup phases are 61.68 and 91.41 seconds; treating the serial pair as a shared fixture would omit 61.68 seconds of work.
- Evidence: [Run 37503901042](https://github.com/david-hoffman/dphtools/actions/runs/37503901042), Windows shard 5 workers 1/2 execution records; `tests/test_verification_distributed.py::test_real_distributed_roundtrip_preserves_complete_nodes_and_child_coverage[4]` and `[8]`, candidate `cb010c05a7c767c21c5d14cd00249026be7e2f6c`.
- Lesson: Normalize setup only for actual shared fixtures. A serial resource group retains every independent setup, call and teardown cost. Whether serialization reduces contention remains a hypothesis requiring a fresh full hosted run.
