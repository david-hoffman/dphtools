# Floating interpreter selectors can violate a sealed distributed runtime

- ID: 20261006T163753Z-TEST-SUITE-PERFORMANCE-001-author-fixed-interpreter-patch
- Date: 2026-10-06T16:37:53Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: One Linux shard selected Python 3.10.22 after collection selected 3.10.21; other platforms used 3.10.11. The portable identity gate rejected the Linux mismatch before tests.
- Evidence: CI run 37494868812, Linux shard 3 job 112378218979, retained collection/shard checks and shard-inputs.log. https://github.com/david-hoffman/dphtools/actions/runs/37494868812
- Lesson: Pin one exact interpreter patch across collection, execution and aggregation. Rerun complete platform proofs after changing that pin; never weaken the identity comparison.
