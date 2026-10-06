# Measured test setup charges move between consumers

- ID: 20261006T163753Z-TEST-SUITE-PERFORMANCE-001-author-shared-fixture-affinity
- Date: 2026-10-06T16:37:53Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: After duration repartitioning, unchanged module fixtures added cold setup to nodes whose prior weights described warm consumers. The same three fixture groups were repeatedly rebuilt across workers.
- Evidence: CI runs 37491893447 and 37494868812; test_verification_shards interface_shards and completed_shards groups and the imported completed_shards group in test_verification_distributed. The latter trial retains 108 combined source records for the advisory model. https://github.com/david-hoffman/dphtools/actions/runs/37494868812
- Lesson: Keep known expensive shared-fixture consumers cohesive at both scheduling levels and charge setup once in the advisory model. Verify actual complete timing; a modeled worker maximum does not account for cross-job setup or launch skew.
