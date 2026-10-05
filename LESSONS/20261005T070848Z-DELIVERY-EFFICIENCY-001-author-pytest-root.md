# Freeze test IDs relative to the intended repository

- ID: 20261005T070848Z-DELIVERY-EFFICIENCY-001-author-pytest-root
- Date: 2026-10-05T07:08:48Z
- Task/role: DELIVERY-EFFICIENCY-001 / author
- Status: confirmed
- Observation: A genuine shard fixture inside the verifier's report directory inherited the surrounding project's pytest configuration. Its collected node IDs gained parent-relative path prefixes, unlike the same fixture under /tmp.
- Evidence: CI run 73 exposed the mismatch. test_collection_node_ids_remain_repository_relative_under_ancestor_configuration reproduced it with an ancestor pytest.ini and passed after collection, shard execution, and full verification all selected --rootdir=run.root. The 87-case shard suite retained exact owned helper statement and branch coverage.
- Lesson: Explicitly bind pytest's root for both collection and execution. Test nested repository fixtures under an ancestor configuration, rather than relying only on an isolated temporary directory.
