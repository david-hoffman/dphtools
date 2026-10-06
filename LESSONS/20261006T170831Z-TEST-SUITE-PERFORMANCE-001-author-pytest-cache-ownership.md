# Pytest cache ownership belongs to each invocation

- ID: 20261006T170831Z-TEST-SUITE-PERFORMANCE-001-author-pytest-cache-ownership
- Date: 2026-10-06T17:08:31Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: Private worker basetemp and coverage did not isolate pytest's shared cache. A sibling run removed a transient pytest-cache-files directory during nested Windows collection.
- Evidence: CI run 37498756376, Windows shard7 worker3 failed test_merge_rejects_changed_worker_execution[missing] before its intended mutation; the fresh independent review of58812a2 confirmed the ownership defect. Two real overlapping cache regressions pass locally after private cache_dir is supplied.
- Lesson: Give collection and execution invocations absolute cache paths inside their private report directories; retain the cache provider and unchanged collection/error gates.
