# Audited startup hooks still have configuration inputs

- ID: 20261005T084809Z-DELIVERY-EFFICIENCY-001-reviewer-startup-config-inputs
- Date: 2026-10-05T08:48:09Z
- Task/role: DELIVERY-EFFICIENCY-001 / reviewer
- Status: confirmed
- Observation: Hashing an audited coverage startup hook and the environment's configuration path did not bind the external file the hook actually reads.
- Evidence: Independent probe `/tmp/review-coverage-start-_khhjdo2` at `ca1a94e` produced original PASS, requested-reuse PASS, and fresh FAIL after only an active coverage plugin option changed. Receipts `fast-xjaagdw1`, `fast-s2khlw1m`, and `fast-5djey0mk` recorded equal docstring identities.
- Lesson: Identify active startup configuration bytes and configuration precedence, or decline reuse. Retain real subprocess measurement when repairing cache eligibility.
