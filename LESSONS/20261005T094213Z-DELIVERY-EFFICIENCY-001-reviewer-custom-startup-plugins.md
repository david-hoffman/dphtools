# Identified plugin code does not identify its external inputs

- ID: 20261005T094213Z-DELIVERY-EFFICIENCY-001-reviewer-custom-startup-plugins
- Date: 2026-10-05T09:42:13Z
- Task/role: DELIVERY-EFFICIENCY-001 / reviewer
- Status: confirmed
- Observation: A coverage plugin's unchanged source and startup configuration could read an unbound external flag, producing stale reused evidence.
- Evidence: Independent probe `/tmp/review-startup-plugin-input-ihw9tqrl/proof.json` at `a14fa3e` records original PASS, reused PASS, and fresh FAIL after only the external flag changed. Config/plugin bytes and all three docstring identities were equal; receipts are `fast-mnd95h4i`, `fast-e6alaxo3`, and `fast-eh6appis`.
- Lesson: Decline reuse for custom startup plugins whose external inputs are unknown. Hashing code and its configuration alone is insufficient; preserve ordinary subprocess measurement.
