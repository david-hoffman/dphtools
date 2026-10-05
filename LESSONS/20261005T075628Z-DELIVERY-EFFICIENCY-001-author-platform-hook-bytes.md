# Audited startup-hook bytes can differ by platform

- ID: 20261005T075628Z-DELIVERY-EFFICIENCY-001-author-platform-hook-bytes
- Date: 2026-10-05T07:56:28Z
- Task/role: DELIVERY-EFFICIENCY-001 / author
- Status: confirmed
- Observation: The locked Windows coverage wheel stores its stock startup hook with CRLF bytes; the exact Linux LF allowlist conservatively rejected valid Windows reuse. Pytest truncated the failure's diagnostic string, so retained XML alone did not expose the cause.
- Evidence: Windows CI run 37276696460, job 111655977215, failed the positive reuse case while executing the check fresh. `reports/ci-measurements/windows-coverage-hook-proof.json` verifies the unchanged lock's Windows wheel hash and its 206-byte `a1_coverage.pth`, SHA-256 `f1498191b7f52180654ccdb6195233612805e26344100c093058343ea04afd36`; Linux LF is 205 bytes with the previously audited hash.
- Lesson: Audit exact platform wheel bytes, retain each recognized hash without normalizing runtime identities, and print full failure-only diagnostics so captured output survives assertion truncation.
