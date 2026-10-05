# Bind imported providers before reusing evidence

- ID: 20261005T070848Z-DELIVERY-EFFICIENCY-001-reviewer-import-provenance
- Date: 2026-10-05T07:08:48Z
- Task/role: DELIVERY-EFFICIENCY-001 / coordinator recording independent review
- Status: confirmed
- Observation: Source and dependency-version hashes did not bind a check provider injected through an external .pth hook, a startup package, or a native root module. The independent reviewer changed external provider data and reproduced stale successful reuse while a fresh invocation failed.
- Evidence: The repairs through fac1de9 and test_external_site_hook_cannot_reuse_changed_check_provider plus test_unknown_import_inputs_decline_even_identical_receipts reject these real provider cases. Independent review approved 112410d while preserving the full-gate requirements.
- Lesson: Audit actual import paths and startup hooks, bind executable provider bytes, and reject unknown providers. A closed receipt and unchanged source hashes alone do not establish unchanged runtime inputs.
