# Keep unchanged-runtime fixtures stable and select their current receipt

- ID: 20261005T070848Z-DELIVERY-EFFICIENCY-001-author-cold-runtime-receipts
- Date: 2026-10-05T07:08:48Z
- Task/role: DELIVERY-EFFICIENCY-001 / author
- Status: confirmed
- Observation: A cold genuine docstring invocation added 49 bytecode files after its runtime fingerprint, correctly invalidating reuse. A test could also select the original receipt because the reuse log intentionally linked that receipt as provenance.
- Evidence: CI run 73 failed the unchanged-reuse assertion; private copied-prefix proof-v2.json under /tmp/reuse-cold-proof-mqn7ythd showed passed-to-passed after the bytecode changes and passed-to-reused with zero changed bytes and distinct receipts under PYTHONDONTWRITEBYTECODE=1. The repaired 23-case suite passed.
- Lesson: Control incidental bytecode and user-site inputs in an unchanged-runtime fixture. Select the newly created receipt by identity, rather than matching any receipt path echoed in diagnostics.
