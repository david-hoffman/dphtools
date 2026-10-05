# Implicit configuration belongs in evidence identities

- ID: 20261005T073809Z-DELIVERY-EFFICIENCY-001-author-implicit-config-inputs
- Date: 2026-10-05T07:38:09Z
- Task/role: DELIVERY-EFFICIENCY-001 / author
- Status: confirmed
- Observation: Pinned pydocstyle recognizes additional root filenames and recursively inherits ancestor configuration. Changing these unbound bytes produced reused PASS while a fresh command failed.
- Evidence: Independent real-command reproductions in `/tmp/review-real-doc-config-8p_iqsiq/`; original, reused, and fresh sealed receipts demonstrate both root `.pydocstylerc` and inherited `.pydocstyle` failures on candidate `fe148fc`. CPython 3.10's Windows FileFinder also normalizes extension case, while the eligibility guard checked case-sensitive suffixes.
- Lesson: Bind every recognized implicit configuration path, including ancestors, and use conservative cross-platform import-provider detection before accepting reused evidence.
