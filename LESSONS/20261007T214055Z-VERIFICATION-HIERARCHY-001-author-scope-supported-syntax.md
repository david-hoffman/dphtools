# Narrow verification needs positive impact evidence

- ID: 20261007T214055Z-VERIFICATION-HIERARCHY-001-author-scope-supported-syntax
- Date: 2026-10-07T21:40:55Z
- Task/role: VERIFICATION-HIERARCHY-001 / author
- Status: confirmed
- Observation: Unchanged import roots and a body blacklist did not establish an isolated impact. Independent review found native-loading calls, changed import reachability, context-manager hooks and falsely resolved names that remained scoped.
- Evidence: `tests/test_verification_scope.py` now covers those cases through real Git/classifier entry points. `reports/verification-hierarchy/syntax-coverage.log` records 78 passing cases; its coverage report contains no missing statements or branches in `tools/verification_scope.py`. Independent review provisionally closed the source findings after the supported-syntax repair.
- Lesson: Define a small supported syntax set and promote unsupported effects to full. Disclose overpromotion. A known filename or import root does not prove that changed execution stays within one domain.
