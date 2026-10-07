# Verify canonical receipt consumers alongside producers

- ID: 20261007T211816Z-VERIFICATION-HIERARCHY-001-author-canonical-receipt-consumer
- Date: 2026-10-07T21:18:16Z
- Task/role: VERIFICATION-HIERARCHY-001 / author
- Status: confirmed
- Observation: The verifier emitted version 2.0 receipts while the release preparation reader accepted only version 1.0. A complete current full receipt could not pass the preparation boundary.
- Evidence: `tests/test_release_verification.py::test_canonical_full_report_version_two_is_accepted_for_preparation` failed at the intended receipt reader after a valid legacy control; `reports/verification-hierarchy/modern-report-red.log` retains that result. `modern-report-and-install-focused.log` records 14 passing integration checks after the reader repair, including rejection of scoped, incomplete, changed and missing evidence.
- Lesson: Exercise the public preparation entry with the producer's current complete receipt format whenever verification metadata changes. Validate the seals, retained logs, entire owned-source inventory and exact reports; a producer-only success does not establish consumer compatibility.
