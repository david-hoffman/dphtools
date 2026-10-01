# Opaque fixture builds need a verified version

- ID: 20260930T205853Z-RELEASE-AUTOMATION-001-A-opaque-build-version
- Date: 2026-09-30T20:58:53Z
- Task/role: RELEASE-AUTOMATION-001 / A, recorded by coordinator
- Status: confirmed
- Observation: Do not assume canonical fixture versioning from a no-Git opaque build. Metadata observed `0+unknown`; isolated fixture Git identities produced the approved example version and passed the archive metadata/Git-absence check. No runtime versioning mechanism was inspected or inferred.
- Evidence: A's retained `reports/release-automation/roles/A-report-before-git-clarification.md`; `test_real_package_fixture_has_declared_metadata_and_no_git_dependency` passed with independently created fixture tags.
- Lesson: Verify built fixture metadata before treating installation failures as product failures. Keep fixture Git identities separate from owner history and test the resulting source archive without Git.
