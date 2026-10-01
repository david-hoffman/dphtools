# Check recovery scheduling beyond successful helper execution

- ID: 20261001T022253Z-RELEASE-AUTOMATION-001-D-skipped-ancestor-scheduling
- Date: 2026-10-01T02:22:53Z
- Task/role: RELEASE-AUTOMATION-001 / D (recorded by coordinator)
- Status: confirmed
- Observation: The original-bundle recovery helper chain passed, but four downstream workflow jobs omitted status conditions after an intentionally skipped verification ancestor. D found this scheduler defect despite passing canonical Python verification.
- Evidence: D-report.md for candidate 569df0e6cbacbc29d51f4cfceeeb2bc11132bb55; .github/workflows/make_release.yml jobs verify, bundle, install-retained, publisher, install-published and finalize. GitHub dependency/status-function documentation supports the finding; no live hosted reproduction is claimed.
- Lesson: a successful recovery helper chain does not establish scheduler recovery; audit skipped ancestors and explicit downstream status conditions against actual Actions semantics.
