# Release triggers and version derivation must agree

- ID: 20260930T175327Z-RELEASE-AUTOMATION-001-intake-version-tag-identity
- Date: 2026-09-30T17:53:27Z
- Task/role: RELEASE-AUTOMATION-001 / intake investigation
- Status: confirmed
- Observation: The release trigger accepts `*.*.*`, including prefixed tags, while Versioneer is configured with an empty prefix and selects digit-leading tags. Local history contains both naming styles. No check compares the triggering tag with wheel/source metadata.
- Evidence: Baseline `e557c7d`; `.github/workflows/make_release.yml:5`, `setup.cfg:6`, `versioneer.py:799`; `git tag --sort=-version:refname`. This is a static mismatch finding, not evidence of a particular incorrectly published package.
- Lesson: Define one release-tag grammar and require equality between the requested version, version-derived build metadata, and installed artifact version before publication.
