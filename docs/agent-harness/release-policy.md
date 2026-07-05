# Release Policy

Publishing is high risk.

Required release controls:

- Release pull requests must use `release-guard`.
- Release workflow changes require owner/human approval.
- Publishing jobs must use protected GitHub environments.
- Release jobs must build from a clean tag or protected release branch.
- Release jobs must test installed artifacts, not only the source tree.
- Test PyPI must succeed before production PyPI publishing.
- Release notes must include API changes, dependency changes, deprecations, and downstream effects.
- Publish credentials must not be available to ordinary pull request workflows.

Current release workflow:

- Triggers only on version tag pushes matching `*.*.*`.
- Builds source and wheel distributions.
- Runs `twine check`.
- Installs the built wheel in a clean virtual environment and imports `dphtools`.
- Uploads build artifacts between jobs.
- Publishes through `test-pypi`, `pypi`, and `anaconda` environments.
- Each release environment has required reviewers, admin bypass disabled, and self-review prevention enabled.

Repository secrets still need owner review. Prefer environment-scoped secrets or trusted publishing where available.
