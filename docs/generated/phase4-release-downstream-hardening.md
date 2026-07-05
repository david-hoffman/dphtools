# Phase 4 release and downstream hardening evidence

Date: 2026-07-05
Repository: `david-hoffman/dphtools`

## Facts

- No downstream consumers were discovered during repository intake.
- Downstream smoke passes with an explicit no-known-consumer message.
- Release workflow now builds once, checks metadata, smoke-tests the installed wheel, and publishes from downloaded artifacts.
- Release publishing jobs use protected environments: `test-pypi`, `pypi`, and `anaconda`.
- Release environments require reviewer approval.
- Release environments have admin bypass disabled.
- Release environments prevent self-review.
- Ordinary pull request CI uses read-only permissions and has no release publishing jobs.

## Release Workflow Controls

- Trigger: version tag pushes matching `*.*.*`.
- Permissions: `contents: read`.
- Build job:
  - Builds source distribution and wheel.
  - Runs `twine check`.
  - Installs the built wheel in a clean virtual environment.
  - Imports `dphtools` from the installed artifact.
- Test PyPI job:
  - Uses `environment: test-pypi`.
  - Publishes downloaded build artifacts.
- PyPI job:
  - Uses `environment: pypi`.
  - Requires Test PyPI job success.
  - Skips release candidates containing `rc`.
- Anaconda job:
  - Uses `environment: anaconda`.
  - Requires PyPI job success.
  - Skips release candidates containing `rc`.

## Remaining Hardening

- Move repository-level publishing secrets into environment-scoped secrets.
- Prefer trusted publishing for PyPI/Test PyPI if supported by the project.
- Add downstream smoke targets if consumers are identified.

## GitHub Environment Read-Back

`gh api repos/david-hoffman/dphtools/environments` returned:

- `test-pypi`: `can_admins_bypass=false`, required reviewers configured, `prevent_self_review=true`.
- `pypi`: `can_admins_bypass=false`, required reviewers configured, `prevent_self_review=true`.
- `anaconda`: `can_admins_bypass=false`, required reviewers configured, `prevent_self_review=true`.
