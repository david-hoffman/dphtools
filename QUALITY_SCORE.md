# QUALITY_SCORE.md

Last updated: 2026-07-05
Default branch: `main`
Default branch commit: `c81ffdf108c33e3571757fd0996d685c47bd7d6a`

## CI
- Required gate present: yes
- Required gate unskipped: yes
- Matrix OS coverage: `ubuntu-latest`, `macos-latest`, `windows-latest`
- Matrix runtime coverage: Python 3.10 only; package metadata declares Python `>=3.8`

## Tests
- Line coverage: 15.56%
- Branch coverage: 11.30%
- Diff coverage policy: enabled for changed product lines under `dphtools/`
- Mutation/property testing status: not enabled
- Flaky tests: unknown

## Harness
- `AGENTS.md` current: yes
- Claude skills validated: not installed
- Clean-context metadata enforced: yes for pull requests with product source changes
- Reference manifest valid: yes
- Branch protection configured: yes, via GitHub API on 2026-07-04
- Required labels configured: yes, via GitHub CLI on 2026-07-05
- Release environments configured: yes, `test-pypi`, `pypi`, and `anaconda`

## Known risks
| Risk | Severity | Owner issue | Current mitigation |
|---|---:|---|---|
| Runtime support metadata and CI matrix are not aligned. | Medium | follow-up required | CI preserves the existing Python 3.10 matrix and documents the gap. |
| Scientific/numerical behavior is under-documented. | Medium | follow-up required | Numerical tolerance, fixture, and oracle policies are now present. |
| Release workflow publishes on tags using repository secrets. | High | follow-up required | Phase 0 leaves release semantics unchanged and documents required human/admin review. |
| Release credentials may still be repository-level secrets. | High | follow-up required | Release jobs now use protected environments; prefer environment-scoped secrets or trusted publishing next. |
