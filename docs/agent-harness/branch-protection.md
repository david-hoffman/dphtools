# Branch protection checklist

Repository: `david-hoffman/dphtools`
Default branch: `main`
Last verified: 2026-07-04
Verified by: Codex via GitHub API

## Required settings

- [x] Pull request required before merge
- [x] Required approvals enabled
- [x] Stale approvals dismissed on new commits
- [x] CODEOWNER review required for protected paths
- [x] Conversations must be resolved
- [x] Status checks required
- [x] `ci-required` selected as required check
- [x] Branch must be up to date before merge or merge queue enabled
- [x] Direct pushes blocked
- [x] Force pushes blocked
- [x] Branch deletion blocked
- [x] Admin bypass disabled after emergency path is confirmed
- [x] GitHub Actions default permissions set to read-only where possible
- [ ] Dependency graph enabled, if available
- [x] Dependabot alerts enabled, if available
- [x] Dependabot security updates enabled, if available
- [x] Secret scanning and push protection enabled, if available
- [ ] Release environments configured, if applicable

## Manual steps

1. Go to repository settings.
2. Open branch protection or repository rulesets.
3. Create a rule for `main`.
4. Enable the required settings above.
5. Select `ci-required` after the workflow has run once and the check name exists.
6. Save the rule.
7. Verify a direct push to `main` is blocked.
8. Verify a PR cannot merge while `ci-required` is failing.

## Notes

Phase 1 branch protection and security settings were configured through the GitHub API on 2026-07-04. See `docs/generated/phase1-github-protection.md`.

Direct-push blocking is verified by branch protection settings, not by attempting a real direct push to `main`.

The `CODEOWNERS` file is committed on the harness branch and takes full effect after that branch is merged to `main`.

Release environments and trusted publishing remain Phase 4 work.
