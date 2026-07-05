# Branch protection checklist

Repository: `david-hoffman/dphtools`
Default branch: `main`
Last verified: 2026-07-04
Verified by: Codex local repository intake

## Required settings

- [ ] Pull request required before merge
- [ ] Required approvals enabled
- [ ] Stale approvals dismissed on new commits
- [ ] CODEOWNER review required for protected paths
- [ ] Conversations must be resolved
- [ ] Status checks required
- [ ] `ci-required` selected as required check
- [ ] Branch must be up to date before merge or merge queue enabled
- [ ] Direct pushes blocked
- [ ] Force pushes blocked
- [ ] Branch deletion blocked
- [ ] Admin bypass disabled after emergency path is confirmed
- [ ] GitHub Actions default permissions set to read-only where possible
- [ ] Dependency graph enabled, if available
- [ ] Dependabot alerts enabled, if available
- [ ] Dependabot security updates enabled, if available
- [ ] Secret scanning and push protection enabled, if available
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

Local repository files cannot configure branch protection, security settings, secrets, environments, trusted publishing, or the default branch. Treat this file as the Phase 1 manual checklist, not evidence that GitHub settings are complete.
