# Phase 1 GitHub protection evidence

Date: 2026-07-04
Repository: `david-hoffman/dphtools`
Default branch: `main`
Configured by: Codex using `gh` authenticated as `david-hoffman`

## Facts

- Viewer permission was `ADMIN`.
- Branch `main` was not protected before this phase.
- Branch protection is now configured for `main`.
- GitHub Actions default workflow permissions are now read-only.
- Dependabot alerts are enabled.
- Dependabot security updates are enabled.
- Secret scanning is enabled.
- Secret scanning push protection is enabled.

## Branch protection read-back

- Required status checks enabled: yes.
- Required status check contexts: `ci-required`.
- Strict status checks enabled: yes.
- Pull request reviews required: yes.
- Required approving review count: 1.
- Stale approvals dismissed: yes.
- CODEOWNER reviews required: yes.
- Last-push approval required: yes.
- Conversation resolution required: yes.
- Admin enforcement enabled: yes.
- Force pushes allowed: no.
- Branch deletion allowed: no.

## Actions and security read-back

- `default_workflow_permissions`: `read`.
- `can_approve_pull_request_reviews`: `false`.
- `dependabot_security_updates`: `enabled`.
- `secret_scanning`: `enabled`.
- `secret_scanning_push_protection`: `enabled`.

## Commands run

```bash
gh repo view --json nameWithOwner,defaultBranchRef,viewerPermission
gh api repos/david-hoffman/dphtools/branches/main/protection
gh api -X PUT repos/david-hoffman/dphtools/branches/main/protection ...
gh api -X PUT repos/david-hoffman/dphtools/actions/permissions/workflow ...
gh api -X PUT repos/david-hoffman/dphtools/vulnerability-alerts --silent
gh api -X PUT repos/david-hoffman/dphtools/automated-security-fixes --silent
gh api -X PATCH repos/david-hoffman/dphtools ...
```

## Not verified by destructive action

No real direct push to `main` was attempted. The protection API reports settings that should block direct pushes, force pushes, and branch deletion.

## Remaining work

- Merge the Phase 0 harness branch so `CODEOWNERS` exists on `main`.
- Let the `ci-required` workflow run at least once from the harness branch.
- Configure release environments and trusted publishing in Phase 4.
