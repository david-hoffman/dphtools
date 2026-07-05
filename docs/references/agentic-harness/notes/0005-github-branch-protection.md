# GitHub Branch Protection

Source: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches

Accessed: 2026-07-04

## Project-use notes

- Protected branches can require pull request review, status checks, conversation resolution, merge queues, and deployment success before merge.
- Force pushes and branch deletion are blocked by default for protected branches unless explicitly allowed.
- Admin bypass must be considered explicitly because defaults may allow administrators or custom bypass roles around protections.
- Required status check names should be unique to avoid ambiguous merge requirements.

## Harness impact

- The manual checklist requires PRs, review, resolved conversations, unique aggregate `ci-required`, direct-push blocking, force-push blocking, deletion blocking, and admin-bypass review.
- Local files cannot enable branch protection; Phase 1 requires human/admin action.
