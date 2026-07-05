# GitHub Status Checks

Source: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks

Accessed: 2026-07-04

## Project-use notes

- Check conclusions can include success, failure, cancelled, neutral, skipped, stale, and timed out.
- GitHub can treat skipped or neutral checks as acceptable in some dependency contexts, so an aggregate gate should make skipped required jobs explicit.
- Workflows may be skipped through commit-message controls; branch protection and review policy should account for that risk.

## Harness impact

- `ci-required` reads the `needs` results and fails if any required job is not `success`.
- Required workflows avoid path filters so the required gate is harder to bypass accidentally.
