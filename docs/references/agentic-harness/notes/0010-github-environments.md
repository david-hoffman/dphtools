# GitHub Environments

Source: https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments

Accessed: 2026-07-04

## Project-use notes

- Environments can apply deployment protection rules and required reviewers.
- Environment secrets are only available to jobs using that environment after configured rules pass.
- Required reviewers and self-review prevention are relevant to publishing workflows.

## Harness impact

- Release hardening is deferred to a high-risk follow-up because the existing release workflow publishes artifacts.
- The branch protection checklist calls out release environments where applicable.
