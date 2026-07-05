# GitHub Dependabot Options

Source: https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference

Accessed: 2026-07-04

## Project-use notes

- Dependabot configuration uses top-level `version: 2`.
- Each update entry defines `package-ecosystem`, `directory`, and `schedule.interval`.
- GitHub Actions and Python package ecosystems can be configured separately.

## Harness impact

- `.github/dependabot.yml` configures weekly updates for GitHub Actions and pip manifests at repository root.
- Dependabot PRs receive area labels to route them through the harness.
