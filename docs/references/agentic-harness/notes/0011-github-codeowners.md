# GitHub CODEOWNERS

Source: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

Accessed: 2026-07-04

## Project-use notes

- CODEOWNERS can live under `.github/`, repository root, or `docs/`; GitHub uses the first one it finds in that order.
- Code owners must have write access.
- Branch protection can require code owner review.
- To protect CODEOWNERS itself, own `.github/` or the CODEOWNERS file.

## Harness impact

- `.github/CODEOWNERS` assigns protected governance, packaging, harness, and release paths to `@david-hoffman`, inferred from the repository remote.
- If that account is not the right owner or lacks write access, update CODEOWNERS in a high-risk governance issue.
