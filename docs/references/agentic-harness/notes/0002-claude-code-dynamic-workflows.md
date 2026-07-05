# Claude Code Dynamic Workflows Blog

Source: https://claude.com/blog/introducing-dynamic-workflows-in-claude-code

Accessed: 2026-07-04

## Project-use notes

- Dynamic workflows are useful for work that benefits from parallel subtasks, independent verification, and adversarial review.
- Long-running workflows should preserve progress and make intermediate state visible.
- Workflows consume more resources than a normal session, so start with scoped tasks.

## Harness impact

- Phase 0 documents dynamic workflow intent but does not add `.claude/workflows/` files because schema validation was unavailable.
- The clean-context protocol separates scout, test author, implementer, reviewer, numerics reviewer, and CI triager roles so future workflows have stable boundaries.
