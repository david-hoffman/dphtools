# OpenAI Symphony

Source: https://openai.com/index/open-source-codex-orchestration-symphony/

Accessed: 2026-07-04

## Project-use notes

- Repository-owned workflow policy keeps agent behavior versioned with code.
- Per-issue workspaces reduce accidental cross-task contamination.
- Orchestration should expose observability and explicit handoff states instead of pretending every run ends at done.
- High-level objectives and tooling often work better than rigid state machines for capable coding agents.

## Harness impact

- `AGENTS.md`, harness docs, and CI scripts are the repo-owned workflow contract for this project.
- The clean-context protocol requires fresh sessions/worktrees for agent roles after enforcement is enabled.
- Phase 0 records manual blockers rather than claiming GitHub-side settings were applied.
