# OpenAI Harness Engineering

Source: https://openai.com/index/harness-engineering/

Accessed: 2026-07-04

## Project-use notes

- Reliable agent work depends on clear scaffolding, legible repository knowledge, and feedback loops.
- Repository knowledge should be a system of record that agents can read directly.
- Agents should run standard development tools and local scripts to gather evidence.
- Review and validation loops should be part of the development environment, not an afterthought.

## Harness impact

- `docs/generated/repo-intake.md`, `ARCHITECTURE.md`, and `QUALITY_SCORE.md` make repository knowledge explicit.
- `make harness-check` and `ci-required` turn harness expectations into repeatable checks.
- The PR template asks for command evidence instead of prose claims.
