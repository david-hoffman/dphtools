# Claude Code Skills Blog

Source: https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills

Accessed: 2026-07-04

## Project-use notes

- Treat skills as folders with instructions, scripts, assets, and references, not only a single Markdown file.
- Keep skill files concise and use progressive disclosure through referenced files when a task needs more detail.
- Include gotchas because recurring failure modes are high-signal guidance for agents.
- Prefer verification skills and deterministic scripts where possible.

## Harness impact

- This repository does not add `.claude/` skills in Phase 0 because local schema validation was not available.
- Future skills should include gotchas and required output sections, and should call deterministic scripts rather than relying only on prose.
