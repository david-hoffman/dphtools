# Clean-Context Protocol

Phase 0 documents this protocol. Phase 2 will enforce metadata.

## Roles

| Role | Context rule | Allowed output | Must not do |
|---|---|---|---|
| Scout | Fresh read-only context | Issue map, affected files, risk assessment, test strategy | Modify code |
| Test author | Fresh context from issue and architecture docs only | Tests, fixtures, test design note, red-test proof | Read implementation plan or change implementation code |
| Implementer | Fresh context from issue plus red-test patch | Implementation code, docs, migration notes | Weaken/delete tests or alter test intent |
| Adversarial reviewer | Fresh context from PR diff and issue | Review report and requested changes | Author implementation |
| Numerics reviewer | Fresh context from PR diff, tests, architecture docs | Scientific/numerical review report | Accept tolerance changes without evidence |
| CI triager | Fresh context from failing CI logs and PR diff | Minimal fix or diagnosis | Hide failures by skipping tests without approval |
| Doc gardener | Fresh context from merged code and docs | Documentation consistency updates | Change behavior |
| Release guard | Fresh context from release issue and workflows | Release readiness evidence | Publish without protected approval |

One person or account may run multiple roles, but each role must use a fresh session and fresh worktree. Metadata must show the separation once Phase 2 is enabled.

## Code-Changing Workflow

1. Create or select a GitHub issue.
2. Scout writes `docs/exec-plans/active/<issue>-scout.md`.
3. Test author starts in a clean worktree at the base commit.
4. Test author reads only the issue, `AGENTS.md`, linked docs, public API docs, existing tests, and scout affected-area map.
5. Test author writes tests and fixtures only.
6. Test author proves tests fail on the base commit for the intended reason.
7. Test author commits to a red-test branch.
8. Implementer starts in a separate clean worktree at the same base commit.
9. Implementer applies only the red-test commit or patch.
10. Implementer changes product code until required tests pass.
11. Implementer may not delete, skip, xfail, loosen, or rewrite red tests.
12. If a red test is wrong, use the test-amendment protocol.
13. Adversarial reviewer reviews the final PR diff from a fresh context.
14. Numerics reviewer reviews changes that touch numerical algorithms, scientific assumptions, fixtures, tolerances, image processing, hardware behavior, or public API behavior.
15. PR cannot be marked ready until required metadata is present and CI passes.

## Test-Amendment Protocol

If red tests are wrong:

1. Stop implementation.
2. Write `docs/agent-harness/test-amendments/<issue>.md`.
3. Explain the incorrect assertion, missing assumption, or invalid fixture.
4. Propose corrected tests.
5. Request a fresh test-review agent or human decision.
6. Resume only after approval is recorded.
