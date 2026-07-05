# Implementation Notes

## Phase 0 Decisions

- Product code under `dphtools/` was not changed.
- Packaging backend was not migrated.
- A `requirements-dev.txt` file was added because no dev extra or dev requirements file existed.
- CI preserves the existing Python 3.10 runtime and existing Ubuntu/macOS/Windows OS matrix.
- CI removes the previous push path filter and exposes `ci-required`.
- Release workflow behavior is unchanged.

## Claude Code Artifacts

`.claude/` artifacts are not added because the installed Claude Code schema was not validated in this environment.

Future work may add skills, subagents, hooks, and workflows after validation against the installed Claude Code version. Until then, repository docs and CI are authoritative.

## Phase 2 Enforcement

`docs/agent-harness/enforcement.json` sets the active harness phase. Pull request validation now enforces clean-context metadata for product source changes during GitHub pull request events.

Local `make harness-check` remains usable outside a pull request because local GitHub event metadata is unavailable.

## Known Follow-Up Issues

1. Configure branch protection for `main`.
2. Run a clean-context harness smoke issue.
3. Establish measured coverage baseline and ratchet policy.
4. Inventory public API and domain invariants.
5. Audit fixtures and generated data.
6. Align runtime support metadata and CI matrix.
7. Add downstream smoke tests if consumers are discovered.
8. Harden release workflow with protected environments and human approval.

## Local Verification Notes

On 2026-07-04, `python -m pytest -q tests` was run with local Python 3.13 and NumPy 2.5.1 after installing `requirements.txt` and `requirements-dev.txt`.

Result: 23 passed, 12 failed, 3 warnings.

Failure pattern: all failures are in existing `split_img` tests because `dphtools/utils/__init__.py` calls `np.product`, which NumPy 2.x removed. Phase 0 does not change product source, so this is documented as a pre-existing compatibility failure for a follow-up product issue.
