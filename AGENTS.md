# AGENTS.md

## Project snapshot
- Project purpose: `dphtools` provides tools for optics and image analysis.
- Main package/source directories: `dphtools/`, with scientific utilities in `dphtools/utils/`.
- Test directories: `tests/`.
- Default branch: `main`.
- Supported runtime versions: Python `>=3.8` in package metadata; existing CI only tests Python 3.10.

## Required first reads
1. `docs/generated/repo-intake.md`
2. `docs/agent-harness/workflow.md`
3. `docs/agent-harness/clean-context-protocol.md`
4. `docs/testing/numerical-tolerance-policy.md`, when numerical/scientific code is touched
5. `ARCHITECTURE.md`
6. Relevant nested `AGENTS.md` files, if any

## Golden rules
- Do not push to the default branch.
- Do not weaken tests to make implementation pass.
- Do not change generated files by hand.
- Do not update numerical tolerances without reviewer evidence.
- Do not change release workflows or secrets without a release issue and human/admin approval.
- Do not change public API without a design doc and changelog entry.
- Label facts, assumptions, and guesses separately.

## Local commands
- Bootstrap: `make bootstrap`
- Fast tests: `make test-fast`
- Full check: `make check`
- Coverage: `make coverage`
- Harness validation: `make harness-check`

## Agent workflow
- For code changes, use the clean-context test-first workflow.
- Test author and implementation agent must be separate sessions.
- Adversarial review must run before a pull request is marked ready.
- Each pull request must include agent run metadata after Phase 2 is enabled.

## CI policy
- `ci-required` must pass before merge.
- Required workflows must not use path filters that can skip checks.
- Fix root causes of CI failures; do not skip tests without a linked issue.

## Hardware/scientific policy
- State units where units matter.
- Prefer analytic or synthetic oracles where possible.
- Use deterministic seeds for randomized tests.
- Record fixture provenance and checksums.
