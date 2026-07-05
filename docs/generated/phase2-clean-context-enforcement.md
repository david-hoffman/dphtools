# Phase 2 clean-context enforcement evidence

Date: 2026-07-05
Repository: `david-hoffman/dphtools`

## Facts

- `docs/agent-harness/enforcement.json` sets active harness phase to 2.
- `validate_pr.py` now enforces clean-context metadata during GitHub pull request events.
- Local `make harness-check` remains usable without fabricated pull request metadata.
- Required GitHub labels were created or updated with `gh label create --force`.

## Pull request enforcement

For pull requests with product source changes under `dphtools/`, CI validation requires:

- Linked issue in the PR body.
- Exactly one risk label.
- Passed `test-author` run metadata.
- Red-test proof for the linked issue.
- Passed `implementer` run metadata.
- Passed `adversarial-reviewer` run metadata.
- Passed `numerics-reviewer` metadata when numerical, scientific, image-processing, fitting, or signal code is touched.

High-risk governance, packaging, release, security, and protected harness changes require:

- `risk:high`.
- A human/admin decision note in the PR body.

## Test-amendment enforcement

`implementer` run metadata may not list test paths in `allowed_paths` unless:

```text
docs/agent-harness/test-amendments/<issue-number>.md
```

exists for the linked issue.

## Validation commands

```bash
make harness-check
python -m black --check -l 99 scripts
python scripts/agent_harness/validate_agent_run.py
```

## Limits

GitHub PR validation relies on GitHub pull request event JSON. It does not make local `--local` runs fail for missing PR body, labels, or event-only metadata.
