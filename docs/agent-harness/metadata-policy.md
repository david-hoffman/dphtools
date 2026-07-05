# Agent Run Metadata Policy

Phase 2 is enabled.

Pull requests with product source changes under `dphtools/` must include:

- A linked issue in the pull request body.
- `test-author` run metadata.
- Red-test proof for the linked issue.
- `implementer` run metadata.
- `adversarial-reviewer` run metadata.
- `numerics-reviewer` run metadata when numerical, scientific, image-processing, fitting, or signal code is touched.

Run metadata files live under:

```text
docs/agent-harness/runs/<issue-number>/<timestamp>-<role>.json
```

Red-test proof files live under:

```text
docs/agent-harness/red-test-proofs/<issue-number>.md
```

The implementer role must not list test paths in `allowed_paths` unless a test-amendment note exists under:

```text
docs/agent-harness/test-amendments/<issue-number>.md
```

High-risk changes to release, security, package governance, branch governance, or protected harness files must use `risk:high` and document the human/admin decision in the pull request body.
