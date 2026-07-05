# Issue #6 CI Triage

## Failing checks

- `harness-validate`: stale pull request event labels and over-broad per-product-issue role requirements.
- OS matrix doctests: NumPy scalar repr changes in `dphtools.utils` examples.

## Fix plan

- Refresh PR body and labels from the current GitHub issue API during CI validation when available.
- Separate PR-level high-risk, CI, and release role evidence from product-source clean-context evidence.
- Track the product doctest compatibility fix under issue #8.
