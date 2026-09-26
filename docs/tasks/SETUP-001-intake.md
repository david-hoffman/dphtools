# SETUP-001: Scientific maintenance intake

**Version 1.0** Pending owner decisions. This is a task intake record, not a replacement delivery specification or an approved numerical contract.

The owner approved compatibility maintenance and baseline tests while preserving public APIs and numerical intent. Independent supplemental test review found the following scope questions. The 100% statement/branch requirement remains unchanged.

## Proposed scope extension

Implement the explicitly documented solver capabilities currently rejected by public calls: the `pyls` method, vector and covariance-matrix weighting for `ls`, numerical Jacobians when `Dfun` is absent, and row-oriented derivatives when `col_deriv=False`. Repair verified mathematical/API discrepancies where the existing public contract determines the answer: histogram statistics, relative covariance scaling, custom drift coordinate names, and normalized rigid registration. No new public interface or speculative algorithm is proposed.

Evidence comes from the approved [public packet](SETUP-001-public-api.md), supplemental tests, and fresh B session `01a0e01d-78bc-7ce1-bba2-43e30a35914d`. For example, multiplying histogram counts `[1, 2, 1]` at locations `[0, 2, 4]` by seven changes returned variance from 2 to 14; the represented distribution is unchanged. The documented relative covariance formula predicts `diag(3, 2)` for a known linear residual, while the Python `ls` result is `diag(1/2, 1/3)`.

This extension was asked asynchronously in the setup conversation. Approval has not yet been recorded. Compatible replacement of removed NumPy/Matplotlib APIs remains within the already approved scope.

The coordinator verified the upstream replacements against [NumPy's migration guide](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html) (`product` → `prod`) and [Matplotlib's removal notes](https://matplotlib.org/3.3.0/api/prev_api_changes/api_changes_3.3.0/removals.html) (`cbook.iterable` → `numpy.iterable`) on 2026-09-26.

## Undefined contracts still requiring decisions

- `fit_ztp` names a zero-truncated Poisson model without specifying an estimator. A maximum-likelihood assertion was rejected by B rather than imposed on the implementation.
- Fitted quadratic uncertainties need an estimation/covariance convention. Coefficients and centers can be tested independently; zero fitted uncertainty is not presumed.
- Power-law fitting and percentiles need parameter, objective, normalization, and model-selection definitions.
- LPSVD parameters and uncertainty outputs need units, phase/damping conventions, and a noise model; successful signal reconstruction alone does not establish these.
- Several filter boundaries, transform composition/weight choices, and handwritten error/demo paths lack enough public behavior to justify exact expectations. These are test-contract gaps, not permission to add exclusions or assert incidental implementation results.

The supplemental tests can establish defined behavior while these gaps remain explicit. Any further numerical contract must be recorded and approved before it drives product changes. A green suite or a passing local helper alone cannot establish setup readiness while coverage and contract gaps remain.
