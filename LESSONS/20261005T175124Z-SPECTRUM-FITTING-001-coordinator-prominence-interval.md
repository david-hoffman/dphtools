# Sequence prominence can hide a scalar-validation defect

- ID: 20261005T175124Z-SPECTRUM-FITTING-001-coordinator-prominence-interval
- Date: 2026-10-05T17:51:24Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: SciPy accepts a two-value prominence sequence as an interval. A
  nominal invalid-scalar test using an interval that removes every peak can pass
  through the separate no-peaks ValueError even if the fitter ignores scalar
  validation.
- Evidence: B round 1 found this in S23; the historical A report records its
  correction. A source-free SciPy 1.18.1 probe with
  `x=linspace(-5,5,101); y=3*exp(-x*x/2)` returned zero peaks for `[1,2]` and one
  for `[0,10]`. The [primary definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.find_peaks.html)
  documents interval prominence. No product mutation was needed.
- Lesson: For scalar-only wrapper validation, choose a sequence that the
  underlying API would accept and that preserves an otherwise valid peak. Keep
  unrelated sample, shape, ordering, and discovery preconditions valid.
