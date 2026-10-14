# Numerical observers must preserve permitted implementations

- ID: 20261005T200454Z-SPECTRUM-FITTING-001-coordinator-numerical-observer-boundaries
- Date: 2026-10-05T20:04:54Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: Dependency-failure observers rejected valid stacked/complex SVD
  operands, and a covariance observer assumed float64 despite no output-dtype
  contract. Real physical outputs do not require real-only internal operands.
- Evidence: B6 reproduced valid stacked `(1,201,3)` rejection; the coordinator's
  complex128 decomposition matched analytic real covariance at rtol1e-12 while
  the old observer rejected it. B7 accepted corrected tests SHA-256
  50bb168312355b5c3035f534195260f4a0081a6f171cdd6a6de52235774837dd
  after46 independent checks. Native wider-range execution was unavailable on
  this ARM host; higher-range reasoning remains explicitly distinguished from
  native execution in the historical test report.
- Lesson: Fault observers should constrain the public contract rather than
  numerical backend representation. Determine representability from the actual
  output dtype. Check legitimate alternate algorithms as well as wrong outputs.
