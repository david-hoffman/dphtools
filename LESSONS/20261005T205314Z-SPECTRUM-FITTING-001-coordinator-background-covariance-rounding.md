# Large background offsets can corrupt finite covariance

- ID: 20261005T205314Z-SPECTRUM-FITTING-001-coordinator-background-covariance-rounding
- Date: 2026-10-05T20:53:14Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: A well-conditioned Gaussian fit with a constant background of 1e12 data units returned finite covariance with sigma variance about 15.3 times too small and no warning, despite complete measured coverage.
- Evidence: Independent D public reproduction on commit820e17dc31d9121b7b1ff2f5c839c0e77d8ad70f, /private/tmp/dphtools-spectrum-D-review-bxbulkvc/covariance_probe.py; coordinator reproduction exit1. Expected sigma variance5.77897e-7 x-unit squared; returned3.77959e-8. Normalized Jacobian condition3.257. Zero-background control agrees. Evidence applies to local CPython3.12.14/macOS ARM; hosted results are not established.
- Lesson: For covariance regressions, vary the background magnitude and compare the complete physical covariance with independently derived derivatives and actual RSS/(N-P). Finiteness and rank alone do not establish derivative accuracy after subtraction of large offsets. Preserve permitted numerical failure rather than accepting misleading finite uncertainty.
