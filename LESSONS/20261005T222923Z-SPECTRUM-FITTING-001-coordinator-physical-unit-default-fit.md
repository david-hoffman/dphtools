# Physical unit conversion can expose an unfinished default fit

- ID: 20261005T222923Z-SPECTRUM-FITTING-001-coordinator-physical-unit-default-fit
- Date: 2026-10-05T22:29:23Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: The resolved R4 spectrum fitted correctly in nanometres, but default LM returned the initial peak guesses as successful when the same coordinates and widths were converted to metres. Its covariance still matched the local recipe at the unfinished parameters.
- Evidence: Candidate f63ebf2de3fc65c047b00a3631f84b8bf0c68437 passed 2118 tests and full owned coverage. Independent D2 public reproduction found RSS 1.19075 V² in metres versus 0.0170722 V² for nanometre/default and metre/realcustom fits; a feasible generating model had RSS 0.0172294 V². The observed center perturbation was 14.9 nm against a 0.85 nm starting width. The narrow public unit-gap packet and complete independent receipts remain in the task conversation.
- Lesson: Include physically identical unit-converted inputs in scientific acceptance. Verify fit quality and stationarity alongside covariance; a solver success flag and mathematically consistent local covariance do not establish that optimization finished.
