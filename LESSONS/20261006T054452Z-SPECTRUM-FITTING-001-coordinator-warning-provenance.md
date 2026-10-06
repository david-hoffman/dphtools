# Keyword matching does not establish numerical warning relevance

- ID: 20261006T054452Z-SPECTRUM-FITTING-001-coordinator-warning-provenance
- Date: 2026-10-06T05:44:52Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: A test diagnostic classifier accepted display-only warnings containing fit/optimizer words and rejected conventional informative linear-algebra failures.
- Evidence: B13 public review /private/tmp/dphtools-spectrum-B17-sqf9i07c/review.md and independent-diagnosis-diagnostics.json: Fit legend unavailable and Optimizer progress display failed were accepted, while Singular matrix and SVD did not converge were rejected. Tests checkpoint 173f8958ff2d2fc35508226143c83876197a80b755254bc0924dc4fc1c56261d was not accepted.
- Lesson: Test warning preservation through known caller-controlled numerical provenance when the contract permits it. Do not claim a generic English substring classifier establishes arbitrary diagnostic relevance; review the actual numerical context separately.
