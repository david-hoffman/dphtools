# Literal configuration names can select different inputs

- ID: 20261005T095459Z-DELIVERY-EFFICIENCY-001-reviewer-implicit-coverage-selector
- Date: 2026-10-05T09:54:59Z
- Task/role: DELIVERY-EFFICIENCY-001 / reviewer
- Status: confirmed
- Observation: Pinned coverage treats literal `.coveragerc` as an implicit selector honoring `COVERAGE_RCFILE`; resolving it to an absolute filename changes selection semantics.
- Evidence: `/tmp/review-startup-sentinel-eejr_7f0/proof-controlled.json` records original PASS, reused PASS, and fresh FAIL when only an external plugin flag changed. The working eligibility guard parsed the plain root file while actual startup selected the unchanged plugin configuration. Pinned `coverage/config.py` defines this sentinel in `config_files_to_try`.
- Lesson: Decline implicit or forced startup selection when its complete inputs are unknown. Preserve inline precedence and explicit-file measurement; use a real command regression rather than a different parser's apparent agreement.
