# Confirm that a public witness reaches the measured coverage gap

- ID: 20261006T054452Z-SPECTRUM-FITTING-001-coordinator-measured-gap-witness
- Date: 2026-10-06T05:44:52Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: Two public numerical-covariance failures had different outcomes. The first supplied witness exercised a nonfinite Jacobian but left the originally missing statement and branch unmeasured.
- Evidence: C3 full receipt full-hq3ox68t at candidate 1c72cadf8f31a7a5925b06eb7183aa5319e5fb52: 3338/3339 statements and 857/858 branches; retained focused2 coverage still missed _spectrum_fit.py line67/branch66→67, while steps-boundary-private-measurement.json and focused3 confirmed that outcome was exercised.
- Lesson: Before handing a blind test author a semantic witness, privately confirm that the public reproduction exercises the actual missing outcome. Keep implementation and raw measurement outside the blind packet.
