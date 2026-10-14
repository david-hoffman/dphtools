# Numerical stress validity does not establish physical acceptance scope

- ID: 20261005T214425Z-SPECTRUM-FITTING-001-coordinator-physical-acceptance-scope
- Date: 2026-10-05T21:44:25Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: A covariance regression using a 1e12-data-unit offset, a height-three peak and noise around 0.01 data units was mathematically resolved but did not establish a physically representative measurement. The owner excluded this extreme offset from required acceptance.
- Evidence: Historical checkpoint e5bc47e1448c5466f981cfb4dd29150e8539552f has two failing extreme-offset cases; public contract R4 records the owner's physical-plausibility clarification and retains ordinary fit/covariance requirements. The original numerical finding remains recorded in the 20261005T205314Z background-covariance lesson.
- Lesson: Establish units, background provenance and a plausible noise model before making a stress fixture a required scientific acceptance case. Keep robustness checks distinct from representative fit and uncertainty checks; a local covariance oracle does not validate repeated-measurement confidence intervals. An owner scope correction must retain the original evidence and receive fresh test review.
