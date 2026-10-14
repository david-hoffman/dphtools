# A positive physical width can yield an unrepresentable difference step

- ID: 20261005T185328Z-SPECTRUM-FITTING-001-coordinator-positive-width-difference-step
- Date: 2026-10-05T18:53:28Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: The smallest positive float is an approved positive finite width, but halving it for a central-difference step rounds to zero. Gaussian and Lorentzian public fits leaked LinAlgError during uncertainty calculation, even after every new-helper statement and branch had been measured by ordinary fixtures.
- Evidence: Candidate c7ee4099e5276c9218a2d5bd90891ba7f017c6c2; clean Python 3.12.14. A 309-case focused diagnostic measured the helper's 155/155 statements and 66/66 branches. A separate public call with x=linspace(-1,1,41), y[20]=2 and other samples zero, full guess [2,0,nextafter(0,1)], and no background returned LinAlgError for both profiles. Inputs stayed unchanged. R3 accepts a verified result or RuntimeError for numerical failure, so the leaked ValueError subclass is a product defect. Raw public probe and interrupted full evidence are retained in the task conversation's external evidence directory.
- Lesson: Check representability of numerical perturbations as well as physical parameter positivity. Exercise valid domain limits independently, preserve numerical warnings, and honor the public failure-type boundary. Complete measured branch coverage does not establish behavior at every floating-point magnitude.
