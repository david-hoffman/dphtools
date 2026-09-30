### 2026-09-28-SETUP-001-B-stopping-oracle | Numerical test contracts
- Status: confirmed
- Observation: a passing singular-solver test wrongly excluded objective convergence and allowed a gradient-convergence status while that check was disabled.
- Evidence: fresh B session `01a0e6d5-55b2-7082-8b8a-cbca532c6d8b` identified the contract mismatch. A corrected only the assertion/comment; B independently passed all 123 solver cases and accepted checkpoint `b14298f`.
- Lesson (returned by B): when several documented stopping predicates can hold at the same accepted point, test returned state and actual callback counts without imposing an undocumented stopping-priority order.
