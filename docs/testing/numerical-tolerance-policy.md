# Numerical Tolerance Policy

Prefer analytic or independent oracles.

Use explicit relative and absolute tolerances. Explain the tolerance source:

- analytic bound
- platform noise
- empirical measurement
- legacy baseline

Do not loosen a tolerance in the same pull request that changes implementation unless a numerics reviewer approves it.

For image or signal tests, assert:

- shape
- dtype
- finite values
- monotonic or physical invariants where applicable
- conservation or normalization where applicable
- border behavior where applicable

Store golden arrays only when synthetic or analytic tests are insufficient. Hash and document binary fixtures.
