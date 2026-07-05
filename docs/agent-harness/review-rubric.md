# Review Rubric

Review agent-produced changes as if the implementation may be subtly wrong even when CI is green.

## Required Checks

- The issue scope is clear and the PR links it.
- The PR states phase, risk, and product behavior impact.
- Tests assert behavior, not implementation accidents.
- Red tests fail on base for the intended reason when product code changes.
- Implementation did not weaken, skip, delete, or loosen tests.
- CI workflows are not bypassed by path filters.
- Dependency changes are justified and documented.
- Numerical tolerances have an oracle or reviewer-approved evidence.
- Fixtures are deterministic, minimal, and documented.
- Public API changes are documented.
- Release and security changes have human/admin approval.

## Findings Format

List blocking findings first, then non-blocking findings. Include file and line references when possible. End with confidence and what would change the conclusion.
