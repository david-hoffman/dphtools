### 2026-09-28-SETUP-001-intercept-units | Separate shape from units
- Status: confirmed
- Observation: the coordinator proposed an absolute power-law intercept assertion using normalized density, although the public docstring leaves the level's units unspecified. That draft assertion was withdrawn before the reviewed checkpoint.
- Evidence: `docs/tasks/SETUP-001-intake.md` records the count-versus-density ambiguity; `reports/B-coverage-continuation/final-review.md` accepts amplitude-independent ratios and retains the unresolved units. No product fix was made for the withdrawn assertion.
- Lesson: derive only the invariants supported by the contract while units remain undecided. Executing a method and checking its shape does not validate an unspecified absolute interpretation.
