# Workflow examples

**Version 1.0** These are examples, not executed runs or approved product requirements.

## Empty repository

Request: “Build a local tool that counts non-empty lines in a file.”

Intake first settles the supported environment, character encoding, whitespace meaning, output, and error behavior. It proposes a small stack and `docs/PROJECT.md`; the owner approves. The first task then defines exact behavior, such as whether invalid UTF-8 is an error. Architecture approval alone does not authorize the feature.

A writes tests that invoke the real command and inspect its output/exit code for valid, empty, whitespace-only, unreadable, and invalid input. B reviews expectations and any custom parser's decision logic: it must accept allowed output and reject malformed output. After valid feature-absence evidence and a local test checkpoint, C implements without changing tests. Full canonical verification must pass on the exact candidate before pushing; CI repeats it on configured platforms. D reviews the passing candidate and results. Green CI and D's acceptance permit presenting the PR for normal merge.

A later request to read standard input needs task intake, not another full setup.

## Existing web application

Request: “Export my customers as CSV.”

Before adopting the system, discovery inventories existing failures, exact measured coverage, environment problems, and unresolved behavior. The owner sees separate scopes and effort for installing tooling and repairing the existing product; installing the tools does not satisfy the 100% coverage requirement.

Intake resolves who “my” includes, allowed fields, escaping, download behavior, and failure outcomes. A uses browser journeys through the actual app and a seeded test database. One scenario checks a user cannot export another organization's records. Smaller tests cover expensive input combinations only where useful. A mock-only browser test is not evidence that the server enforces access.

C cannot fix a failing export test by editing its expected CSV. A broken database connection goes to authorized environment/tooling diagnosis, then the interrupted role resumes. A test parser that accepts malformed CSV returns to fresh A/B and a revised checkpoint. A valid test exposing an approved export defect goes to C; missing field-selection intent returns to intake. Only the valid product failure supplies product-red evidence.

Each role receives a narrow packet and an explicit completion condition. Related cases share a checkpoint. One Current state section records approvals, checkpoint, candidate, checks, repair/budget use, blockers, and next action. Replace superseded status; Git retains history. Renaming the task or replacing the checkpoint does not reset the repair allowance.

## Numerical contract

Suppose an API fits a power law, but its intercept units are unspecified. A must not choose counts versus density merely to cover the intercept branch. Intake resolves that choice. An approved scale-ratio invariant may support a narrower test, with the unresolved absolute interpretation recorded. B checks that the numerical oracle accepts legitimate alternatives and rejects plausible errors without introducing an undocumented solver stopping priority.

## Learn, then improve the spec

Suppose an E2E test fails because the test process does not wait for the application to become ready. The agent records an entry in `LESSONS.md`, with a reproducible failure and the readiness evidence—not “tests are flaky.”

Use `delivery doctor` once installed, or its prompt beforehand. It classifies the failure and distinguishes an instruction gap from failure to follow an existing rule. If readiness is already covered, it reports the test or environment/tooling defect without adding another rule. For an evidenced gap, it writes the smallest correction to the spec on a docs branch. The owner reviews and approves; Git records the change. The document still says version 1.0.

Doctor never fixes the red build by changing tests or dropping a requirement. That repair is separate authorized work.
