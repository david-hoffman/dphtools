# Workflow examples

**Version 1.0** These are examples, not executed runs or approved product requirements.

## Empty repository

Request: “Build a local tool that counts non-empty lines in a file.”

Intake first settles the supported environment, character encoding, whitespace meaning, output, and error behavior. It proposes a small stack and `docs/PROJECT.md`; the owner approves. The first task then defines exact behavior, such as whether invalid UTF-8 is an error. Architecture approval alone does not authorize the feature.

A writes tests that invoke the real command and inspect its output/exit code for valid, empty, whitespace-only, unreadable, and invalid input. B reviews the expectations. After meaningful failing evidence and the test commit, C implements without changing tests. CI runs checks. D reviews the candidate and results. The PR merges normally.

A later request to read standard input needs task intake, not another full setup.

## Existing web application

Request: “Export my customers as CSV.”

Intake resolves who “my” includes, allowed fields, escaping, download behavior, and failure outcomes. A uses browser journeys through the actual app and a seeded test database. One scenario checks a user cannot export another organization's records. Smaller tests cover expensive input combinations only where useful. A mock-only browser test is not evidence that the server enforces access.

C cannot fix a failing export test by editing its expected CSV. A suspected test defect returns to A/B. A task note identifies the test and implementation commits plus CI results; no separate evidence service is involved.

## Learn, then improve the spec

Suppose an E2E test fails because the test process does not wait for the application to become ready. The agent records an entry in `LESSONS.md`, with a reproducible failure and the readiness evidence—not “tests are flaky.”

Run `delivery doctor`. If the spec already addresses readiness, doctor reports a test/workflow defect and does not add another rule. If a real uncovered process gap exists, it writes the smallest relevant correction to the spec on a docs branch. The owner reviews and approves; Git records the change. The document still says version 1.0.

Doctor never fixes the red build by changing tests or dropping a requirement. That repair is separate authorized work.
