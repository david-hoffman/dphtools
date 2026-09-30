# Workflow examples

**Version 1.0** These are examples, not executed runs or approved product requirements.

## Empty repository

Request: “Build a local tool that counts non-empty lines in a file.”

Intake first settles the supported environment, character encoding, whitespace meaning, output, and error behavior. It proposes a small stack and `docs/PROJECT.md`; the owner approves. The first task then defines exact behavior, such as whether invalid UTF-8 is an error. Its scenario table contains five cases: valid, empty, whitespace-only, unreadable, and invalid input. The owner approves this single task; no separate slice-plan document is needed. Architecture approval alone does not authorize the feature.

A writes tests that invoke the real command and inspect its output/exit code for those scenarios. B reviews expectations and any custom parser's decision logic: it must accept allowed output and reject malformed output. After valid feature-absence evidence and a reviewed test checkpoint, C implements without changing tests. A pre-PR push may back up the failing checkpoint. Full canonical verification must pass on the exact candidate before opening/reopening a PR, including a draft, or pushing an update to an open PR. CI repeats verification on configured platforms. D reviews the passing candidate and results. Green CI and D's acceptance permit presenting the PR for normal merge.

A later request to read standard input needs task intake, not another full setup.

## Existing web application

Request: “Export my customers as CSV.”

Before adopting the system, discovery inventories existing failures, exact measured coverage, environment problems, and unresolved behavior. The owner sees separate scopes and effort for installing tooling and repairing the existing product. If a five-scenario slice cannot bring a legacy baseline to full verification and global 100% coverage, report that dependency now. The owner chooses adoption scope case by case and may approve a larger slice; D does not accept a failing slice as a workaround.

Intake resolves who “my” includes, allowed fields, escaping, download behavior, and failure outcomes. The owner then approves the complete end-to-end slice plan and contracts together. Each ordinary slice task has one checkpoint lineage and defaults to at most five distinct scenarios in its contract table. Independent slices can use separate tasks/worktrees. A dependent slice waits for its prerequisites to be completed and integrated; separate files alone do not establish independence.

A uses browser journeys through the actual app and a seeded test database. One scenario checks a user cannot export another organization's records. Smaller tests cover expensive input combinations only where useful. Existing approved behavior may already pass these tests, including during remediation. Record that result; do not mutate the product or force a red result. C may confirm that no product edit is needed. A claimed access-control bug still requires the intended failure to establish reproduction.

C cannot fix a failing export test by editing its expected CSV. A broken database connection goes to authorized environment/tooling diagnosis, then the interrupted role resumes. A test parser that accepts malformed CSV returns to fresh A/B and a revised checkpoint. A valid test exposing an approved export defect goes to C; missing field-selection intent returns to intake. Only the valid product failure supplies product-red evidence.

Each role receives a narrow packet and an explicit completion condition. If B cannot accept after two reviews, B diagnoses missing contract decisions or identifies real split points in an oversized slice. Clarification returns to intake; redistributing unchanged approved scenarios needs no renewed approval. Resume only under the diagnosis-resolution rule in specification section 9. Splitting preserves history, total budget, and used C repairs.

One Current state section records approvals, checkpoint, candidate, checks, blockers, and next action. Its compact metrics line records scenario count, A/B rounds, C repair use, and spend when known. A/B review rounds and implementation repairs are separate; changing a task name or checkpoint does not replenish consumed resources.

## Numerical contract

Suppose an API fits a power law, but its intercept units are unspecified. A must not choose counts versus density merely to cover the intercept branch. Intake resolves that choice. An approved scale-ratio invariant may support a narrower test, with the unresolved absolute interpretation recorded.

A chooses and justifies absolute and/or relative tolerances from the permitted algorithm, mathematical, and precision information. A does not inspect implementation or widen tolerances to fit observed output. B checks both false-positive and false-negative boundaries: plausible wrong results must fail, and legitimate alternatives must pass. Neither role may lower the contract or impose an undocumented solver stopping priority.

## Learn, then improve the spec

Suppose an E2E test fails because the test process does not wait for the application to become ready. The agent creates a new timestamped Markdown file in `LESSONS/`, with a reproducible failure and the readiness evidence—not “tests are flaky.”

Use `delivery doctor` once installed, or its prompt beforehand. It classifies the failure and distinguishes an instruction gap from failure to follow an existing rule. If readiness is already covered, it reports the test or environment/tooling defect without adding another rule. The compact scenario/round/repair/spend metrics may also support a proposed slice-size adjustment. Doctor cannot activate that adjustment autonomously. The owner reviews an evidenced patch; Git records the approved change. The document still says version 1.0.

Doctor never fixes the red build by changing tests or dropping a requirement. That repair is separate authorized work.
