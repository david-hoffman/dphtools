# Workflow examples

**Version 1.0** These are examples, not executed runs or approved product requirements.

## Empty repository

Request: “Build a local tool that counts non-empty lines in a file.”

Intake first settles the supported environment, character encoding, whitespace meaning, output, and error behavior. It proposes a small stack and `docs/PROJECT.md`; the owner approves. The first task then defines exact behavior, such as whether invalid UTF-8 is an error. Its scenario table contains five cases: valid, empty, whitespace-only, unreadable, and invalid input. The owner approves this single task; no separate slice-plan document is needed. Architecture approval alone does not authorize the feature.

This is routine work: the output contract is straightforward and has no new scientific or safety oracle. The author writes meaningful real-command tests and the smallest implementation, reusing existing project context. After focused checks and the highest owner-approved project tier pass on the exact candidate, the author may open/reopen a PR (including a draft) or update it. One fresh independent reviewer checks expectations, actual results, scope, and simplicity without inheriting the author's conversation. Full local verification is useful as a reference or for diagnosis when warranted. Merge waits for the unchanged reviewed candidate's complete required platform CI, exact 100% coverage in every required whole owned domain (fresh global/per-package for full), verified native required-check protection, and owner authorization. A pre-PR backup may retain an incomplete checkpoint but is not readiness evidence. If protection is unavailable, report the merge blocker; green workflow text does not supply it.

A later request to read standard input needs task intake, not another full setup.

## Existing web application

Request: “Export my customers as CSV.”

Before adopting the system, discovery inventories existing failures, exact measured coverage, environment problems, and unresolved behavior. The owner sees separate scopes and effort for installing tooling and repairing the existing product. If a five-scenario slice cannot bring a legacy baseline to full verification and global 100% coverage, report that dependency now. The owner chooses adoption scope case by case and may approve a larger slice; D does not accept a failing slice as a workaround.

Intake resolves who “my” includes, allowed fields, escaping, download behavior, and failure outcomes. A changed cross-organization authorization boundary presents genuine data-safety risk, so this slice uses specialist A/B/C/D; an ordinary CSV formatting repair under an established contract need not. The owner approves the slice plan and contracts together. New contracts default to at most five distinct scenarios; specialist slices have one checkpoint lineage. Independent slices can use separate tasks/worktrees. Dependent slices wait for completed, integrated prerequisites; separate files alone do not establish independence.

A uses browser journeys through the actual app and a seeded test database. One scenario checks a user cannot export another organization's records. Smaller tests cover expensive input combinations only where useful. Existing approved behavior may already pass these tests, including during remediation. Record that result; do not mutate the product or force a red result. C may confirm that no product edit is needed. A claimed access-control bug still requires the intended failure to establish reproduction.

C cannot fix a failing export test by editing its expected CSV. A broken database connection goes to authorized environment/tooling diagnosis, then the interrupted role resumes. A test parser that accepts malformed CSV returns to fresh A/B and a revised checkpoint. A valid test exposing an approved export defect goes to C; missing field-selection intent returns to intake. Only the valid product failure supplies product-red evidence.

Each role receives a narrow packet and an explicit completion condition. If B cannot accept after two reviews, B diagnoses missing contract decisions or identifies real split points in an oversized slice. Clarification returns to intake; redistributing unchanged approved scenarios needs no renewed approval. Resume only under the diagnosis-resolution rule in specification section 9. Splitting preserves history, total budget, and used C repairs.

One Current state section records approvals, checkpoint, candidate, checks, blockers, and next action. Its compact metrics line records scenario count, A/B rounds, C repair use, and spend when known. A/B review rounds and implementation repairs are separate; changing a task name or checkpoint does not replenish consumed resources.

The risk/check plan calls for full local reference verification before submission. The merge still needs fresh D acceptance of the unchanged candidate and full required platform CI under verified protection. Raw logs stay outside Git; narrow packets and concise evidence links avoid replaying whole sessions. Host-required progress updates remain mandatory.

## Numerical contract

Suppose an API fits a power law, but its intercept units are unspecified. A new scientific contract/custom numerical oracle requires specialist A/B/C/D. A must not choose counts versus density merely to cover the intercept branch. Intake resolves that choice. An approved scale-ratio invariant may support a narrower test, with the unresolved absolute interpretation recorded. An unrelated documentation repair may reuse this settled contract on the routine route.

A chooses and justifies absolute and/or relative tolerances from the permitted algorithm, mathematical, and precision information. A does not inspect implementation or widen tolerances to fit observed output. B checks both false-positive and false-negative boundaries: plausible wrong results must fail, and legitimate alternatives must pass. Neither role may lower the contract or impose an undocumented solver stopping priority.

## Learn, then improve the spec

Suppose an E2E test fails because the test process does not wait for the application to become ready. The agent creates a new timestamped Markdown file in `LESSONS/`, with a reproducible failure and the readiness evidence—not “tests are flaky.”

Use `delivery doctor` once installed, or its prompt beforehand. It classifies the failure and distinguishes an instruction gap from failure to follow an existing rule. If readiness is already covered, it reports the test or environment/tooling defect without adding another rule. The compact scenario/round/repair/spend metrics may also support a proposed slice-size adjustment. Doctor cannot activate that adjustment autonomously. The owner reviews an evidenced patch; Git records the approved change. The document still says version 1.0.

Doctor never fixes the red build by changing tests or dropping a requirement. That repair is separate authorized work.
