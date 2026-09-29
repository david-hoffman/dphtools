# Agentic Software Delivery System

**Version: 1.0** Git commits version edits; do not bump the release number while designing this system. This is the current specification, not installed software. Earlier conversation drafts are superseded, not released versions.

> One monorepo. Clarify first. Independent agents test, implement, and review. Ordinary CI checks the result. Record lessons and improve the instructions.

## 1. Keep the system small

Use an existing coding harness and four skills. Agents write all product/helper code and tests; the owner supplies intent and approvals, not engineering. Keep product code, tests, delivery instructions, and GitHub Actions in the same repository. Run one task at a time. Use native harness sessions, Git commits, Markdown, and existing test tools; add only small launch scripts where necessary.

**Do not build a delivery platform:** no second repository, custom controller/state engine, GitHub App, external merge authority, permission broker, signed attestations, immutable evidence store, agent swarm, or tracing service. Fresh role sessions and ordinary CI remain; custom enforcement of role/file restrictions does not.

The owner accepts that “do not edit tests/workflows” and “append only” are prompt rules. Agents technically retain access. A workflow change, false report, or ignored instruction may defeat this process. GitHub checks only the configured CI, not whether agents obeyed these rules. Do not describe this design as tamper-proof or guaranteed to prevent bugs.

**Names:** the delivery system is the whole process; a coding harness runs an agent; a role is one session's responsibility; a skill is its procedure. No custom orchestrator is required.

**Normative language:** MUST is required behavior, not a claim of mechanical enforcement. This file defines policy and rationale; repository `AGENTS.md` is the single operational home for shared instructions. Skills contain role-specific procedures and refer to that home. Supporting prompts and generation templates must agree; they are not additional live rulebooks.

## 2. Setup and architecture intake

Start read-only. Inspect existing instructions, manifests, interfaces, tests, and documentation. Before proposing remediation, produce a small baseline inventory: existing failures, exact measured statement/branch coverage and its scope, environment/tooling problems, and unresolved behavior. Identify the revision, environment, and commands behind the evidence; label stale results. Use existing checks in a usable isolated environment, keeping generated reports temporary and leaving product/tooling files unchanged. If measurement is blocked, report it as unknown with the blocker and proposed measurement step. Do not run downloaded or repository setup scripts just to discover the project.

| Starting point | Route |
|---|---|
| Usable, owner-approved architecture exists | Reuse it. `docs/PROJECT.md` may simply link existing canonical documents. |
| Empty repository, no architecture | Interview the owner about the first useful outcome and constraints; propose a small architecture record. |
| Existing code, no usable architecture | Describe observed structure, ask what is intended, and resolve material gaps. Do not assume existing behavior is correct. |
| Architecture is unapproved, contradictory, or affected by this task | Reconcile only the affected decisions, then obtain approval. |

Architecture intake produces one short `docs/PROJECT.md`, using [the template](templates/PROJECT.md): purpose, constraints, stack, component responsibilities, public interfaces, verification commands, and operation. Prefer one deployable component. Markdown is sufficient; diagrams are optional. Link actual schemas instead of copying them. Architecture approval does not approve every future task.

Infer languages/frameworks from evidence in an existing monorepo. For an empty one, recommend a stack from the owner's requirements and obtain approval. Do not prescribe a product language because the coding harness uses it.

Choose current, coherent ecosystem tooling from primary documentation: one formatter, lint/static checks, native tests and coverage, dependency locking, and suitable security checks. Preserve sound existing choices. For example, a Python project might use Black plus Ruff linting and fast pre-commit hooks; this is an example, not a universal prescription. Repeat necessary checks in CI. Pin dependencies/actions appropriately; do not install competing tools.

Setup is the explicit exception that may create initial workflows and test infrastructure. Show the plan before applying it, separating delivery-tooling installation from existing-product remediation, with expected effort, measurement gaps, and required owner decisions for each. Installation approval does not approve product repairs or settle numerical behavior. In a repository without a commit, an authorized metadata-only initial commit may establish `main`; do not put an untested application there. A minimal callable scaffold may support the first tests, but label missing-feature evidence honestly.

Repeating setup inspects and completes approved gaps; it does not overwrite working conventions. Missing architecture prevents A or product implementation, but not read-only discovery or approved setup. Retain the 100% requirement. Expose substantial baseline maintenance or missing contracts before remediation, and route them to scoped approval. Installed tooling does not make a failing/under-covered product ready.

## 3. Step 0: clarify and approve each task

Use the same `intake` skill for architecture and tasks. Before task delivery, check the architecture route above. Reuse answers rather than repeating interviews.

Read the request, then ask about decisions that affect behavior, scope, interfaces, permissions, data, errors, cost, or acceptance. Usually ask up to three high-value questions per turn. Challenge vague answers using concrete alternatives. Recommend a simple default but get agreement; never infer consent from silence. Stop questioning settled or irrelevant future details.

The target is **no unresolved material ambiguity**, not a guarantee of zero ambiguity. A complete request can need only a meaningful read-back and approval. Internal implementation choices can remain delegated.

Create one task document from [the template](templates/TASK.md). Capture observable requirements with stable IDs, non-goals, public interfaces, positive/error/boundary examples, allowed scope, checks, risk, and a preparation/execution budget. Each expected result must follow from approved behavior, an applicable primary reference, or a mathematical invariant. Unspecified units, conventions, and estimator choices return to intake; coverage requirements do not resolve them. A reference or invariant must apply to the approved interface, not silently select missing behavior. For a bug, separate observed behavior, expected behavior, reproduction evidence, and root-cause hypotheses. An uncertain bug can receive an approved, bounded investigation; its output is a report, not a speculative fix.

Present the exact draft. Record the owner's explicit approval in the conversation or GitHub discussion, with a reference to the approved document/commit. No custom approval service or hash-signing protocol. Agents must not fabricate approval. A material change returns to intake and renews affected tests/reviews; approval is not a blank check.

Intake drafts examples, not the executable test suite. Pass A/B the approved behavior, public contracts, approved fixtures, and test conventions—not the interview transcript, patches, private design hypotheses, or implementation notes.

## 4. Four fresh delivery sessions

| Role | Work and completion condition | Restrictions by instruction |
|---|---|---|
| **A: test author** | Return behavior-focused tests, requirement/expectation sources, and classified baseline results, or a specific contract blocker. Prefer end-to-end tests. | Do not inspect implementation, its history, the learning log, or implementation conversations. |
| **B: test reviewer** | Accept an identified test revision or return concrete test defects/contract gaps, including oracle review. | Same blind inputs as A, plus A's tests. Return corrections to A; do not implement. |
| **C: implementer** | Return the smallest product change, candidate commit, and check results, or a classified blocker. | Do not change tests, fixtures, snapshots, coverage/discovery settings, workflows, delivery rules, or unrelated interfaces. |
| **D: final reviewer** | Accept the exact passing candidate or return evidenced findings on correctness, security, scope, and simplicity. | Do not fix the candidate and approve the same repair. Do not inherit C's conversation. |

Start new root sessions using the selected harness's supported controls. Do not simulate four roles in one chat or fork the implementer's conversation for review. Disable avoidable inherited memory where the harness permits it. A/B receive only a narrow input packet and are instructed not to browse other files. Before their public-API probes or test runs, configure warning and traceback rendering without implementation source excerpts; preserve diagnostic categories/messages, failure counts, and exit status. In Python, pytest's `--tb=no` alone is insufficient: warnings also need source-free formatting. Disclose accidental source exposure in the handoff; a later clean run does not restore that session's blindness. No custom filesystem isolation is required; disclose that independence is procedural, not guaranteed.

Give each role a narrow packet identifying the approved requirement group, permitted inputs, relevant test/candidate revision, requested output, and completion condition above. Group related approved cases into coherent checkpoints; do not launch another role cycle for each assertion or uncovered line.

A **test oracle** is the rule deciding whether a result is correct. B reviews that decision logic as well as the assertions. For custom numerical or plotting oracles, use small distinguishing examples to show that the rule accepts legitimate alternatives and rejects plausible wrong results. Check parsers/observers, units, tolerances, and allowed outcomes against the contract; do not repair malformed output in the oracle or invent a stopping priority. Use this existing B review, not another reviewer role or a mandatory mutation-testing platform.

After B accepts the tests, run them against the approved baseline. Only a valid test that reaches the intended behavior and fails for the approved reason supplies meaningful product-red evidence. A first-project scaffold establishes feature absence, not reproduction of an old bug. Behavior-preserving refactors need regression evidence, not an invented failure. Before any repair cycle, classify the failure and route it:

| Failure category | Next action and checkpoint consequence |
|---|---|
| Environment/tooling failure | Diagnose through the authorized setup/maintenance path, then resume the interrupted role and rerun affected checks. Dependency/import/fixture setup failures alone establish no product defect. Unchanged tests keep their checkpoint; fixture or test changes require fresh A/B review and a revised checkpoint. |
| Test defect | Fresh A corrects the test/oracle/fixture; fresh B reviews it. Record a revised checkpoint and establish valid baseline evidence before C resumes. C never repairs tests. |
| Product defect | With valid evidence and approved behavior, route initial implementation or an allowed repair to fresh C, then passing checks and fresh D. Unchanged tests retain their checkpoint. |
| Unresolved requirement | Return the affected decision to intake. After approval, renew affected A/B work and the checkpoint before C; coverage pressure supplies no approval. |

An unclassified failure needs bounded diagnosis before a product repair. Record mixed causes separately. A failure to follow an existing rule is an execution error, not automatically a missing rule.

Commit the reviewed tests locally and record that commit in the task. This is the **test checkpoint**, frozen by convention—not read-only storage. Then start C. Keep failing checkpoints local. If tests are wrong or incomplete, follow the routing above. Label tests written after implementation as supplemental, never as original test-first evidence.

Run cheap checks before expensive tests. **Run the canonical full verification command against the exact candidate before pushing. Any known failure, including incomplete coverage, blocks submission. CI repeats that verification on its configured platforms.** Record the candidate commit, command, environment, and result; confirm the tested tree matches that commit and remains unchanged through verification and push. Reporting a failure does not permit pushing it. A changed candidate needs a new full run. Do not spend D's review on a known failing candidate. D checks the tested commit, meaningful assertions, the real application result, and that C did not alter restricted files. Report an encountered violation and restore the intended baseline before continuing.

One bounded implementation repair after the initial C attempt may use fresh C and D; section 9 defines its accounting. A change of requirements or tests repeats affected earlier work. Keep one concise **Current state** section in the task document or its linked PR/discussion: actual approvals, candidate commit, reviewed test checkpoint, role/review references, check results and CI link, repair/budget use, blockers, and next action. Record a final candidate's own commit identity and results outside its tracked tree, in the conversation or PR; leave a pointer in the task prepared before verification. Do not maintain competing status copies or modify a verified candidate merely to record its verification. Replace superseded status; Git preserves superseded document versions. Link retained evidence instead of repeating reports or handoffs. Only `LESSONS.md` requires append-only treatment. No separate evidence database.

After green CI and D's review, present the PR for a normal merge. A further code change needs renewed checks/review. Do not bypass failures. Merge is not deployment; production release remains an explicit owner action with the project's smoke check and recovery instructions.

## 5. End-to-end first, not end-to-end only

**Coverage is primarily earned by exercising useful behavior through the real entry point.** For a web app, run important browser journeys through the real application/backend and a disposable test database. For a CLI, run its executable and inspect outputs, exit codes, and files. For a service or library, use its public API and observable effects; do not invent a browser layer.

For each changed requirement, start with the smallest realistic success, failure, and boundary scenarios. Assert the outcome, not merely that a mock was called, a file contains a string, or the application compiles. Trace expected results to the contract/reference/invariant required by section 3, never the implementation being tested. Contract-supported invariants may test part of a behavior while an absolute interpretation remains unresolved; record that limit rather than claiming full behavioral validation. Do not duplicate every end-to-end assertion with unit tests. These choices borrow pstack's [behavior](REFERENCES.md#r2--test-behavior-not-implementation) and [real-artifact](REFERENCES.md#r3--verify-the-real-artifact) emphasis, and, for browser projects, [Playwright's user-visible testing guidance](REFERENCES.md#r8--end-to-end-testing-practice).

Use integration or unit tests for a genuine gap: difficult error injection, combinatorial logic, or a branch that would make a whole-system test slow or brittle. No unit-test quota or fixed test-type percentage. Do not forbid valid negative assertions or property tests just because a heuristic dislikes them.

Test owned components together where practical. Stub uncontrollable external services at their boundary, document what is not exercised, and use an approved provider sandbox check where integration correctness requires it. Do not mock the internal path whose behavior is the requirement. Use synthetic data, isolated state, readiness checks rather than fixed sleeps, and bounded waits. A suite that fails and then passes on retry still needs investigation; do not hide flakiness.

Keep the original **100% measured statement and branch coverage** requirement across instrumentable owned runtime code, globally and per package, from the combined suite. Instrument relevant subprocesses, servers, and browser code; a passing browser test alone does not measure backend coverage. Include never-imported files. Inspect exact metrics, not rounded display values.

Narrow documented exclusions may cover vendor, generated, or non-executable files—not inconvenient handwritten branches. Missing/unsupported coverage is a reported gap, never fabricated success. Do not reduce the threshold through `doctor`. Complete measured coverage is not proof of complete behavior. Use existing tools and the test review; no mandatory mutation-testing platform.

## 6. Ordinary GitHub CI

Use the monorepo's ordinary GitHub Actions workflow. Prefer one required job, `verify`, running the project's canonical checks in order. A genuine multi-platform need may require several jobs; keep every required result visible. No custom gate publisher or GitHub App.

Apply section 4's exact-candidate local verification condition to every push. Use fast local hooks for formatting, lint, and docstrings, and a pre-push full check. Hooks are bypassable feedback. CI repeats the shared verification command on the submitted commit and its supported platform matrix; local success does not certify other platforms. A newly discovered CI failure returns through section 4's routing before another submission.

CI should install locked dependencies, run formatting/lint/types as applicable, build, run tests/coverage, and retain useful failure reports. Fail on failed commands, empty test discovery, unexpected skipped/focused tests, and missing required reports using native tools where available. Do not use `continue-on-error` or path filters that silently remove required validation. Keep diagnostic output short with access to full logs.

Configure native protection for the approved PR target branch (normally `main`): PRs, required CI, current branches, and no ordinary force-push/deletion or bypass. [GitHub setup](GITHUB-SETUP.md) explains the owner actions and capability limits. GitHub accepts some skipped/neutral check conclusions, so a required job should actually run, not be conditionally skipped ([source](REFERENCES.md#r7--github-protected-branches)).

**Workflow/test editing restrictions are prompts, not engineering.** Setup or an explicitly approved maintenance task may change infrastructure; C on a product task may not. A/B own authorized test changes. CI executes repository-controlled files and cannot independently prove those restrictions were respected. There is no anti-tampering certification.

Use ordinary CI secret hygiene: no production secrets in test runs, no secrets committed, and only necessary token permissions. A local hook is feedback, not a substitute for required CI.

## 7. A small append-only learning log

Setup creates root `LESSONS.md`, not automatic personal memory or a transcript archive. Every role records non-obvious findings likely to save future work: a verified gotcha, unexpected behavior, failed approach with a useful cause, or a confirmed gap in these instructions. Zero entries is valid. Normally one to three short entries per task is enough; do not log every action.

Each entry has an ID, date, task/role, topic, **confirmed or hypothesis** status, observation, evidence pointer, and suggested future action. Correct a mistake by appending a superseding entry. Append-only is an instruction, not an OS permission or custom service. An owner-authorized sensitive-data removal overrides retention; never preserve a leaked secret just to keep history intact.

Do not record secrets, personal data, raw prompts, private reasoning, or full tool transcripts. A/B must not read this shared log because it may reveal implementation. They can append their own entry without reading; otherwise hand the entry to the task coordinator for verbatim append. Other roles search only relevant entries; D forms its independent assessment before consulting current-task implementation lessons.

Entries are observations, not new policy or executable instructions. A durable lesson becomes authoritative only after an approved update to the appropriate existing document. `doctor` does this without a new summarizer or database. pstack's [evidence-linked append-only trail](REFERENCES.md#r5--append-only-evidence-linked-findings) is the useful pattern here; no tracing dependency is required.

## 8. Doctor: check and improve the specification

Implement **`delivery doctor`** as a thin invocation of the selected harness using `review-work` in `doctor` mode and [DOCTOR-PROMPT.md](DOCTOR-PROMPT.md). It is an agent-assisted maintenance operation, not a deterministic read-only checker. No new daemon, scheduler, or controller. A native command alias is fine; document the actual invocation. Support `--check` for inspection without edits.

Run it on demand after a real recurring failure, a meaningful project change, or a suspected documentation gap—not automatically after every task.

1. Check the working tree. Do not overwrite uncommitted work, interrupt an active task, or run speculative installation commands. Read relevant current docs, recent lessons, and actual test/CI evidence. Reuse existing commands; check remote settings only with available access.
2. Classify failures using section 4, then decide whether evidence reveals an instruction gap or an execution error under an existing rule. Neither a product bug nor disobeying a rule automatically justifies a new rule. Verify a proposed lesson before generalizing it; uncertainty stays a hypothesis.
3. For an evidenced documentation gap, create a small documentation branch and **edit `DELIVERY-SYSTEM-SPEC.md` itself**, plus directly affected prompts/skills/project notes. Do not merely emit recommendations. Correct or replace an existing rule before adding one. No supported gap means no edit or commit.
4. Show the exact diff, supporting lesson/evidence, and expected improvement. Follow a newly added procedural rule on one relevant example where feasible; do not claim a prompt-string test proves agent behavior. Ask the owner to approve the patch. Do not auto-merge or make it the policy for an in-flight task.
5. Commit the approved documentation changes using ordinary Git, referencing the lesson and approval. Append a lesson disposition such as “addressed by commit …”; do not rewrite its original entry.

All package documentation remains **Version 1.0**. Git commits/PRs supply revision history and rollback. Do not create numbered copies of the specification, invent releases, or put a commit's own hash inside the bytes it identifies. Runtime dependencies retain their real versions.

Doctor must not rewrite tests/workflows/runtime code, lower acceptance thresholds, remove an inconvenient requirement, or bless architecture drift merely to make the current result pass. Those findings become scoped, owner-approved tasks. Material product/architecture questions return to intake. Diagnostic docs edits are the explicit exception to ordinary workers' “do not edit the spec” rule; the limits are still instructions, not enforced permissions.

## 9. Cost and completion

One active task, one intake conversation, four delivery sessions on the normal path, and at most one corrective implementation cycle after the initial C attempt by default. Record that allowance against the approved task and identified checkpoint before execution. Replacement checkpoints, regrouped cases, renamed tasks, and fresh sessions inherit the used allowance for the same work; they do not reset it. Further repairs need an explicit owner-approved extension. Environment/test corrections follow section 4 rather than being counted as product repairs, but all preparation, diagnosis, corrections, and failed attempts count toward the task budget. Stop and report at its limit. Use native runtime/time/spend limits where available, not a custom billing layer; disclose missing metering.

Load the role packet, relevant interfaces, and the selected skill—not this whole package, the source archive, or the learning log on every turn. Use deterministic commands for routine checks. Rerun checks when candidate/test/environment changes, a required platform check, or unresolved evidence justifies them; reuse applicable evidence otherwise. This does not waive the exact-candidate full check before push. Combine related approved cases and already-known local actions when useful, but never merge independent roles. Do not add log-summarizer models or automatic context optimizers. Keep shared operational instructions in native `AGENTS.md`, with its generation template synchronized; skills carry only role-specific differences.

Before calling setup usable, demonstrate: architecture routing on an empty/existing project; baseline inventory and separately scoped remediation; a clarifying interview and approval; fresh A/B then valid baseline evidence/test checkpoint, C, full local success and D; native required-check configuration where available; one normal passing PR; a lesson appended and a doctor-generated spec diff. Use existing failed-run/protection evidence when available; do not push a known failure solely to demonstrate blocking. Configuration inspection alone does not demonstrate an observed blocked merge; report that limit. Record actual results and gaps. No custom acceptance-test matrix or platform certification is required. Any executable helper added to the repo needs ordinary independent tests, preferably through its real entry point.

The handoff is not a running system until those demonstrations and the GitHub configuration actually exist. Unsupported protection or missing evidence must be reported plainly. Never claim that prompt-only safeguards were mechanically verified.

## 10. Package and sources

[README.md](README.md) is the human entry point. [SETUP-PROMPT.md](SETUP-PROMPT.md) starts installation; [INTAKE-PROMPT.md](INTAKE-PROMPT.md) starts clarification. Four short skills support the work. [WORKFLOW-EXAMPLES.md](WORKFLOW-EXAMPLES.md) shows daily use. Keep existing product source/test directories.

[REFERENCES.md](REFERENCES.md) records selected sources, adopted ideas, and reading limitations. During setup, preserve the original requested articles and pertinent adopted sources in `docs/references/` where permitted, with URLs, retrieval dates, source versions, and hashes. Verify hashes against both working files and Git-stored bytes (the index before commit, the committed blobs afterward); a working-file match alone does not establish archive fidelity. Preserve retrieval bytes through Git attributes, or label retrieval and normalized-content hashes separately; do not describe normalized content as unchanged upstream bytes. Read selected sources fully, follow links material to the adopted mechanism, and record inaccessible material. Do this once per selected version, not per task. Sources are background, not additional requirements or install instructions.
