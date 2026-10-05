# Delivery efficiency implementation plan

**Goal:** Reduce routine delivery latency and model context cost while retaining
independent review, real installation tests, and complete platform verification.

**Execution:** Implement in a fresh Codex task on `codex/delivery-efficiency`.
Use one governing repository workflow. The owner has already instructed us to
make the plan and implement the preceding six recommendations; another generic
planning approval round is unnecessary. This is authorized delivery-maintenance
work, including its tests, workflows, instructions, and project configuration.

**Architecture:** Extend the existing verifier and GitHub Actions workflow.
Use explicit dependencies, ordinary JSON receipts, immutable fixture artifacts,
and isolated test workers. Do not add an orchestration service or a database.

**Stack:** Existing Python tools, pytest, coverage.py, Git, and GitHub Actions.
Keep Python >=3.8 package metadata, Python 3.10 CI, the pinned dependency lock,
and Linux/macOS/Windows verification.

## Current state

- Owner authorized all four implementation phases, delivery instructions, tests,
  tooling, CI, codex-main protection, commit/push/PR, and merge after fresh
  independent review and complete platform CI. No release is authorized.
- Fresh worktree `/workspace/dphtools-delivery-implementation`, branch
  `codex/delivery-efficiency-implementation`, preserves planning commit
  `5b82cddc24bed7aa4aa7c853c333d0fcd369ce3e`. Fetched codex-main remains
  `4e00fe4e466d58beabf97e4d360947437e9fd895`; integration was already current.
  Existing worktrees and their changes are untouched.
- Host is Linux with an isolated copied Python 3.10.21 distribution at
  `.python/verification310`, installed from unchanged hashed requirements-dev.lock.
  The first standalone-interpreter venv failed a real copied-venv installation
  observer. The isolated full prefix repairs that environment defect without
  changing tests, dependencies, or the system interpreter. Locked offline wheels
  are retained outside Git; pip check and the real observer passed.
- All four implementation phases are integrated: proportional instructions,
  preflight/dependency receipts, immutable artifacts/two isolated CI shards per
  platform, and narrow explicit evidence reuse plus deterministic metrics.
  Focused verifier/shard/reuse/metrics checks precede the final full reference.
  Final candidate, independent review, full/CI receipts, and PR belong in the
  conversation/PR outside the tracked candidate. This record is the task pointer.
- Release pilot: 97 real cases passed in 651.34 seconds with one immutable
  wheel/sdist build shared across four modules. The prior 850.77-second run had
  an environment failure and different provisioning, so its elapsed difference
  is not a controlled speedup. Every installation and injected failure still runs.
- Genuine merge blocker: codex-main branch readback reports protected=false and
  no required check; repository rulesets returned none. The GitHub integration
  cannot read or change administration settings (403); host API access is
  forbidden. Administration-capable access/configuration was requested. Never
  merge or infer protection from workflow text while this remains unresolved.
- Metrics tooling records latency, repeated inputs/checks, author/reviewer
  launches, runner seconds, available model tokens, and defects. Historical full
  platform CI took 2,680 seconds wall and 5,067 seconds summed runner time;
  compare revised CI only after it completes, labeling uncontrolled differences.
  Three implementation authors, two read-only support agents, and one fresh
  independent reviewer launched.
  The reviewer reproduced false cache hits through external .pth providers,
  startup packages, and native import providers; repaired runtime identities
  reject these unbound inputs. Independent source review accepted the repairs.
  PR [17](https://github.com/david-hoffman/dphtools/pull/17) retains candidate
  hashes and current verification evidence. CI exposed a stale hook expectation
  and inherited pytest roots; focused regressions now cover both repairs.
  Reuse fixtures also keep cold bytecode caches stable and identify each new
  receipt separately from the original receipt linked by provenance. CI retains
  required proof files without uploading disposable installations and temp trees.
  The macOS framework's stock Headers directory alias exposed an overly strict
  eligibility rule. Runtime identities now bind internal directory aliases and
  their fully hashed canonical targets, while rejecting external targets. Real
  venv and alias-change regressions cover this repair; final review and all gates
  still apply to the final candidate linked from the PR.
  Independent review also reproduced stale reuse after supported root and
  inherited pydocstyle configuration changed. Bind the complete configuration
  chain and reject unknown import providers with case-insensitive suffix checks.
  The earlier full reference passed at exact 100% owned coverage, but approval
  was withdrawn for these observed defects; repaired source requires new gates.
  Windows CI then declined valid reuse because the pinned coverage wheel's stock
  startup hook has CRLF bytes rather than the audited Linux LF bytes. The wheel
  SHA is in the unchanged lock. Accept only both audited exact hook hashes;
  keep runtime byte identities and unknown-hook rejection. Private-environment
  regression proof and complete Windows CI remain required for this correction.
  Re-review and a fresh full reference on the final repaired candidate remain
  required. An initial full run stopped at a read-only audit cache; a
  writable XDG_CACHE_HOME passed the real audit. The next run was interrupted for
  the evidenced source repair and remains incomplete, never passing evidence. Token/billing data is unavailable. Ten comparable completed tasks do
  not yet exist; the pilot remains incomplete rather than inventing observations.
- Next: finish focused integration checks, full reference gate, fresh independent
  final review, update the PR, complete Linux/macOS/Windows aggregation, and merge
  only after actual ci-required protection is verified. No publication.

## Requirements and review focus

1. A routine task uses an author and one fresh independent reviewer. New
   scientific contracts/custom oracles, release safety, or other material risk
   retain separate test-author/test-reviewer/implementer/final-reviewer work.
   Classify risk by changed behavior and lost proof, not patch size.
2. Routine PRs may open after meaningful focused checks and cheap verification.
   Merging requires the full platform matrix, exact complete owned statement and
   branch coverage, independent review, and an unchanged reviewed candidate.
3. Configure `ci-required` as an actual required check on `codex-main` before
   relying on CI as the authoritative merge gate. Preserve production `main`
   release controls. Missing administrative access is a real blocker, not a
   reason to claim protection exists or bypass it.
4. Failures, blocked dependencies, missing reports, omitted sources, skipped
   tests, and unsupported measurement remain visible and cannot produce PASS.
5. Reuse immutable bytes and valid evidence. Never share mutable installation
   environments, silently pool operating systems, or relabel old commands as
   having run against a new revision.

The main risks to review are false cache hits, partial/foreign shard receipts,
missed subprocess measurement, shared mutable fixtures, and inconsistent policy
copies. Preserve numerical behavior, public APIs, dependencies, and publication
approval requirements throughout.

## Phase 1: Proportional workflow and enforceable CI

**Files:** `AGENTS.md`, `.agents/skills/{intake,design-tests,implement-task,review-work}/SKILL.md`,
`docs/PROJECT.md`, `docs/tasks/LOCAL-VERIFICATION-CONTRACT.md`, and the relevant
specification, prompts, examples, `GITHUB-SETUP.md`, and `templates/AGENTS.md` under
`docs/agentic-software-delivery-v1.0/`.

- [x] Introduce explicit routine/high-risk routing. Reuse established contracts
  for routine maintenance; retain independent numerical oracle work when needed.
  Do not require a fresh specialist or a new document for every assertion.
- [x] Align every live operational copy with the revised routing and submission
  gate. Keep fast Git hooks. Document when a full local run is useful or required
  for diagnosis/high-risk work, without requiring it before every routine PR.
- [ ] Require the stable `ci-required` context on `codex-main` using supported
  GitHub settings, preserving unrelated branch/release settings. Verify the
  returned configuration; report an observed enforcement demonstration separately.
- [x] Use narrow packets and one Current state record. Reuse relevant context;
  store full logs outside Git and emit concise structured summaries. Use native
  CI completion/auto-merge facilities only after protection is established.
- [x] Document host limits honestly: repository instructions cannot override
  mandatory host progress updates or create unavailable event-wait tools.

**Acceptance:** Trace one routine path and one high-risk path; verify no
conflicting live instructions remain. Show required-check configuration and one
ordinary green PR. No workflow claim substitutes for unavailable enforcement.

## Phase 2: Cheap preflight, dependency-aware checks, and receipts

**Files:** `tools/verification.py`, `tests/test_verification.py`, the local
verification contract, and project commands. Add a focused helper under `tools/`
only if it has a distinct responsibility; owned-source discovery must include it.

- [x] Add `python tools/verification.py preflight`. Check the selected interpreter,
  locked dependencies, imports, and actual nested-venv/pip operation in disposable
  paths. Detect installation blockers without modifying the system interpreter.
  Reviewer connectivity is checked only when a task uses that external reviewer;
  a definitive connection rejection ends that attempt promptly.
- [x] Add an explicit step dependency table and a small `run_step` helper.
  Independent cheap checks may still run after another cheap check fails.
  Expensive build/install/tests wait for successful prerequisites. Installation
  requires a successful build and exactly one expected wheel.
- [x] Keep ordinary `fast` and unsharded `full` available. Preserve coverage
  processing after failed tests when fresh data exists, for diagnostics; the
  failure remains a failure. Never use old reports to fill an interrupted run.
- [x] Write each step receipt immediately with command, state
  (`passed`, `failed`, `blocked`, later `reused`), return code where applicable,
  monotonic duration, dependencies/blocking reasons, and relevant input identity.
  Stream subprocess output to its log so failures are observable before the end.
- [x] Bind source/configuration/lock inputs, interpreter/dependency identity,
  platform, coverage settings, check version, and artifact hashes. Build/install
  identities additionally include Versioneer's actual Git inputs: revision,
  applicable tags/history/shallow state, and dirty state/version result.

**Meaningful tests:** Exercise the real verifier command for healthy preflight,
broken nested environments, prerequisite/build failures, absent/extra wheel
artifacts, process launch failure, available diagnostics after test failure,
incremental receipts, and stale report rejection. Revise the existing
`test_full_failure_stays_failed_and_attempts_all_later_steps` contract explicitly.
Retain the real tests for never-imported owned files and executed subprocess code.

**Acceptance:** Focused verifier tests and `fast` pass; a failing prerequisite
avoids dependent expensive commands and still returns failure with useful logs.

## Phase 3: Immutable test inputs and isolated CI shards

**Files:** `tests/test_release_smoke.py`, `tests/test_release_workflow.py`, relevant
release test support, `tools/verification.py`, a small `tools/verification_shards.py`
if warranted, `tests/test_verification.py`, and `.github/workflows/ci.yml`.

- [x] Make reusable wheel/sdist artifacts session scoped and immutable, with
  verified hashes. Continue copying them into each test's private bundle.
  Keep real fresh installers, child-process observers, and measurement hooks.
- [x] Consolidate repeated clean controls only where the exact artifact, driver,
  interpreter/platform/dependencies/environment, and measurement inputs match.
  Continue executing every injected negative case; preserve recovery ordering.
- [x] Add narrow `collect`, `shard`, and `aggregate` verifier modes. Collection
  freezes real pytest/doctest node IDs and a manifest digest. A shard receives
  that manifest plus a zero-based index/count and a private report directory.
  Ordinary `full` remains the local/reference path.
- [x] Use a separate checkout, interpreter/environment, install destination,
  temporary directory, and report/coverage directory for each CI worker.
  Assign every node exactly once. Start with a measured small shard count and
  balance using recorded durations; do not add unrestricted worker fan-out.
- [x] Aggregate matching successful shards for each operating system separately.
  Reject absent/duplicate shards or nodes, mismatched inputs, foreign platforms,
  stale/corrupt artifacts, and incomplete executions. Combine actual parent and
  child coverage data, then apply the exact existing owned-source/report checks.
  `ci-required` requires successful complete aggregation on all three platforms.

**Meaningful tests:** Compare unsharded and partitioned real test collection;
prove completeness/disjointness, failure propagation, receipt mismatch rejection,
and actual child coverage aggregation. Preserve mutation/isolation tests for
private bundles and real installation/transport controls.

**Acceptance:** Every original case runs once per required platform; 100% owned
statement and branch coverage remains exact. Report wall time and runner usage
against the unsharded reference, not only test-count reductions.

## Phase 4: Conservative evidence reuse and a measurable pilot

**Files:** existing verification helpers/tests and operational documentation.

- [x] Add opt-in reuse from an explicit receipt. Begin with an audited whitelist
  of deterministic checks whose complete relevant inputs can be identified.
  Unknown inputs, failed/blocked/incomplete receipts, tampering, and changed
  commands invalidate reuse. Log the original command and receipt as provenance.
- [x] Revalidate artifact/report bytes and every declared input before reuse.
  Do not cache advisory auditing without a freshness policy. Do not reuse a whole
  build/install/full-test proof across Git-derived version changes by tree hash.
- [ ] Use duration and receipt data to compare ten comparable maintenance tasks:
  time to PR/merge, available model tokens, agent launches, repeated checks,
  runner usage, and defects. Label missing billing data and estimated savings.
  Summarize with a deterministic command or existing report tooling; no dashboard
  or summarizer model is required.
- [x] Consider deduplicating PR/post-merge checks only after trustworthy proof of
  the actual merged tree and version inputs exists. Until then preserve them.

**Acceptance:** Unchanged eligible checks reuse evidence; changed or unverifiable
inputs rerun. Required full CI and release preparation cannot be satisfied by a
partial cache receipt. Record pilot results and remaining host limitations.

## Execution and final verification

- Work in the fresh task; implement phases in dependency order. This plan is
  maintenance authority, not permission to weaken test or report expectations.
- Use meaningful public-command tests before behavior changes. One independent
  reviewer checks the complete candidate; add specialists only for unresolved
  scientific/security questions. Preserve reviewer independence.
- Run relevant focused checks after each phase. Run one full reference check once
  integration is stable, and the complete revised platform gate before merging.
  Extra full runs need changed inputs, a failure, or another concrete reason.
- Keep the candidate unchanged during final validation and review. Put its hash,
  results, review, CI, and timing receipts in the conversation/PR, outside the
  tracked candidate. Do not infer publication permission from task delivery.
- Update this Current state section instead of creating competing status files.
