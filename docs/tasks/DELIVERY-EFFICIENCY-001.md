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

- Owner authorized all four phases, delivery instructions, tests, tooling, CI,
  codex-main protection, commit/push/PR, and merge after independent review and
  complete platform CI. No release is authorized. This is routine maintenance;
  numerical behavior, public APIs, locked dependencies, and release controls stay
  unchanged. Specialist independence remains required for scientific/safety risk.
- Fresh worktree `/workspace/dphtools-delivery-implementation`, branch
  `codex/delivery-efficiency-implementation`, preserves planning commit
  `5b82cddc24bed7aa4aa7c853c333d0fcd369ce3e`. Fetched codex-main remains
  `4e00fe4e466d58beabf97e4d360947437e9fd895`; integration was already current.
  Existing worktrees and their changes are untouched.
- Host: Linux, isolated copied Python 3.10.21 at `.python/verification310`,
  unchanged hashed development lock, retained offline wheels, writable report
  caches. Real installation/subprocess checks passed after repairing the private
  interpreter layout; neither dependency pins nor the system interpreter changed.
- All four phases are integrated: proportional delivery, real cheap preflight,
  dependency-aware immediate receipts/live logs, immutable release artifacts,
  two isolated CI workers per platform, complete assignment/coverage aggregation,
  conservative explicit docstring reuse, and deterministic metrics.
- Independent review found unbound startup/import/configuration inputs that could
  cause stale cache passes. Runtime identities now bind known inputs and decline
  unknown providers, custom coverage plugins, forced configuration selection,
  and coverage's implicit `.coveragerc` selector. Ordinary file/inline subprocess
  coverage remains active. Retained regression probes and lessons record the
  findings. Earlier full/CI successes are superseded trials; final source needs
  fresh independent review, a full reference, and complete platform CI.
- Candidate hash, review, current receipts, and measured CI results belong in
  [PR 17](https://github.com/david-hoffman/dphtools/pull/17) and the conversation,
  outside this tracked candidate. This section is the sole task pointer.
- Merge blocker: actual codex-main readback reports unprotected with no required
  checks or rulesets. The GitHub connection lacks administration access (403).
  Administration-capable configuration was requested; merge requires verified
  actual strict required `ci-required` protection, never workflow text alone.
- Metrics: 3 implementation authors, 2 read-only support agents, 1 fresh reviewer;
  no specialist rounds. A real 97-case release pilot used one master build across
  four modules (75% fewer). Historical CI used 2,680 seconds wall/5,067 runner
  seconds; retain all new trial costs and label uncontrolled comparisons. Token
  billing and ten comparable completed maintenance tasks are unavailable, so the
  ten-task pilot remains incomplete. No invented gains or passing partial proof.
- Next: close focused regression/review, freeze and push the candidate, run the
  full reference and all platform gates, report complete measurements, and merge
  only after protection is configured and verified. No publication.

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
