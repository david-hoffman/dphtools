# Agentic Harness Implementation Specification

This document is an implementation brief for a coding agent working inside one repository.

Implement the harness for the **current repository only**. Do not assume the repository name, default branch, package name, test runner, supported Python versions, release process, or downstream dependency graph. Discover those facts from the repository and record them before changing files.

The harness must make agent-produced changes hard to merge unless they have clear issue scope, clean-context tests, independent implementation, review evidence, deterministic local checks, and required GitHub CI gates. It cannot prove code is bug-free. Treat it as defense in depth.

## 0. Non-negotiable operating rules

1. Work from a branch. Never push directly to the default branch.
2. Do not implement all phases in one pull request. Phase 0 is the default task unless the issue explicitly requests another phase.
3. Do not change product behavior in Phase 0. Harness, CI, docs, templates, and validation scripts are allowed. Package source changes are not.
4. Do not assume the default branch is `main`. Discover it.
5. Do not assume the project uses `src/`, `tests/`, `pyproject.toml`, `setup.py`, `pytest`, or any specific package manager. Discover it.
6. Do not assume every workflow can be made strict immediately. Add baselines and ratchets where legacy gaps exist.
7. Do not weaken, delete, skip, or loosen tests to make implementation pass.
8. Do not change release workflows, secrets, protected files, or security settings without a high-risk issue and human/admin approval.
9. Do not infer the contents of web references. Read each required reference and each pertinent sublink before summarizing or using it.
10. If a required reference or tool is unavailable, record the blocker and continue only with work that does not depend on the missing fact.

## 1. Phase model

Each phase must be a separate issue and pull request unless the repository owner explicitly combines phases. Earlier phases must be green before later phases become blocking.

### Phase 0: Bootstrap the harness

Goal: add the repo-local harness and CI structure without changing product behavior.

Allowed changes:

- Repository intake documentation.
- Reference manifest and notes.
- `AGENTS.md`, `CLAUDE.md`, harness docs, testing docs, and quality score.
- Issue and pull request templates.
- CI workflow with an aggregate `ci-required` job.
- Validation scripts for harness files and references.
- Local command wrappers such as `Makefile` and `noxfile.py`.
- Tool configuration for existing tools or new development-only tools.
- Claude Code project skills, subagents, hooks, and workflows if syntax can be validated against the installed Claude Code version.
- Placeholder/stub docs where repository-specific facts still need owner confirmation.

Disallowed changes:

- Product source behavior changes.
- Public API changes.
- Default branch rename.
- Release publishing changes that could publish artifacts.
- Coverage thresholds that immediately fail on known legacy gaps unless the repo already satisfies them.
- Required security jobs that are known to fail on existing code without a staged remediation plan.

Phase 0 acceptance criteria:

- The current test suite still passes, or existing failures are documented as pre-existing with exact reproduction commands.
- `make harness-check` passes.
- The CI workflow has no required path filters and exposes an aggregate `ci-required` job.
- A manual GitHub configuration checklist exists.
- `QUALITY_SCORE.md` is initialized from discovered facts.
- The pull request clearly states that no product behavior changed.

### Phase 1: Configure GitHub protections

Goal: make GitHub reject unsafe merges to the discovered default branch.

This phase usually needs a human/admin because repository settings, branch protection, rulesets, environments, secrets, and security features may not be writable by the coding agent.

Acceptance criteria:

- Direct pushes to the default branch are blocked.
- Pull requests are required before merge.
- `ci-required` is a required status check.
- Required conversations must be resolved.
- CODEOWNER review is required for harness, CI, release, security, and packaging files.
- Force pushes and branch deletion are blocked.
- Admin bypass is disabled after an emergency access path is confirmed.

### Phase 2: Enforce clean-context agent workflow

Goal: require evidence that code changes were developed through independent test-author, implementer, and reviewer roles.

Acceptance criteria:

- PRs with source changes fail validation when linked issue, red-test proof, implementation metadata, or review metadata is missing.
- Red tests must be shown to fail on the base commit for the intended reason.
- Implementation must not modify red tests except through the documented test-amendment protocol.
- Numerical/scientific changes require numerics review metadata.

### Phase 3: Ratchet test coverage and test quality

Goal: move from baseline coverage to practical complete coverage.

Acceptance criteria:

- Coverage baseline is recorded.
- Total coverage cannot decrease without a linked waiver issue.
- Diff coverage is required for changed Python code after the repo can support it.
- Public APIs have behavior assertions, not only import or smoke tests.
- Numerical tolerances and fixtures follow repo policy.

### Phase 4: Harden downstream and release flows

Goal: protect users, dependent repositories, and package publishing.

Acceptance criteria:

- If the repo has downstream consumers, it can run downstream smoke tests against a local build.
- Package build and installed-artifact smoke tests are required.
- Release workflows require protected environments and human approval.
- Release tokens are not exposed to ordinary pull request workflows.

## 2. Repository intake

Run repository intake before making Phase 0 changes. Write the result to:

```text
docs/generated/repo-intake.md
```

The intake must identify facts, assumptions, and open questions separately.

### 2.1 Required facts to discover

Record:

- Repository name and remote URL.
- Default branch.
- Current branch and base commit SHA.
- Primary language and secondary languages.
- Package/import name or names.
- Public package/module layout.
- Existing test directories and test runner.
- Existing CI workflows and triggers.
- Existing release workflows and publishing targets.
- Existing package metadata and supported runtime versions.
- Existing dependency files.
- Development dependency mechanism, if any.
- Existing linters, formatters, type checkers, and docs tools.
- Current coverage tooling and coverage baseline, if available.
- Existing fixtures, generated files, and binary data.
- Hardware-facing, scientific, numerical, image-processing, or device-control areas.
- Downstream projects or consumers, if documented in the repo.
- Files that should require human/admin review.

### 2.2 Required intake commands

Use commands appropriate to the repo. The following are examples, not assumptions:

```bash
git remote -v
git branch --show-current
git rev-parse HEAD
git symbolic-ref refs/remotes/origin/HEAD || true
find . -maxdepth 3 -type f | sort | sed 's#^./##' | head -300
find .github -maxdepth 3 -type f -print 2>/dev/null | sort
find . -maxdepth 3 \( -name 'pyproject.toml' -o -name 'setup.py' -o -name 'setup.cfg' -o -name 'requirements*.txt' -o -name 'tox.ini' -o -name 'noxfile.py' \) -print
find . -maxdepth 3 \( -name 'test*.py' -o -name '*_test.py' \) -print | sort
```

If Python packaging exists, inspect metadata using the safest available route:

```bash
python - <<'PY'
from pathlib import Path
for name in ['pyproject.toml', 'setup.py', 'setup.cfg']:
    path = Path(name)
    if path.exists():
        print(f'--- {name} ---')
        print(path.read_text(errors='replace')[:6000])
PY
```

Do not run release, publish, deployment, hardware-control, or destructive commands during intake.

### 2.3 Intake output template

```markdown
# Repository intake

Date: <YYYY-MM-DD>
Base commit: <sha>
Default branch: <branch>
Current branch: <branch>

## Facts

## Assumptions

## Open questions

## Package and public API map

## Existing tests

## Existing CI and release workflows

## Dependency and runtime support

## Hardware/scientific/numerical areas

## Fixtures and generated data

## Downstream or integration surface

## Files requiring human/admin review

## Phase 0 implementation notes
```

## 3. Reference preservation

The repository must preserve the source basis for the harness. Create:

```text
docs/references/agentic-harness/
  README.md
  sources.yml
  notes/
    0001-claude-code-skills.md
    0002-claude-code-dynamic-workflows.md
    0003-openai-symphony.md
    0004-openai-harness-engineering.md
    0005-github-branch-protection.md
    0006-github-status-checks.md
    0007-github-actions-python.md
    0008-github-dependabot-options.md
    0009-github-dependency-security.md
    0010-github-environments.md
  snapshots/
    .gitkeep
  checksums/
    .gitkeep
```

### 3.1 Required sources

Read these references in full before using their content:

```text
https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills
https://claude.com/blog/introducing-dynamic-workflows-in-claude-code
https://openai.com/index/open-source-codex-orchestration-symphony/
https://openai.com/index/harness-engineering/
```

Read these official GitHub references before implementing CI, branch protection instructions, Dependabot configuration, security settings, or release environments:

```text
https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks
https://docs.github.com/en/actions/tutorials/build-and-test-code/python
https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference
https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/manage-your-dependency-security
https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments
```

Also read official sublinks that are directly pertinent to files being implemented. Pertinent examples include Claude Code docs for skills, hooks, subagents, worktrees, dynamic workflows, and large codebases, plus OpenAI Symphony implementation materials.

Do not add a source merely because it looks related. Add it only when it was read and used.

### 3.2 Snapshot policy

The repository should preserve references without silently violating copyright or terms.

Rules:

- `sources.yml` must include URL, title, publisher, access date, note path, snapshot path if committed, checksum if committed, and license/terms note.
- If a source clearly permits committing a full snapshot, store it under `snapshots/` and record its SHA-256 checksum.
- If permission is unclear, do not commit the full text. Store a project-use note, URL, access date, and a short terms note.
- Notes must summarize only design-relevant points.
- Notes must not pretend to be the source of truth.
- CI must fail if required URLs are missing from `sources.yml` or committed snapshot checksums do not match.

### 3.3 `sources.yml` template

```yaml
sources:
  - id: claude-code-skills-blog
    title: "Lessons from building Claude Code: How we use skills"
    publisher: "Anthropic"
    url: "https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0001-claude-code-skills.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: claude-code-dynamic-workflows-blog
    title: "Introducing dynamic workflows in Claude Code"
    publisher: "Anthropic"
    url: "https://claude.com/blog/introducing-dynamic-workflows-in-claude-code"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0002-claude-code-dynamic-workflows.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: openai-symphony
    title: "An open-source spec for Codex orchestration: Symphony"
    publisher: "OpenAI"
    url: "https://openai.com/index/open-source-codex-orchestration-symphony/"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0003-openai-symphony.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: openai-harness-engineering
    title: "Harness engineering: leveraging Codex in an agent-first world"
    publisher: "OpenAI"
    url: "https://openai.com/index/harness-engineering/"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0004-openai-harness-engineering.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: github-branch-protection
    title: "About protected branches"
    publisher: "GitHub Docs"
    url: "https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0005-github-branch-protection.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: github-status-checks
    title: "About status checks"
    publisher: "GitHub Docs"
    url: "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0006-github-status-checks.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: github-actions-python
    title: "Building and testing Python"
    publisher: "GitHub Docs"
    url: "https://docs.github.com/en/actions/tutorials/build-and-test-code/python"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0007-github-actions-python.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: github-dependabot-options
    title: "Dependabot options reference"
    publisher: "GitHub Docs"
    url: "https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0008-github-dependabot-options.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: github-dependency-security
    title: "Managing your dependency security"
    publisher: "GitHub Docs"
    url: "https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/manage-your-dependency-security"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0009-github-dependency-security.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."

  - id: github-environments
    title: "Managing environments for deployment"
    publisher: "GitHub Docs"
    url: "https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments"
    accessed_at: "<YYYY-MM-DD>"
    note_path: "docs/references/agentic-harness/notes/0010-github-environments.md"
    snapshot_path: null
    snapshot_sha256: null
    no_snapshot_reason: "Do not commit full text unless license/terms permit it."
```

## 4. Target repository layout

Implement this layout progressively. Phase 0 creates the core harness. Later phases may add stricter enforcement and optional workflows.

```text
.
├── AGENTS.md
├── CLAUDE.md
├── ARCHITECTURE.md
├── QUALITY_SCORE.md
├── Makefile
├── noxfile.py                      # if Python/nox is appropriate
├── pyproject.toml                   # tool config; do not force backend migration in Phase 0
├── .gitignore
├── .github/
│   ├── CODEOWNERS                  # if owner mapping is known
│   ├── dependabot.yml
│   ├── pull_request_template.md
│   ├── ISSUE_TEMPLATE/
│   │   ├── agent_task.yml
│   │   ├── bug_report.yml
│   │   └── release_checklist.yml
│   └── workflows/
│       ├── ci.yml
│       ├── nightly.yml             # optional in Phase 0; non-blocking initially
│       ├── release.yml             # preserve existing release semantics unless issue says otherwise
│       └── downstream.yml          # only if downstream consumers are discovered
├── .claude/
│   ├── settings.json               # only if validated against installed Claude Code
│   ├── agents/
│   │   ├── scout.md
│   │   ├── test-author.md
│   │   ├── implementer.md
│   │   ├── adversarial-reviewer.md
│   │   ├── numerics-reviewer.md
│   │   ├── ci-triager.md
│   │   └── doc-gardener.md
│   ├── skills/
│   │   ├── repo-intake/SKILL.md
│   │   ├── reference-snapshot/SKILL.md
│   │   ├── write-red-tests/SKILL.md
│   │   ├── implement-to-tests/SKILL.md
│   │   ├── adversarial-review/SKILL.md
│   │   ├── scientific-numerics-review/SKILL.md
│   │   ├── ci-triage/SKILL.md
│   │   ├── coverage-gap-hunt/SKILL.md
│   │   ├── fixture-audit/SKILL.md
│   │   ├── doc-gardener/SKILL.md
│   │   └── release-guard/SKILL.md
│   └── workflows/
│       ├── clean-context-test-first.js
│       ├── adversarial-review.js
│       ├── coverage-gap-sweep.js
│       ├── ci-repair-loop.js
│       └── doc-garden.js
├── docs/
│   ├── agent-harness/
│   │   ├── README.md
│   │   ├── workflow.md
│   │   ├── branch-protection.md
│   │   ├── clean-context-protocol.md
│   │   ├── agent-run-schema.json
│   │   ├── coverage-policy.md
│   │   ├── test-quality-rubric.md
│   │   ├── review-rubric.md
│   │   ├── implementation-notes.md
│   │   └── runs/.gitkeep
│   ├── design-docs/
│   │   ├── active/.gitkeep
│   │   └── completed/.gitkeep
│   ├── exec-plans/
│   │   ├── active/.gitkeep
│   │   ├── completed/.gitkeep
│   │   └── tech-debt/.gitkeep
│   ├── generated/
│   │   ├── README.md
│   │   └── repo-intake.md
│   ├── product-specs/
│   │   └── README.md
│   ├── references/
│   │   └── agentic-harness/
│   └── testing/
│       ├── README.md
│       ├── fixture-policy.md
│       ├── numerical-tolerance-policy.md
│       └── oracle-policy.md
└── scripts/
    └── agent_harness/
        ├── __init__.py
        ├── validate_harness.py
        ├── validate_references.py
        ├── validate_agent_run.py
        ├── validate_pr.py
        ├── validate_write_scope.py
        ├── validate_bash_command.py
        ├── prove_red_tests.py
        ├── coverage_gate.py
        ├── diff_coverage_gate.py
        ├── fixture_audit.py
        ├── downstream_smoke.py
        ├── format_touched.py
        └── session_stop_check.py
```

If the repository is not Python, adapt command wrappers and validation scripts to the repo language while preserving the same control-plane concepts.

## 5. Documentation contract

### 5.1 `AGENTS.md`

`AGENTS.md` is the short, repo-local entry point for coding agents. It is not a giant instruction dump.

Maximum length: 250 lines.

Required content:

```markdown
# AGENTS.md

## Project snapshot
- Project purpose: <one paragraph from README/intake>
- Main package/source directories: <paths>
- Test directories: <paths>
- Default branch: <discovered branch>
- Supported runtime versions: <from package metadata or intake>

## Required first reads
1. `docs/generated/repo-intake.md`
2. `docs/agent-harness/workflow.md`
3. `docs/agent-harness/clean-context-protocol.md`
4. `docs/testing/numerical-tolerance-policy.md`, when numerical/scientific code is touched
5. `ARCHITECTURE.md`
6. Relevant nested `AGENTS.md` files, if any

## Golden rules
- Do not push to the default branch.
- Do not weaken tests to make implementation pass.
- Do not change generated files by hand.
- Do not update numerical tolerances without reviewer evidence.
- Do not change release workflows or secrets without a release issue and human/admin approval.
- Do not change public API without a design doc and changelog entry.
- Label facts, assumptions, and guesses separately.

## Local commands
- Bootstrap: `<repo-specific bootstrap command>`
- Fast tests: `<repo-specific fast test command>`
- Full check: `make check`
- Coverage: `make coverage`
- Harness validation: `make harness-check`

## Agent workflow
- For code changes, use clean-context test-first workflow.
- Test author and implementation agent must be separate sessions.
- Adversarial review must run before PR is marked ready.
- Each PR must include agent run metadata after Phase 2 is enabled.

## CI policy
- `ci-required` must pass before merge.
- Required workflows must not use path filters that can skip checks.
- Fix root causes of CI failures; do not skip tests without a linked issue.

## Hardware/scientific policy
- State units where units matter.
- Prefer analytic or synthetic oracles where possible.
- Use deterministic seeds for randomized tests.
- Record fixture provenance and checksums.
```

### 5.2 `CLAUDE.md`

`CLAUDE.md` exists only to point Claude Code at the repo source of truth. It must not duplicate `AGENTS.md`.

```markdown
# CLAUDE.md

Read `AGENTS.md` first.

Then read the docs named by `AGENTS.md` for the current task.
Do not rely on this file as the source of truth.
The source of truth is the repository documentation, tests, CI, issue acceptance criteria, and pull request evidence.
```

### 5.3 `ARCHITECTURE.md`

Create or update `ARCHITECTURE.md` with discovered facts. If the repo lacks architecture documentation, Phase 0 may create a factual stub and mark unknown areas.

Required sections:

- Package or product purpose.
- Public API map.
- Internal module map.
- Dependency direction rules.
- Data model and file format conventions.
- Units and coordinate-system conventions, if applicable.
- Numerical tolerance conventions, if applicable.
- Hardware or device boundaries, if applicable.
- Known fragile areas.
- Release compatibility policy.
- Downstream dependency notes, if applicable.

### 5.4 `QUALITY_SCORE.md`

Initialize this file in Phase 0. Use facts where available and `unknown` otherwise. Do not invent scores.

```markdown
# QUALITY_SCORE.md

Last updated: <YYYY-MM-DD>
Default branch: <branch>
Default branch commit: <sha>

## CI
- Required gate present: yes/no
- Required gate unskipped: yes/no
- Matrix OS coverage: <value/unknown/not applicable>
- Matrix runtime coverage: <value/unknown/not applicable>

## Tests
- Line coverage: <value/unknown>
- Branch coverage: <value/unknown>
- Diff coverage policy: <value/unknown/not enabled>
- Mutation/property testing status: <value/unknown/not enabled>
- Flaky tests: <value/unknown>

## Harness
- `AGENTS.md` current: yes/no
- Claude skills validated: yes/no/not installed
- Clean-context metadata enforced: yes/no/not yet
- Reference manifest valid: yes/no
- Branch protection configured: yes/no/manual/unknown

## Known risks
| Risk | Severity | Owner issue | Current mitigation |
|---|---:|---|---|
```

`validate_harness.py` must fail if required fields are missing. In Phase 0 it must not fail merely because a score is low or unknown.

## 6. Issue and pull request control plane

Use GitHub Issues and Pull Requests as the durable coordination layer. If the agent cannot update labels or issue state directly, it must add comments requesting the transition.

### 6.1 Required labels

Create or document these labels:

```text
agent-ready
agent-blocked
agent-running
needs-red-tests
red-tests-ready
implementation-ready
needs-adversarial-review
needs-numerics-review
needs-ci-triage
needs-human-decision
ready-for-ci
ready-for-human-review
done
risk:low
risk:medium
risk:high
area:ci
area:harness
area:docs
area:tests
area:packaging
area:numerics
area:api
area:release
area:downstream
area:hardware
```

### 6.2 Issue template

`.github/ISSUE_TEMPLATE/agent_task.yml` must capture:

- Problem statement.
- Desired behavior.
- Out of scope.
- Acceptance criteria.
- Affected modules.
- Public API impact.
- Hardware/scientific assumptions.
- Data or fixture requirements.
- Expected tests.
- Risk label.
- Downstream impact.
- Human approval required: yes/no.

### 6.3 Pull request template

`.github/pull_request_template.md`:

```markdown
## Linked issue
Closes #

## Change summary

## Phase
- [ ] Phase 0 harness bootstrap only
- [ ] Phase 1 GitHub protection/configuration
- [ ] Phase 2 clean-context enforcement
- [ ] Phase 3 coverage ratchet
- [ ] Phase 4 downstream/release hardening
- [ ] Other: <explain>

## Agent workflow evidence
- [ ] Scout run metadata committed or not required for this phase
- [ ] Red-test author run metadata committed or not required for this phase
- [ ] Red tests failed on base commit or not required for this phase
- [ ] Implementation run metadata committed or not required for this phase
- [ ] Adversarial review run metadata committed or not required for this phase
- [ ] Numerics review run metadata committed or not required
- [ ] CI triage run metadata committed or not required

## Tests and commands
Paste exact commands and concise results. Do not paste huge logs.

## Coverage
- Line coverage before/after:
- Branch coverage before/after:
- Diff coverage:

## Scientific, hardware, or numerical impact
State facts, assumptions, and guesses separately.

## Public API impact
- [ ] No public API change
- [ ] Public API change documented in design doc and changelog

## Downstream impact
- [ ] Not applicable
- [ ] Downstream smoke run
- [ ] Downstream impact documented

## Release impact
- [ ] No release impact
- [ ] Release checklist required

## Human/admin decisions needed
```

### 6.4 State transitions

Normal path:

```text
agent-ready
  -> needs-red-tests
  -> red-tests-ready
  -> implementation-ready
  -> needs-adversarial-review
  -> needs-numerics-review, if applicable
  -> ready-for-ci
  -> ready-for-human-review, if applicable
  -> done
```

Blocked states:

```text
agent-blocked
needs-human-decision
needs-ci-triage
```

Phase 0 harness bootstrap may skip red-test states if it changes no product behavior. The PR must state this explicitly.

## 7. Clean-context protocol

The clean-context protocol is enforced in Phase 2 and later. Phase 0 creates the docs and validation scaffolding.

### 7.1 Required roles

| Role | Context rule | Allowed output | Must not do |
|---|---|---|---|
| Scout | Fresh read-only context | Issue map, affected files, risk assessment, test strategy | Modify code |
| Test author | Fresh context from issue and architecture docs only | Tests, fixtures, test design note, red-test proof | Read implementation plan; change implementation code |
| Implementer | Fresh context from issue plus red-test patch | Implementation code, docs, migration notes | Weaken/delete tests; alter test intent |
| Adversarial reviewer | Fresh context from PR diff and issue | Review report and requested changes | Author implementation |
| Numerics reviewer | Fresh context from PR diff, tests, architecture docs | Scientific/numerical review report | Accept tolerance changes without evidence |
| CI triager | Fresh context from failing CI logs and PR diff | Minimal fix or diagnosis | Hide failures by skipping tests without approval |
| Doc gardener | Fresh context from merged code and docs | Documentation consistency updates | Change behavior |
| Release guard | Fresh context from release issue and workflows | Release readiness evidence | Publish without protected approval |

One human or account may run multiple roles, but each role must use a fresh session and fresh worktree. Metadata must show that separation.

### 7.2 Code-changing workflow

1. Create or select a GitHub Issue.
2. Scout writes `docs/exec-plans/active/<issue>-scout.md`.
3. Test author starts in a clean worktree at the base commit.
4. Test author reads only the issue, `AGENTS.md`, linked docs, public API docs, existing tests, and scout affected-area map.
5. Test author writes tests and fixtures only.
6. Test author proves tests fail on the base commit for the intended reason.
7. Test author commits to branch `agent/<issue>-red-tests`.
8. Implementer starts in a separate clean worktree at the same base commit.
9. Implementer applies only the red-test commit or patch.
10. Implementer changes product code until required tests pass.
11. Implementer may not delete, skip, xfail, loosen, or rewrite red tests.
12. If a red test is wrong, implementer stops and uses the test-amendment protocol.
13. Adversarial reviewer reviews the final PR diff from a fresh context.
14. Numerics reviewer reviews if code touches numerical algorithms, scientific assumptions, fixtures, tolerances, image processing, hardware behavior, or public API behavior.
15. PR cannot be marked ready until required metadata is present and CI passes.

### 7.3 Test-amendment protocol

If red tests are wrong:

1. Stop implementation.
2. Write `docs/agent-harness/test-amendments/<issue>.md`.
3. Explain the incorrect assertion, missing assumption, or invalid fixture.
4. Propose corrected tests.
5. Request a fresh test-review agent or human decision.
6. Resume only after approval is recorded.

## 8. Agent run metadata

Each agent run writes JSON under:

```text
docs/agent-harness/runs/<issue-number>/<timestamp>-<role>.json
```

Schema file path:

```text
docs/agent-harness/agent-run-schema.json
```

Required schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AgentRun",
  "type": "object",
  "required": [
    "issue",
    "role",
    "agent_tool",
    "session_label",
    "base_sha",
    "branch",
    "started_at",
    "ended_at",
    "allowed_paths",
    "commands_run",
    "artifacts",
    "result"
  ],
  "properties": {
    "issue": {"type": "string"},
    "role": {
      "type": "string",
      "enum": [
        "scout",
        "test-author",
        "implementer",
        "adversarial-reviewer",
        "numerics-reviewer",
        "ci-triager",
        "doc-gardener",
        "release-guard"
      ]
    },
    "agent_tool": {"type": "string"},
    "session_label": {"type": "string"},
    "base_sha": {"type": "string"},
    "branch": {"type": "string"},
    "started_at": {"type": "string", "format": "date-time"},
    "ended_at": {"type": "string", "format": "date-time"},
    "allowed_paths": {"type": "array", "items": {"type": "string"}},
    "commands_run": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["command", "exit_code"],
        "properties": {
          "command": {"type": "string"},
          "exit_code": {"type": "integer"},
          "summary": {"type": "string"}
        }
      }
    },
    "artifacts": {"type": "array", "items": {"type": "string"}},
    "result": {"type": "string", "enum": ["passed", "failed", "blocked", "needs-human"]},
    "notes": {"type": "string"}
  },
  "additionalProperties": false
}
```

`validate_agent_run.py` must validate the schema and enforce required roles based on PR labels after Phase 2 is enabled:

- `risk:low`: test author, implementer, adversarial reviewer.
- `risk:medium`: test author, implementer, adversarial reviewer, plus numerics reviewer when numerical/scientific files are touched.
- `risk:high`: all applicable roles plus a human/admin approval note.
- Hardware-impacting, release, security, and default-branch governance changes are high risk by default.

## 9. Claude Code project artifacts

These files are for Claude Code users. A Codex agent may create them as repository files, but it must not invent unsupported schema. Validate syntax against the installed Claude Code version or mark the artifact as a draft in `docs/agent-harness/implementation-notes.md`.

### 9.1 Skill rules

Each skill must:

- Live under `.claude/skills/<skill-name>/SKILL.md`.
- Use frontmatter supported by the installed Claude Code version.
- Keep `SKILL.md` concise.
- Put long references in `refs/`, scripts in `scripts/`, and examples in `examples/`.
- Include `Gotchas` and `Required output` sections.
- Prefer deterministic scripts over prose-only instructions.
- Never claim success without running the named command or stating why it could not run.

### 9.2 Required skills

| Skill | Purpose | Required output |
|---|---|---|
| `repo-intake` | Build current map of repo structure, packaging, tests, CI, public API, and risks. | `docs/generated/repo-intake.md` plus run metadata |
| `reference-snapshot` | Save reference manifest, notes, snapshots/checksums where allowed. | `docs/references/agentic-harness/sources.yml`, notes, checksums |
| `write-red-tests` | Write tests from issue acceptance criteria without implementing. | Test commit, red-test proof, test design note |
| `implement-to-tests` | Implement code to satisfy red tests without weakening tests. | Code/docs commit and command evidence |
| `adversarial-review` | Look for cheating, brittle tests, missing edge cases, CI bypasses. | Review report and blocking/non-blocking findings |
| `scientific-numerics-review` | Check numerical/scientific correctness, units, tolerances, fixtures. | Numerics review report |
| `ci-triage` | Diagnose and fix CI failures without hiding failures. | Minimal fix or diagnosis issue |
| `coverage-gap-hunt` | Find untested behavior and propose red-test issues. | New issues or test PRs |
| `fixture-audit` | Check fixtures for provenance, determinism, size, and oracle quality. | Fixture audit report |
| `doc-gardener` | Keep docs current after merges. | Docs-only PR |
| `release-guard` | Validate release readiness and packaging. | Release checklist and dry-run evidence |

### 9.3 Minimal `write-red-tests` skill

```markdown
---
name: write-red-tests
description: Write failing tests from issue acceptance criteria before implementation.
---

# Write red tests

## Goal
Create tests that encode the issue acceptance criteria and fail on the base commit for the right reason.

## Required inputs
- GitHub issue number or local issue note.
- Base commit SHA.
- Relevant public API docs and existing tests.
- `docs/agent-harness/test-quality-rubric.md`.
- `docs/testing/numerical-tolerance-policy.md` for numerical work.

## Constraints
- Do not edit implementation files.
- Do not read an implementation plan from a future implementer.
- Do not assert exact floating-point values unless there is a justified oracle.
- Use deterministic seeds for randomized tests.
- Use units in test names or comments when units matter.
- Prefer small synthetic data over large binary fixtures.

## Process
1. Restate acceptance criteria as testable claims.
2. Identify the smallest public API surface that should demonstrate each claim.
3. Add tests and fixtures only.
4. Run the focused test command and record failure.
5. Run `python scripts/agent_harness/prove_red_tests.py --base <sha> --tests <paths>`.
6. Write `docs/exec-plans/active/<issue>-red-test-design.md`.
7. Write agent run metadata.

## Required output
- Failing test files.
- Red-test proof summary.
- Test design note.
- Agent run metadata JSON.

## Gotchas
- A test that fails because of import errors, missing optional packages, or bad fixture paths is not useful.
- A test that reproduces current behavior without asserting desired behavior is not a red test.
- Broad snapshot tests are weak unless backed by semantic assertions.
```

### 9.4 Minimal `implement-to-tests` skill

```markdown
---
name: implement-to-tests
description: Implement behavior required by proven red tests without weakening tests.
---

# Implement to tests

## Goal
Make the red tests pass by fixing product code, documentation, or packaging as appropriate.

## Required inputs
- Issue acceptance criteria.
- Base commit SHA.
- Red-test commit SHA or patch.
- Red-test proof.
- Relevant architecture docs.

## Constraints
- Do not weaken, skip, xfail, delete, or loosen red tests.
- Do not update golden files unless the issue explicitly requires an oracle update and a reviewer approves.
- Do not broaden dependencies without packaging and downstream evidence.
- Keep changes minimal.

## Process
1. Run the red tests and confirm current failure.
2. Inspect implementation.
3. Implement the smallest correct fix.
4. Run focused tests.
5. Run full local gate or explain exactly why it could not run.
6. Write run metadata.

## Gotchas
- Passing a test by hard-coding fixture values is a failure.
- Lowering numerical tolerances to hide instability is a failure.
- Adding broad `try/except` blocks without preserving errors is usually a failure.
```

### 9.5 Minimal `adversarial-review` skill

```markdown
---
name: adversarial-review
description: Review an agent-produced PR for cheating, underspecified tests, CI bypasses, weak oracles, and hidden regressions.
---

# Adversarial review

## Goal
Find reasons this PR could be wrong even if CI is green.

## Required checks
- Red tests fail on base for the intended reason.
- Implementation did not weaken tests.
- New tests exercise public behavior, not implementation accidents.
- CI workflow was not bypassed or path-filtered.
- Dependency changes are justified.
- Numerical tolerances are justified.
- Fixtures are deterministic, minimal, and documented.
- Public API changes are documented.
- Downstream smoke tests ran when needed.

## Required output
Write `docs/exec-plans/active/<issue>-adversarial-review.md` with blocking findings, non-blocking findings, commands run, confidence, and what would change the conclusion.
```

### 9.6 Minimal `scientific-numerics-review` skill

```markdown
---
name: scientific-numerics-review
description: Review scientific or numerical changes for assumptions, units, array conventions, tolerances, and numerical stability.
---

# Scientific and numerical review

## Required checks
- State facts, assumptions, and guesses separately.
- Identify units and coordinate conventions.
- Check shape and axis conventions.
- Check dtype behavior and casting.
- Check random seeds and reproducibility.
- Check tolerance justification.
- Compare against analytic, synthetic, or independent oracle where possible.
- Check edge cases: empty inputs, singleton dimensions, NaN/Inf, negative values, saturation, and border effects.

## Required output
Write `docs/exec-plans/active/<issue>-numerics-review.md` with accepted invariants, rejected or uncertain claims, required follow-up tests, confidence, and what would change the conclusion.
```

## 10. Claude subagents and hooks

Subagents and hooks are local reliability aids. CI and branch protection remain authoritative.

### 10.1 Subagents

Create subagent definitions only if their syntax can be validated. Required intents:

- `scout`: read-only repository mapping.
- `test-author`: tests and fixtures only.
- `implementer`: product changes to satisfy proven red tests.
- `adversarial-reviewer`: read-only skeptical review.
- `numerics-reviewer`: read-only scientific/numerical review.
- `ci-triager`: minimal CI failure fixes.
- `doc-gardener`: docs-only synchronization.

Write-scope policy:

- Test author may write tests, fixtures, test docs, and run metadata only.
- Reviewers are read-only.
- Implementer may not edit tests unless the test-amendment protocol is approved.
- Release guard may not change product source.
- No agent may edit release workflows, `CODEOWNERS`, `.claude/settings.json`, security config, or package metadata without high-risk issue approval.

### 10.2 Hooks

If Claude hooks are configured, they should call deterministic scripts:

- `validate_bash_command.py` before shell commands.
- `validate_write_scope.py` before writes.
- `format_touched.py` after writes.
- `session_stop_check.py` at session end.

Hooks must block or warn on:

- Direct default-branch pushes.
- Force pushes unless explicitly approved.
- Destructive file removal outside safe temp directories.
- Final evidence commands that hide failures using skip/ignore patterns.
- Commands that expose secrets.
- Release publication commands outside approved release workflow.

Hooks are bypassable. Re-enforce important checks in CI.

## 11. Dynamic workflows

Dynamic workflows orchestrate repeated multi-agent tasks. They must not replace CI or branch protection.

### 11.1 `clean-context-test-first.js`

Inputs:

```text
issue_number
base_ref
risk_level
```

Steps:

1. Spawn `scout` in a read-only worktree.
2. Spawn `test-author` in a fresh worktree from `base_ref`.
3. Require red-test proof.
4. Spawn `implementer` in a fresh worktree from `base_ref`, applying only the red-test patch.
5. Run focused tests.
6. Spawn `adversarial-reviewer` in a fresh worktree.
7. Spawn `numerics-reviewer` when labels or changed files require it.
8. Run the full local check.
9. Produce a PR-ready checklist.

Stop and report `needs-human-decision` when:

- Tests cannot be written from acceptance criteria.
- A required dependency cannot install.
- Red tests fail for environmental reasons.
- Implementation requires a breaking API change not in the issue.
- Numerics reviewer rejects the oracle or tolerance.

### 11.2 `adversarial-review.js`

Run independent reviews for:

- Test cheating.
- CI bypass.
- Numerical/scientific correctness.
- Packaging and API impact.
- Fixture provenance.

Output a consolidated review note with blocking findings and suggested follow-up issues.

### 11.3 `coverage-gap-sweep.js`

1. Read coverage XML/JSON.
2. Identify uncovered public API and high-risk branches.
3. Spawn test-author agents for independent areas.
4. Create issues for ambiguous behavior.
5. Create small red-test PRs for obvious missing tests.

### 11.4 `ci-repair-loop.js`

Rules:

- Do not skip tests by default.
- Do not remove platforms or runtime versions from the matrix without issue approval.
- Distinguish dependency-resolution failure from product failure.
- Preserve failure summaries and links.

### 11.5 `doc-garden.js`

Rules:

- Docs-only by default.
- If code appears wrong relative to docs, create an issue instead of changing behavior.
- Update `QUALITY_SCORE.md` after CI or coverage changes.

## 12. Local command contract

Every repository should expose a small set of stable commands. Adapt implementations to the repo’s actual stack.

Required commands:

```bash
make bootstrap
make test-fast
make coverage
make harness-check
make package
make check
```

Phase 0 rule: if the repo cannot support a command yet, implement the command with a clear failure message and open a follow-up issue. Do not fake success.

Example Python `Makefile`:

```makefile
.PHONY: bootstrap check test-fast coverage harness-check package lint format

bootstrap:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]" || python -m pip install -r requirements-dev.txt

lint:
	python -m ruff check .
	python -m ruff format --check .

test-fast:
	python -m pytest -q

coverage:
	python -m pytest --cov --cov-branch --cov-report=term-missing --cov-report=xml
	python scripts/agent_harness/coverage_gate.py

harness-check:
	python scripts/agent_harness/validate_harness.py
	python scripts/agent_harness/validate_references.py
	python scripts/agent_harness/validate_pr.py --local

package:
	python -m build
	python -m twine check dist/*

check: lint coverage harness-check package
```

Adjust for existing tools. Do not introduce tool churn in Phase 0 if existing tools already satisfy the role.

## 13. CI policy

### 13.1 Required principles

- Required CI must run on every pull request.
- Required CI must run on pushes to the discovered default branch.
- Required workflows must not use `paths:` filters.
- Use one aggregate job named `ci-required` as the branch-protection required check.
- `ci-required` must depend on all required jobs and fail if any required job fails, is skipped unexpectedly, or is cancelled.
- Optional expensive jobs may run nightly or by manual dispatch.
- Default workflow permissions should be `contents: read` unless a job needs more.
- Use Dependabot or equivalent to keep actions and dependencies current.

### 13.2 Runtime matrix

Build the matrix from discovered package metadata and actual dependency compatibility.

Rules:

- CI and package metadata must not disagree long term.
- If declared runtime support is broader than what currently installs, Phase 0 may keep the known-good matrix and create a follow-up issue.
- Do not silently reduce declared runtime support.
- Do not silently reduce tested operating systems if the repo already tests multiple systems.

### 13.3 Required CI skeleton

This is a template. Replace `<DEFAULT_BRANCH>` and commands with discovered repo facts.

```yaml
name: ci

on:
  pull_request:
  push:
    branches:
      - <DEFAULT_BRANCH>
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  harness-validate:
    name: harness-validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "<HARNESS_PYTHON_VERSION>"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: <repo-specific dev install command>
      - run: python scripts/agent_harness/validate_harness.py
      - run: python scripts/agent_harness/validate_references.py
      - run: python scripts/agent_harness/validate_pr.py --ci

  tests:
    name: tests (${{ matrix.os }}, ${{ matrix.runtime }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [<DISCOVERED_OS_MATRIX>]
        runtime: [<DISCOVERED_RUNTIME_MATRIX>]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        if: ${{ startsWith(matrix.runtime, 'python-') }}
        with:
          python-version: ${{ replace(matrix.runtime, 'python-', '') }}
          cache: pip
      - run: <repo-specific dev install command>
      - run: <repo-specific test command>

  package:
    name: package
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "<HARNESS_PYTHON_VERSION>"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install build twine
      - run: python -m build
      - run: python -m twine check dist/*

  ci-required:
    name: ci-required
    runs-on: ubuntu-latest
    needs: [harness-validate, tests, package]
    if: always()
    steps:
      - name: Fail if required jobs failed, skipped, or were cancelled
        env:
          NEEDS_JSON: ${{ toJson(needs) }}
        run: |
          python - <<'PY'
          import json
          import os
          import sys

          needs = json.loads(os.environ["NEEDS_JSON"])
          bad = {
              name: data["result"]
              for name, data in needs.items()
              if data["result"] != "success"
          }
          if bad:
              print("Required CI jobs did not succeed:", bad)
              sys.exit(1)
          print("All required CI jobs succeeded.")
          PY
```

### 13.4 Nightly workflow

Nightly checks are initially non-blocking. They may include:

- Full optional-dependency tests.
- Slow integration tests.
- Property tests with larger example counts.
- Mutation testing on high-risk modules.
- Downstream smoke tests.
- Docs link checks.
- Dependency pre-release smoke tests.

Nightly failures should create or update issues. Promote a nightly check to required only after it is stable.

### 13.5 Dependabot

Create `.github/dependabot.yml` for ecosystems discovered in the repo.

Minimum for GitHub Actions and Python repos:

```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "area:ci"
      - "agent-ready"

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "area:packaging"
      - "agent-ready"
```

Add other ecosystems only when present.

## 14. GitHub configuration checklist

Create:

```text
docs/agent-harness/branch-protection.md
```

Use this template:

```markdown
# Branch protection checklist

Repository: <repo>
Default branch: <branch>
Last verified: <YYYY-MM-DD>
Verified by: <person/tool>

## Required settings
- [ ] Pull request required before merge
- [ ] Required approvals enabled
- [ ] Stale approvals dismissed on new commits
- [ ] CODEOWNER review required for protected paths
- [ ] Conversations must be resolved
- [ ] Status checks required
- [ ] `ci-required` selected as required check
- [ ] Branch must be up to date before merge or merge queue enabled
- [ ] Direct pushes blocked
- [ ] Force pushes blocked
- [ ] Branch deletion blocked
- [ ] Admin bypass disabled after emergency path is confirmed
- [ ] GitHub Actions default permissions set to read-only where possible
- [ ] Dependency graph enabled, if available
- [ ] Dependabot alerts enabled, if available
- [ ] Dependabot security updates enabled, if available
- [ ] Secret scanning and push protection enabled, if available
- [ ] Release environments configured, if applicable

## Manual steps
1. Go to repository settings.
2. Open branch protection or repository rulesets.
3. Create a rule for `<branch>`.
4. Enable the required settings above.
5. Select `ci-required` after the workflow has run once and the check name exists.
6. Save the rule.
7. Verify a direct push to `<branch>` is blocked.
8. Verify a PR cannot merge while `ci-required` is failing.

## Notes
```

The coding agent may not be able to configure:

- Branch protection or rulesets.
- Security and analysis settings.
- Secrets or environments.
- Trusted publishing.
- Default branch rename.
- GitHub App installation.
- Required checks before the workflow has run at least once.

When blocked, write exact manual steps. Do not claim completion.

## 15. CODEOWNERS

Create `.github/CODEOWNERS` if repository ownership is known. Use the repository owner or team from intake. If unknown, create a commented template and record the blocker.

Recommended protected paths:

```text
# Harness and repository governance require human/admin review.
/.github/                  <OWNER_OR_TEAM>
/.claude/                  <OWNER_OR_TEAM>
/AGENTS.md                 <OWNER_OR_TEAM>
/CLAUDE.md                 <OWNER_OR_TEAM>
/ARCHITECTURE.md           <OWNER_OR_TEAM>
/QUALITY_SCORE.md          <OWNER_OR_TEAM>
/scripts/agent_harness/    <OWNER_OR_TEAM>
/pyproject.toml            <OWNER_OR_TEAM>
/setup.py                  <OWNER_OR_TEAM>
/setup.cfg                 <OWNER_OR_TEAM>
/requirements*.txt         <OWNER_OR_TEAM>
```

Add release, hardware, firmware, calibration, or device-control paths discovered during intake.

## 16. Packaging and dependency policy

### 16.1 Phase 0 packaging rule

Do not migrate packaging backends in Phase 0 unless the issue explicitly requests it. Add tool configuration without changing package semantics.

For Python repos, `pyproject.toml` may contain tool config even if build metadata remains in `setup.py` or `setup.cfg`.

### 16.2 Dependency changes

Agents must not add dependencies casually. A dependency PR must include:

- Reason the dependency is needed.
- Why the standard library or existing dependencies are insufficient.
- License check.
- Import-time and install-time cost.
- Runtime compatibility.
- Downstream impact.
- Dependabot behavior.

### 16.3 Dev dependencies

Prefer adding development dependencies in the least disruptive existing mechanism:

1. Existing dev extra, if present.
2. Existing requirements-dev file, if present.
3. New requirements-dev file, if package metadata is fragile.
4. Optional dev extra, if packaging is stable.

Do not break editable install.

## 17. Test policy

### 17.1 Coverage policy

Definitions:

- **Line coverage:** executable lines run by tests.
- **Branch coverage:** meaningful branches tested.
- **Diff coverage:** changed executable lines covered by tests.
- **Behavior coverage:** public behavior asserted, not merely executed.

Required staged policy:

- Phase 0: record baseline where possible. Do not fail on legacy gaps.
- Phase 1: fail if PR decreases coverage without a waiver issue.
- Phase 2: require diff coverage for changed product code when tooling is stable.
- Phase 3: ratchet total coverage toward the documented target.

Coverage exclusions must be explicit and justified in `docs/agent-harness/coverage-policy.md`.

### 17.2 Test quality rubric

A valid test must:

- Assert desired public behavior.
- Fail on the base commit if it is a regression or feature test.
- Use deterministic data.
- Avoid network dependency unless marked integration and isolated.
- Avoid local machine state.
- State units and coordinate conventions when relevant.
- Use tolerances that catch real regressions without creating platform flakes.
- Avoid implementation details unless the issue targets internal behavior.

Weak tests include:

- Import-only tests.
- Execution without assertions.
- Broad snapshots without semantic checks.
- Exact floating-point assertions without an oracle.
- Expected values recomputed by the same code path under test.
- Random data without a seed.
- Platform skips without an issue.

### 17.3 Numerical tolerance policy

Create `docs/testing/numerical-tolerance-policy.md` with:

- Prefer analytic or independent oracles.
- Use explicit relative and absolute tolerances.
- Explain tolerance source: analytic bound, platform noise, empirical measurement, or legacy baseline.
- Never loosen tolerance in the same PR that changes implementation unless a numerics reviewer approves.
- For image or signal tests, assert shape, dtype, finite values, monotonic or physical invariants, conservation/normalization where applicable, and edge behavior.
- Store golden arrays only when synthetic or analytic tests are insufficient.
- Hash and document binary fixtures.

### 17.4 Fixture policy

New binary fixtures require metadata:

```yaml
name: <fixture file>
created_by: synthetic | hardware | external
creator: <person/tool>
generator: <path/to/script or explanation>
seed: <value or not applicable>
units: <units or not applicable>
license: <license or internal>
sha256: <hash>
expected_behavior:
  - <claim>
```

Large fixtures require a size budget and provenance note.

## 18. Hardware, scientific, and numerical domains

The harness must support hardware development without assuming the hardware domain.

During intake, identify whether the repo contains or interacts with:

- Hardware control code.
- Calibration files.
- Firmware interfaces.
- Scientific algorithms.
- Numerical optimization.
- Image, signal, or sensor processing.
- Unit conversions.
- Coordinate transforms.
- Binary or vendor-specific data formats.
- Generated data or golden fixtures.

For those areas, create `docs/product-specs/README.md` with:

- Known physical quantities and units.
- Coordinate systems and axis order.
- Valid input ranges.
- Invalid input behavior.
- Safety limits.
- Calibration data provenance.
- Known invariants.
- Hardware assumptions.
- Test oracle strategy.

Do not invent scientific facts. Mark unknowns as unknown and create follow-up issues.

## 19. Downstream compatibility

If intake finds downstream consumers, add a downstream policy. If no downstream consumers are known, create a placeholder stating that none were discovered.

For a library repo, downstream smoke should:

1. Build a local package artifact.
2. Create an isolated environment.
3. Install the downstream project with the local artifact.
4. Run a minimal smoke subset.
5. Record logs and versions.

If a change breaks downstream behavior, choose one:

1. Fix compatibility in the current repo.
2. Open coordinated downstream PRs.
3. Document a deliberate breaking change with versioning and release notes, then require human/admin approval.

## 20. Harness validation scripts

### 20.1 `validate_harness.py`

Checks:

- Required files exist for the current phase.
- `AGENTS.md` is present and below max length.
- `CLAUDE.md` points to `AGENTS.md`.
- Required docs exist.
- Required skills exist if `.claude/` is enabled.
- Required subagents exist if `.claude/agents/` is enabled.
- `.claude/worktrees/` is ignored by git.
- `Makefile` exposes required targets.
- `QUALITY_SCORE.md` has required fields.
- No required CI workflow has `paths:` filters.
- Required CI aggregate job `ci-required` exists.
- Protected-path templates exist or blockers are recorded.

### 20.2 `validate_references.py`

Checks:

- Required source URLs are present in `sources.yml`.
- Each source has access date, title, publisher, note path, and checksum or explicit no-snapshot reason.
- Committed snapshots match checksums.
- Notes exist for each required source.
- No unexpected external source is listed without a note explaining why it was used.

### 20.3 `validate_pr.py`

In CI:

- Detect pull request metadata through GitHub event JSON when available.
- Verify linked issue exists in PR body.
- Verify PR template sections are filled.
- Verify required agent run metadata for the enabled phase and risk level.
- Verify source changes have tests or documented test waiver.
- Verify CI changes trigger harness validation.
- Verify release/security/harness changes require owner review metadata where possible.

Locally:

- Warn when GitHub event context is unavailable.
- Validate file-level evidence.
- Never fake GitHub-side facts.

### 20.4 `prove_red_tests.py`

Responsibilities:

- Check out or compare against base commit safely.
- Run only the new/changed tests.
- Confirm failure occurs on base.
- Confirm failure is not due to import error, missing dependency, bad fixture path, or environment failure unless the issue is specifically about that failure.
- Write a concise proof artifact.

### 20.5 `coverage_gate.py`

Responsibilities:

- Read coverage output.
- Load `docs/agent-harness/coverage-baseline.json`.
- Phase 0: create/update baseline only with explicit flag.
- Later phases: fail on coverage decrease unless waiver exists.
- Fail if configured ratchet threshold is missed.

### 20.6 `diff_coverage_gate.py`

Responsibilities:

- Identify changed product lines against PR base.
- Require changed executable lines to be covered after diff coverage enforcement is enabled.
- Allow documented exclusions only.

## 21. Release safety

If the repo publishes packages, firmware, hardware configs, binaries, or documentation artifacts, treat release automation as high risk.

Required release policy:

- Release PR must use `release-guard`.
- Release workflow changes require owner/human approval.
- Publishing requires protected environments.
- Release job must build from a clean tag or protected release branch.
- Release job must test installed artifacts, not only the source tree.
- Dry-run or staging publish must pass before production publish where supported.
- Release notes must include API changes, dependency changes, deprecations, and downstream effects.
- Publish credentials must not be available to ordinary pull request workflows.

Do not modify release publishing behavior in Phase 0 unless the issue explicitly requests it.

## 22. Minimal Phase 0 pull request

If the full Phase 0 scope is too large, this is the minimum acceptable PR:

1. `docs/generated/repo-intake.md`.
2. `AGENTS.md` and `CLAUDE.md`.
3. Core `docs/agent-harness/` docs.
4. `docs/references/agentic-harness/sources.yml` and notes.
5. `scripts/agent_harness/validate_harness.py`.
6. `scripts/agent_harness/validate_references.py`.
7. `.github/workflows/ci.yml` with unfiltered `ci-required`.
8. `.github/pull_request_template.md`.
9. Agent task issue template.
10. `.github/dependabot.yml` where applicable.
11. `Makefile` or equivalent command wrapper.
12. `QUALITY_SCORE.md`.
13. `docs/agent-harness/branch-protection.md` manual checklist.

Do not merge Phase 0 until current tests and `ci-required` pass, or until pre-existing failures are documented and explicitly accepted by the repository owner.

## 23. Phase 0 execution steps

1. Create branch `agent/bootstrap-harness` or another non-default branch.
2. Run repository intake.
3. Write `docs/generated/repo-intake.md`.
4. Read required references and pertinent sublinks.
5. Create reference manifest and notes.
6. Add or update root agent docs.
7. Add harness docs and validation scripts.
8. Add issue and PR templates.
9. Add command wrappers.
10. Update CI with unfiltered `ci-required`.
11. Add Dependabot config for discovered ecosystems.
12. Add `.claude/` artifacts only when syntax can be validated or clearly marked as draft.
13. Run `make harness-check`.
14. Run existing tests.
15. Run package build if supported.
16. Open a PR using the PR template.

Phase 0 PR title:

```text
Add agentic harness bootstrap
```

Phase 0 PR labels:

```text
area:harness
area:ci
area:docs
risk:medium
```

## 24. Merge readiness checklist

A PR is merge-ready only when all applicable items are true:

- [ ] Linked issue exists.
- [ ] Correct risk label is present.
- [ ] Phase is stated.
- [ ] Required agent run metadata exists for the enabled phase.
- [ ] Red tests were proven red for code changes.
- [ ] Implementation did not weaken tests.
- [ ] Adversarial review completed for code changes.
- [ ] Numerics review completed when applicable.
- [ ] Full local check passes or failure reason is documented and accepted.
- [ ] `ci-required` passes.
- [ ] Coverage gate passes if enabled.
- [ ] Package builds and imports from installed artifact, if applicable.
- [ ] Public API changes are documented.
- [ ] Downstream smoke ran when applicable.
- [ ] Branch protection and owner-review requirements are satisfied.
- [ ] Human/admin approval exists for high-risk, release, security, or hardware-impacting changes.

## 25. Known risks and required mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Overclaiming safety | No harness proves absence of bugs. | State limits; use CI, review, branch protection, and human gates. |
| Agent writes weak tests | Tests may encode current behavior or implementation details. | Clean-context red-test role, red-test proof, adversarial review, rubric. |
| Agent cheats tests | Implementation may hard-code fixtures or weaken assertions. | Separate roles, write-scope checks, fixture audit, diff coverage. |
| CI skipped by filters | Required checks can appear successful without running required jobs. | No path filters on required CI; aggregate `ci-required` always runs. |
| Dependency rot | Dependency resolution can break over time. | Runtime matrix, Dependabot, package build tests, explicit support policy. |
| Flaky numerical tests | Platform noise can hide or create failures. | Tolerance policy, deterministic seeds, analytic oracles, numerics review. |
| Docs drift | Agents rely on stale repo docs. | Doc-gardener workflow, harness validation, docs as system of record. |
| Release token exposure | Publishing credentials are high impact. | Protected environments, minimal permissions, no release secrets in PR workflows. |
| Cross-repo breakage | Libraries may have downstream users. | Downstream smoke, coordinated PR policy, release notes. |
| Hook bypass | Local hooks are not a security boundary. | Re-enforce with CI and branch protection. |
| Schema drift | Claude Code artifact schemas may change. | Validate against installed version; record deviations. |

## 26. Follow-up issues after Phase 0

Create these issues after Phase 0, adjusted to the repository facts:

1. **Configure branch protection for current default branch**
   - Labels: `area:harness`, `risk:high`, `needs-human-decision`.
2. **Run clean-context harness smoke issue**
   - Labels: `agent-ready`, `area:harness`, `risk:low`.
3. **Establish coverage baseline and ratchet policy**
   - Labels: `agent-ready`, `area:tests`, `risk:medium`.
4. **Inventory public API and domain invariants**
   - Labels: `agent-ready`, `area:api`, `risk:medium`.
5. **Audit fixtures and generated data**
   - Labels: `agent-ready`, `area:tests`, `risk:medium`.
6. **Align runtime support metadata and CI matrix**
   - Labels: `agent-ready`, `area:packaging`, `risk:medium`.
7. **Add downstream smoke tests if downstream consumers were discovered**
   - Labels: `agent-ready`, `area:downstream`, `risk:medium`.
8. **Harden release workflow if publishing exists**
   - Labels: `area:release`, `risk:high`, `needs-human-decision`.

## 27. Final program acceptance criteria

The harness program is complete for a repository when:

- Default branch is protected.
- Direct push is blocked.
- `ci-required` is required and unskipped.
- Required docs are present and validated.
- Required skills, subagents, hooks, and workflows are present or documented as not applicable.
- Clean-context test-first protocol is enforced for code changes.
- Coverage baseline is recorded and non-decreasing.
- Diff coverage is enforced for changed product code where tooling supports it.
- Package build and installed-artifact smoke tests are required where packaging exists.
- Dependabot or equivalent dependency updates are configured.
- Release workflow is gated where publishing exists.
- Reference manifest is present and valid.
- `QUALITY_SCORE.md` is current.
- Downstream smoke exists where downstream consumers are known.
- A future agent can start from `AGENTS.md`, follow repo docs, and produce a PR that cannot merge unless mechanical gates pass.
