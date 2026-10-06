# Project

**Version 1.0** Git versions revisions.

## Purpose and non-goals

`dphtools` is a Python library for optics and image analysis. The owner approved retaining this architecture in the setup conversation on 2026-09-26: “I approve this plan, but with unlimited budget, only come back to me with a real blocker to solve.” The approved plan retained the scientific stack, public interfaces, packaging, and declared Python >=3.8 compatibility. This record transcribes that approved baseline; it does not approve every existing numerical result or a future product task.

Setup adds delivery instructions, verification infrastructure, and a thin doctor command. It adds no product features, web server, browser layer, orchestration platform, permission broker, or additional repository.

## Constraints and boundaries

- One library distribution in this repository. Keep the root product README, notebooks, source layout, development environment, and Setuptools/Versioneer packaging. On 2026-09-30 the owner authorized removing the Conda release process, including its publication recipe; this supersedes the earlier recipe-preservation constraint. The development environment remains separate from publishing.
- `dphtools.utils` provides array/image operations. Its submodules provide fitting, histogram statistics, drift analysis, registration, signal analysis, and rolling-ball filtering. `dphtools.display` provides Matplotlib visualization. This describes observed responsibilities, not independently approved mathematical contracts.
- Public boundaries are Python imports/functions/classes in those modules. Existing docstrings and tests supply evidence for task intake. Resolve any disagreement between a request, documentation, and implementation before product changes.
- NumPy, pandas, SciPy, Matplotlib, and scikit-image are declared dependencies in `requirements.txt`. No external service or persistent database is required by the library.
- Packaging declares Python >=3.8. Existing CI targets Python 3.10 on Linux, macOS, and Windows. Setup does not silently narrow compatibility; untested interpreter versions remain an explicit verification gap.
- Use synthetic data and Matplotlib's noninteractive `Agg` backend in tests. No production secrets belong in test runs.

## Verify and operate

Black at 99 columns, Flake8 critical-error checks, NumPy-style docstrings, pytest/unittest/doctests, coverage.py, gradual type checks, and dependency auditing are installed. `requirements-dev.in` and the universal hashed `requirements-dev.lock` record tool versions; regenerate with pinned uv only when deliberately updating dependencies. [SETUP-001](tasks/SETUP-001.md) and [DELIVERY-MAINTENANCE-001](tasks/DELIVERY-MAINTENANCE-001.md) retain historical installation/reconciliation evidence; [DELIVERY-EFFICIENCY-001](tasks/DELIVERY-EFFICIENCY-001.md) governs current efficiency work. Old measurements never establish a later candidate's readiness.

The library has no deployed service. Build wheels/source distributions using the recorded build command. On 2026-09-30 the owner selected `main` as the sole release source and PyPI as the sole production package destination, and authorized removal of Conda publishing. The release workflow candidate replaces tag-triggered publishing with explicit preparation and protected publication. Activation and hosted approval enforcement remain separate setup work; an installed workflow alone does not establish readiness. [RELEASE-AUTOMATION-001](tasks/RELEASE-AUTOMATION-001.md) records the proposal and its current state. The [release runbook](RELEASING.md) records commands, activation requirements, approval, and recovery. Setup must not create release tags or invoke publishing. Release remains an explicit owner action after green verification. Recover a bad release through the package registry's supported controls and a reviewed corrected release; do not claim Git rollback reverses an already published package.

## Harness and delivery

The owner clarified on 2026-09-30 that `codex-main` is a playground and production releases come from `main`. Ordinary playground work retains the 2026-09-28 PR target of `codex-main` (`gh pr create --base codex-main`). The owner plans a consolidation PR into `main` containing a major release bump; that PR targets `main` and is an explicit exception to the earlier all-PRs-to-`codex-main` direction. This plan does not authorize creating or merging that PR now, changing the repository default branch, or publishing a release. CI runs for both branches. Verify the actual post-merge `main` commit before release; the exact major version remains to be selected.

Use the installed Codex CLI, checked as `0.155.0-alpha.16.4` for DELIVERY-MAINTENANCE-001. Four canonical skills live under `.agents/skills/`. Root `AGENTS.md` is native; no bridge is needed. Start a fresh root role session with:

```sh
codex exec --cd "$PWD" --disable memories --disable multi_agent --sandbox workspace-write --json - < APPROVED_ROLE_PACKET
```

Use `--sandbox read-only` for B/D where the checks can obtain their necessary temporary/report-file access. Never resume/fork a different role or use another role's transcript or automatic personal memory. A tooling clarification may resume the same role; that is not another fresh independent session. Save concise session IDs/results in the task; keep raw local runs out of Git. Retain the configured model unless the owner requests otherwise. A/B do not read this record's implementation references or the task's execution section; give them the separate narrow contract packet.

Routine work uses an author and one fresh independent reviewer, reusing established contracts and narrow relevant context. Specialist A/B/C/D are reserved for new scientific contracts/custom correctness oracles, release/publication safety, or other material behavior/security risk; classify by changed behavior and lost proof. The task/PR records the route and rationale. Repository quiet/event-wait preferences cannot override mandatory host progress updates or supply unavailable tools. Use supported native CI completion facilities after protection is verified; keep full logs outside Git and return concise structured results.

The invocation is `python tools/delivery doctor [--check]`; `PATH="$PWD/tools:$PATH" delivery doctor` is the equivalent executable command on POSIX systems. Doctor invokes the installed harness once using `review-work` and the canonical `DOCTOR-PROMPT.md`. Default mode proposes a docs-branch patch and waits for approval; `--check` requests inspection only in the native read-only sandbox. This is an agent operation, not a scheduled checker or controller. Historical real invocation and approval-stop demonstrations are linked from SETUP-001; current regression evidence belongs to the active task.

## Canonical verification commands

Create an isolated environment with an appropriate installed Python (`python3.10 -m venv .venv-delivery` for the existing CI target), then invoke that environment's interpreter and install using `python -m pip install --require-hashes -r requirements-dev.lock`. Verify the selected interpreter and actual nested environment/install operation on the current host; old disposable prefixes are not portable commands. The universal lock carries interpreter/platform markers, including Windows-only dependencies. It is a verification lock, not a narrowing of the library's declared Python >=3.8 metadata.

Use a dedicated Python installation, such as the CI setup-python runtime, for repeatable suite timings. Runtime reuse checks fingerprint both the selected environment and its base interpreter from fresh bytes; an ambient base installation with unrelated packages adds work even when the selected environment is isolated. Worker scratch files use the operating system's temporary location outside the checkout.

Real nested-environment regression tests use `system_site_packages=True` and need the locked verification tools available in the base interpreter, as CI's direct setup-python installation provides. Installing them only in an outer virtual environment does not expose them to these nested environments. Verify this boundary before running the complete suite.

Use the shared verifier locally and in CI. Routine PR opening/reopening (drafts included) and updates require meaningful focused checks and `fast` on the exact candidate. `full` is the local/reference path, useful for integration, diagnosis, and scientific/safety work when the approved risk/check plan requires it; it is not mandatory before every routine PR:

```sh
python tools/verification.py preflight
python tools/verification.py fast
python tools/verification.py full
```

The script uses the selected interpreter and streams child output while saving atomic, digest-bound version 2.0 receipts after each step in a fresh `reports/verification/` directory. `preflight` checks the lock, imports, and an actual disposable copied environment with pip installation before expensive work. Dependent work is blocked after failed prerequisites; available fresh coverage diagnostics still run after test failure. `fast` runs Black (99 columns), Flake8 (`E9,F63,F7,F82`, critical errors only), and recursive NumPy-style pydocstyle across `dphtools`. `full` adds configured types, the hashed dependency audit, build/install, all tests/doctests, sequential coverage reports, and exact report validation. Missing tools/reports, failed or skipped tests, omitted owned files, exclusions, or incomplete coverage fail the command. Preserve useful failure diagnostics. A host success does not establish Linux/macOS/Windows success. Merge requires the complete platform matrix, exact 100% owned statements/branches, fresh independent review, and an unchanged reviewed candidate. Rely on the stable `ci-required` aggregate as the authoritative merge gate only after actual required-check protection on `codex-main` has been read back and verified; missing protection is a merge blocker. Preserve production `main` protections and publication approval. Current activation evidence and blockers belong in [DELIVERY-EFFICIENCY-001](tasks/DELIVERY-EFFICIENCY-001.md), not in claims inferred from workflow text. Known failures block readiness; pending CI remains pending. Record candidate, commands, environment, and results outside the tracked tree, and rerun invalidated evidence. Pre-PR backups may retain failing checkpoints when fast/state checks pass; label them incomplete.

Ordinary Git hooks under `.githooks/` run `fast` before both commit and push; focused and risk-based full checks are separate invocations. Hooks do not query remote PR state. Inspect `git config --show-origin --get core.hooksPath` and the current hooks directory before installing; preserve any existing hooks. For a clone without active hooks, install with `git config --local core.hooksPath .githooks`. The hooks use the active `python`, or an explicit `DPHTOOLS_PYTHON` executable. Pre-commit rejects unstaged tracked changes. Pre-push requires a clean tracked/untracked checkout and each non-deletion source revision to equal HEAD. They are bypassable local feedback, not permission enforcement. Required platform CI remains mandatory before merge. The [verification contract](tasks/LOCAL-VERIFICATION-CONTRACT.md) describes the public commands and hook behavior.

Inspect hook activation in each clone/worktree; do not assume a historical `core.hooksPath=.githooks` setting exists on the current host. Preserve existing settings and use a per-command hook-path override when appropriate. Changing shared Git configuration can affect other worktrees. Both hooks recheck the relevant index/HEAD/checkout state after successful checks; a change during checks invalidates the result. Other clones need their own inspected installation.

The verifier sets `MPLBACKEND=Agg` and `PYTHONHASHSEED=0`. Its recursive owned-source discovery includes never-imported `dphtools` modules, `tools/*.py` descendants, and `tools/delivery`; extend discovery for new owned runtime outside those directories. Coverage paths combine copied CLI fixtures into `tools`, and subprocess measurement must actually work in the selected environment. Exact global and per-file 100% statements/branches establish measured completeness across included packages. Generated `dphtools/_version.py` is the only omitted runtime file. Vendor Versioneer tooling, notebooks, and tests are outside owned runtime measurement; handwritten demo/error paths remain included. Python coverage cannot measure shell-hook statements/branches; process-boundary tests do not erase that reported limit.

Historical macOS Python 3.10/3.13 artifact, nested-environment, and startup-hook repairs are recorded in [SETUP-001](tasks/SETUP-001.md) and [DELIVERY-MAINTENANCE-001](tasks/DELIVERY-MAINTENANCE-001.md). Their disposable paths and local receipts are evidence for those hosts/revisions, not current setup guarantees. Recreate a locked isolated environment and verify real subprocess measurement on each new host. Local interpreter runs establish neither the hosted platform matrix nor Python 3.8 compatibility.

Type checking is gradual, not a claim that this predominantly unannotated library is fully typed. SciPy's dynamic, untyped exports produced false missing-attribute reports for working public imports; the narrowly named SciPy import boundary is skipped by mypy and remains a static-checking gap. Owned annotated code still reports errors. Public numerical tests, not a suppressed type error, must establish the real behavior.

Regenerate the single verification lock deliberately with the pinned uv, rather than during CI:

```sh
uv pip compile --universal --python-version 3.10 --generate-hashes --no-strip-extras requirements-dev.in -o requirements-dev.lock
```

Tool choices were checked against primary [Black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html), [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html), [coverage.py](https://coverage.readthedocs.io/en/latest/config.html), [mypy](https://mypy.readthedocs.io/en/stable/existing_code.html), [uv](https://docs.astral.sh/uv/pip/compile/), and [pip-audit](https://github.com/pypa/pip-audit) documentation. Candidate results and measurement limits belong in the active task's evidence, not in claims inferred from these commands.

CI uses `collect`, `shard`, and `aggregate` in separate jobs, with four isolated machines for Linux and macOS, eight for Windows, and four bounded local workers per machine. Every real pytest/doctest node executes exactly once per platform, and each platform combines its actual parent and child coverage independently. The measured platform duration seeds are advisory; legacy universal seeds remain valid and unknown tests receive equal default weight. The public sharded CLI defaults to two machines and supports sealed counts of two, four or eight. Ordinary `full` uses up to eight bounded local subprocess workers and always runs fresh. `full --workers 1` retains serial execution. Each local worker has private temporary paths, reports and parent/child coverage; the immutable checkout and locked interpreter are shared. [TEST-SUITE-PERFORMANCE-001](tasks/TEST-SUITE-PERFORMANCE-001.md) records this performance maintenance.

Optional `fast --reuse /absolute/path/to/checks.json` can reuse only a closed successful docstring check with unchanged complete inputs, runtime bytes, command, and retained logs. Unknown imports, external import paths, unaudited site hooks/customizations, or invalid evidence rerun it. Only the byte-identified stock setuptools/coverage startup hooks in the unchanged lock are eligible. Reused steps retain their original command and receipt provenance, with no invented new return code. Audits, builds, installation, full tests, CI, and release evidence never use this cache. See [the verification contract](tasks/LOCAL-VERIFICATION-CONTRACT.md) for identities and limitations. `python tools/verification_metrics.py TASK_RECORDS.json` summarizes timings, repeated checks, agent launches, runner seconds, available tokens, and defects; missing data and uncontrolled comparisons stay labeled. Ten comparable tasks are required to complete the pilot.

For repeatable clean-install tests, prepare an ordinary wheel directory once from the exact verification lock. CI already uses this setup. Installers still create private environments and perform real package and dependency installation; no installed environment or full-suite result is reused:

```sh
python -m pip download --require-hashes --only-binary=:all: -r requirements-dev.lock --dest reports/verification-wheels
python -m pip install --no-index --find-links reports/verification-wheels --require-hashes -r requirements-dev.lock
```

Then set `PIP_NO_INDEX=1`, `PIP_FIND_LINKS` to that directory's absolute path, and `PIP_COMPILE=0` for verification. Keep audit networking available: offline package resolution does not replace the dependency audit. `PIP_COMPILE=0` suppresses installation-time bytecode compilation; ordinary imports still execute the installed sources. Select writable cache directories when the host's home directory is read-only. These settings and the invoking interpreter remain part of the measured environment.


## Approval and remaining decisions

The owner approved the setup plan and architecture baseline in the conversation on 2026-09-26, replacing the proposed two-hour cap with unlimited budget. No reliable dollar metering is available. Repair allowances are tracked per approved task and inherited on corrections/splits; this project record does not replenish them. Future tasks still need their own approval.

No new product behavior is approved here. DELIVERY-MAINTENANCE-001 records the owner-approved installation reconciliation; prior exhausted scientific repair allowances remain exhausted. Report supported interpreter/platform limits, GitHub protection limits, and missing setup demonstrations honestly. See [GitHub setup](agentic-software-delivery-v1.0/GITHUB-SETUP.md) and the historical [setup evidence](tasks/SETUP-001.md).
