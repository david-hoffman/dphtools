# Project

**Version 1.0** Git versions revisions.

## Purpose and non-goals

`dphtools` is a Python library for optics and image analysis. The owner approved retaining this architecture in the setup conversation on 2026-09-26: “I approve this plan, but with unlimited budget, only come back to me with a real blocker to solve.” The approved plan retained the scientific stack, public interfaces, packaging, and declared Python >=3.8 compatibility. This record transcribes that approved baseline; it does not approve every existing numerical result or a future product task.

Setup adds delivery instructions, verification infrastructure, and a thin doctor command. It adds no product features, web server, browser layer, orchestration platform, permission broker, or additional repository.

## Constraints and boundaries

- One library distribution in this repository. Keep the root product README, notebooks, source layout, Conda recipe/environment, and Setuptools/Versioneer packaging.
- `dphtools.utils` provides array/image operations. Its submodules provide fitting, histogram statistics, drift analysis, registration, signal analysis, and rolling-ball filtering. `dphtools.display` provides Matplotlib visualization. This describes observed responsibilities, not independently approved mathematical contracts.
- Public boundaries are Python imports/functions/classes in those modules. Existing docstrings and tests supply evidence for task intake. Resolve any disagreement between a request, documentation, and implementation before product changes.
- NumPy, pandas, SciPy, Matplotlib, and scikit-image are declared dependencies in `requirements.txt`. No external service or persistent database is required by the library.
- Packaging declares Python >=3.8. Existing CI targets Python 3.10 on Linux, macOS, and Windows. Setup does not silently narrow compatibility; untested interpreter versions remain an explicit verification gap.
- Use synthetic data and Matplotlib's noninteractive `Agg` backend in tests. No production secrets belong in test runs.

## Verify and operate

Black at 99 columns, flake8 critical-error checks, NumPy-style docstrings, pytest/unittest/doctests, and coverage.py are retained. Black was updated from 23.3.0 to 26.5.1 after the setup audit found vulnerabilities; uv 0.12.19 generates the universal hashed verification lock. Dependency locks, appropriate type checks, and dependency security auditing are setup work. Actual commands and results will be recorded below and in [SETUP-001](tasks/SETUP-001.md); no unrun command is evidence of success.

The library has no deployed service. Build wheels/source distributions using the recorded build command. Existing tag-triggered publishing targets PyPI/TestPyPI and Conda; setup must not create release tags or invoke publishing. Release remains an explicit owner action after green verification. Recover a bad release through the package registry's supported controls and a reviewed corrected release; do not claim Git rollback reverses an already published package.

## Harness and delivery

Until the owner changes this direction, target all pull requests to `codex-main` (`gh pr create --base codex-main`). This was requested on 2026-09-28; it does not change the repository's default branch. CI runs for both `main` and `codex-main`.

Use the installed Codex CLI, observed as `0.155.0-alpha.16.4`. Four canonical skills live under `.agents/skills/`. Root `AGENTS.md` is native; no bridge is needed. Start a fresh root role session with:

```sh
codex exec --cd "$PWD" --disable memories --disable multi_agent --sandbox workspace-write --json - < APPROVED_ROLE_PACKET
```

Use `--sandbox read-only` for B/D where the checks can obtain their necessary temporary/report-file access. Never resume/fork a different role or use another role's transcript or automatic personal memory. A tooling clarification may resume the same role; that is not another fresh independent session. Save concise session IDs/results in the task; keep raw local runs out of Git. Retain the configured model unless the owner requests otherwise. A/B do not read this record's implementation references or the task's execution section; give them the separate narrow contract packet.

The invocation is `python tools/delivery doctor [--check]`; `PATH="$PWD/tools:$PATH" delivery doctor` is the equivalent executable command on POSIX systems. Doctor invokes the installed harness once using `review-work` and the canonical `DOCTOR-PROMPT.md`. Default mode proposes a docs-branch patch and waits for approval; `--check` requests inspection only in the native read-only sandbox. This is an agent operation, not a scheduled checker or controller. Commit `f7a7416` implements the launcher and its 39 process-boundary tests pass; the setup task separately records real agent invocation and approval-stop evidence.

## Canonical verification commands

Create an isolated environment with an appropriate installed Python (`python3.10 -m venv .venv-delivery` for the existing CI target, or `python3.13 -m venv .venv`). Install using `python -m pip install --require-hashes -r requirements-dev.lock`. This macOS 27 host needs the Python 3.10 artifact/bootstrap workaround recorded below; its ordinary Python 3.13 environment works. The universal lock carries interpreter/platform markers, including Windows-only dependencies. It is a verification lock, not a narrowing of the library's declared Python >=3.8 metadata.

Use one shared verification command locally and in CI. Fast checks give feedback while editing; the full check must pass before any push, including the 100% statement and branch gate:

```sh
python tools/verification.py fast
python tools/verification.py full
```

The script invokes the existing tools with the selected interpreter and saves their actual commands, statuses, and logs in a fresh `reports/verification/` directory. `fast` runs Black (99 columns), Flake8 (`E9,F63,F7,F82`, critical errors only), and recursive NumPy-style pydocstyle across `dphtools`. `full` adds configured types, the hashed dependency audit, build/install, all tests/doctests, sequential coverage reports, and exact report validation. Missing tools/reports, failed or skipped tests, omitted owned files, exclusions, or incomplete measured coverage fail the command. It retains diagnostics after ordinary tool failures. A host success does not establish Linux/macOS/Windows matrix success; CI repeats `full` on each configured platform. No known local failure may be sent to CI as a candidate.

Ordinary Git hooks under `.githooks/` run `fast` before commit and `full` before push. Inspect `git config --show-origin --get core.hooksPath` and the current hooks directory before installing; preserve any existing hooks. For a clone without active hooks, install with `git config --local core.hooksPath .githooks`. The hooks use the active `python`, or an explicit `DPHTOOLS_PYTHON` executable. They require staged tracked content for commit and a clean checkout of the commit being pushed. They are bypassable local feedback, not permission enforcement. Required CI remains enabled. The [verification contract](tasks/LOCAL-VERIFICATION-CONTRACT.md) describes the public commands and hook behavior.

Installed in this clone on 2026-09-28 after confirming no existing hook configuration or active hooks. Readback: `file:.git/config .githooks`. Both hooks recheck Git state after successful verification; a change during checks requires another run. Other clones must install the hooks separately.

Local measurement note: Python 3.13 skips hidden `.pth` startup files. On this host the checkout's `.venv` coverage hook had that flag, and clearing it did not persist. A fresh environment created by `/Users/davidhoffman/miniconda3/bin/python3.13 -m venv /private/tmp/dphtools-verify-313-jmr47yhf`, followed by its `python -m pip install --require-hashes -r requirements-dev.lock`, restored subprocess measurement. Its 39 launcher tests measured all 24 statements. Use that environment's `bin/python` for the canonical commands on this host; the ordinary hosted CI environments also measure the launcher. This changes no dependency, test, or coverage rule. See Python's [startup-file handling](https://github.com/python/cpython/blob/3.13/Lib/site.py) and coverage.py's [subprocess documentation](https://coverage.readthedocs.io/en/7.16.1/subprocess.html).

On this host, use `/private/tmp/dphtools-verify-313-jmr47yhf/bin/python tools/verification.py full` and set `DPHTOOLS_PYTHON=/private/tmp/dphtools-verify-313-jmr47yhf/bin/python` for Git operations. The script sets `MPLBACKEND=Agg` and `PYTHONHASHSEED=0` itself. Its recursive owned-source discovery includes never-imported `dphtools` modules, `tools/*.py` descendants, and `tools/delivery`; extend discovery for any new runtime outside those directories. Coverage paths combine copied CLI fixtures into `tools`; subprocess measurement is enabled. Global 100% with zero missing statements/branches implies every included file and package is complete; the validator checks exact per-file counts. Generated `dphtools/_version.py` is the only omitted runtime file. Vendor Versioneer tooling, notebooks, and tests are not product/runtime coverage targets; handwritten module demo/error paths remain included. Python coverage cannot measure the shell hooks' statements/branches; process-boundary tests do not erase that reported measurement limit.

Python 3.10 verification also works locally after selecting the already-locked `scipy-1.15.3-cp310-cp310-macosx_12_0_arm64.whl` (SHA-256 `ad3432cb0f9ed87477a8d97f03b763fd1d57709f1bbde3c9369b1dff5503b253`); the default macOS-14 artifact has a malformed Mach-O section. The standalone interpreter's nested virtual environments also lose their library/stdlib paths during pip-audit bootstrap. A complete disposable copy of the existing Python distribution, with its own locked dependencies and process-local loader fallback, resolves that separately. Original runtimes, system settings, lock and audit flags remain unchanged. Artifact acquisition, prefix-copy/install commands and failed attempts are recorded in `reports/coverage-continuation/python310-environment.md` and `reports/coverage-continuation/python310-audit-recovery.md`.

The actual additional full-check command on this host is:

```sh
DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/dphtools-python310-prefix-pijntny1/python/lib \
  /private/tmp/dphtools-python310-prefix-pijntny1/python/bin/python3.10 tools/verification.py full
```

Both interpreters ran the same canonical verifier on runtime commit `9189914`. All tests and noncoverage checks passed; the 100% coverage gate still failed. Exact results belong in the setup record. These two local interpreter runs do not establish the hosted operating-system matrix or Python 3.8 compatibility.

Type checking is gradual, not a claim that this predominantly unannotated library is fully typed. SciPy's dynamic, untyped exports produced false missing-attribute reports for working public imports; the narrowly named SciPy import boundary is skipped by mypy and remains a static-checking gap. Owned annotated code still reports errors. Public numerical tests, not a suppressed type error, must establish the real behavior.

Regenerate the single verification lock deliberately with the pinned uv, rather than during CI:

```sh
uv pip compile --universal --python-version 3.10 --generate-hashes --no-strip-extras requirements-dev.in -o requirements-dev.lock
```

Tool choices were checked against primary [Black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html), [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html), [coverage.py](https://coverage.readthedocs.io/en/latest/config.html), [mypy](https://mypy.readthedocs.io/en/stable/existing_code.html), [uv](https://docs.astral.sh/uv/pip/compile/), and [pip-audit](https://github.com/pypa/pip-audit) documentation. Baseline results, limits, and subsequent green evidence belong in the setup record, not in claims inferred from these commands.

## Approval and remaining decisions

The owner approved the setup plan and architecture baseline in the conversation on 2026-09-26, replacing the proposed two-hour cap with unlimited budget. No reliable dollar metering is available. The default one implementation repair remains. Future tasks still need their own approval.

No new product behavior is approved here. The existing coverage baseline, scientific behavior in untested branches, actual supported interpreter matrix, GitHub reviewer requirements, and setup demonstrations must be reported honestly before calling the system ready. See [GitHub setup](agentic-software-delivery-v1.0/GITHUB-SETUP.md) and [setup evidence](tasks/SETUP-001.md).
