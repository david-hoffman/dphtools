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

Use the installed Codex CLI, observed as `0.155.0-alpha.16.4`. Four canonical skills live under `.agents/skills/`. Root `AGENTS.md` is native; no bridge is needed. Start a fresh root role session with:

```sh
codex exec --cd "$PWD" --disable memories --disable multi_agent --sandbox workspace-write --json - < APPROVED_ROLE_PACKET
```

Use `--sandbox read-only` for B/D where the checks can obtain their necessary temporary/report-file access. Never resume/fork a different role or use another role's transcript or automatic personal memory. A tooling clarification may resume the same role; that is not another fresh independent session. Save concise session IDs/results in the task; keep raw local runs out of Git. Retain the configured model unless the owner requests otherwise. A/B do not read this record's implementation references or the task's execution section; give them the separate narrow contract packet.

The planned invocation is `python tools/delivery doctor [--check]`; `PATH="$PWD/tools:$PATH" delivery doctor` is the equivalent executable command on POSIX systems. Doctor invokes the installed harness once using `review-work` and the canonical `DOCTOR-PROMPT.md`. Default mode proposes a docs-branch patch and waits for approval; `--check` writes nothing. This is an agent operation, not a scheduled checker or controller. The setup task records when implementation and real invocation have actually passed.

## Canonical verification commands

Create an isolated environment with an appropriate installed Python (`python3.10 -m venv .venv-delivery` for the existing CI target, or `python3.13 -m venv .venv` on this macOS 27 host). The initial local Python 3.10/SciPy 1.15.3 wheel could not load on macOS 27; current Python 3.13 wheels do load. This environment gap is not a product regression reproduction. Install using `python -m pip install --require-hashes -r requirements-dev.lock`. The universal lock carries interpreter/platform markers, including Windows-only dependencies. It is a verification lock, not a narrowing of the library's declared Python >=3.8 metadata.

With that environment's `python` active, run the same commands as CI, cheap checks first:

```sh
python -m black --check --line-length 99 dphtools tests tools/delivery setup.py versioneer.py notebooks
python -m flake8 dphtools tests tools/delivery setup.py versioneer.py
python -m pydocstyle --count dphtools
python -m mypy --follow-untyped-imports dphtools tools/delivery
python -m pip_audit --require-hashes -r requirements-dev.lock
python -m build --no-isolation
python -m pip install --no-deps --no-build-isolation .
python -m coverage erase
MPLBACKEND=Agg python -m coverage run -m pytest --doctest-modules dphtools tests -ra --junitxml=reports/pytest.xml
python -m coverage combine
python -m coverage json -o reports/coverage.json dphtools/*.py dphtools/utils/*.py tools/delivery
python -m coverage xml -o reports/coverage.xml dphtools/*.py dphtools/utils/*.py tools/delivery
python -m coverage report --fail-under=100 dphtools/*.py dphtools/utils/*.py tools/delivery
```

The environment assignment is POSIX syntax; CI uses Bash on every target. Report commands return failure below 100%; run each to preserve diagnostics even after an earlier failure. CI uses separate unconditional report steps without suppressing failure. Explicit owned-source globs include never-imported modules; extend them when introducing a new runtime directory. Coverage paths combine copied CLI artifacts back into `tools/delivery`; subprocess measurement is enabled. Global 100% with zero missing statements/branches implies every included file and package is complete; the JSON retains exact counts. Generated `dphtools/_version.py` is the only omitted runtime file. Vendor Versioneer tooling, notebooks, and tests are not product/runtime coverage targets; handwritten module demo/error paths remain included.

Type checking is gradual, not a claim that this predominantly unannotated library is fully typed. SciPy's dynamic, untyped exports produced false missing-attribute reports for working public imports; the narrowly named SciPy import boundary is skipped by mypy and remains a static-checking gap. Owned annotated code still reports errors. Public numerical tests, not a suppressed type error, must establish the real behavior.

Regenerate the single verification lock deliberately with the pinned uv, rather than during CI:

```sh
uv pip compile --universal --python-version 3.10 --generate-hashes --no-strip-extras requirements-dev.in -o requirements-dev.lock
```

Tool choices were checked against primary [Black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html), [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html), [coverage.py](https://coverage.readthedocs.io/en/latest/config.html), [mypy](https://mypy.readthedocs.io/en/stable/existing_code.html), [uv](https://docs.astral.sh/uv/pip/compile/), and [pip-audit](https://github.com/pypa/pip-audit) documentation. Baseline results, limits, and subsequent green evidence belong in the setup record, not in claims inferred from these commands.

## Approval and remaining decisions

The owner approved the setup plan and architecture baseline in the conversation on 2026-09-26, replacing the proposed two-hour cap with unlimited budget. No reliable dollar metering is available. The default one implementation repair remains. Future tasks still need their own approval.

No new product behavior is approved here. The existing coverage baseline, scientific behavior in untested branches, actual supported interpreter matrix, GitHub reviewer requirements, and setup demonstrations must be reported honestly before calling the system ready. See [GitHub setup](agentic-software-delivery-v1.0/GITHUB-SETUP.md) and [setup evidence](tasks/SETUP-001.md).
