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

Existing Black 23.3.0 at 99 columns, flake8 critical-error checks, NumPy-style docstrings, pytest/unittest/doctests, and coverage.py are retained. Dependency locks, appropriate type checks, and dependency security auditing are setup work. Actual commands and results will be recorded below and in [SETUP-001](tasks/SETUP-001.md); no unrun command is evidence of success.

The library has no deployed service. Build wheels/source distributions using the recorded build command. Existing tag-triggered publishing targets PyPI/TestPyPI and Conda; setup must not create release tags or invoke publishing. Release remains an explicit owner action after green verification. Recover a bad release through the package registry's supported controls and a reviewed corrected release; do not claim Git rollback reverses an already published package.

## Harness and delivery

Use the installed Codex CLI, observed as `0.155.0-alpha.16.4`. Four canonical skills live under `.agents/skills/`. Root `AGENTS.md` is native; no bridge is needed. Start a fresh root role session with:

```sh
codex exec --cd "$PWD" --disable memories --disable multi_agent --sandbox workspace-write --json - < APPROVED_ROLE_PACKET
```

Use `--sandbox read-only` for B/D. Do not use `resume`, `fork`, another role's transcript, or automatic personal memory. Save concise session IDs/results in the task; keep raw local runs out of Git. Retain the configured model unless the owner requests otherwise. A/B do not read this record's implementation references or the task's execution section; give them the separate narrow contract packet.

The planned invocation is `python tools/delivery doctor [--check]`; `PATH="$PWD/tools:$PATH" delivery doctor` is the equivalent executable command on POSIX systems. Doctor invokes the installed harness once using `review-work` and the canonical `DOCTOR-PROMPT.md`. Default mode proposes a docs-branch patch and waits for approval; `--check` writes nothing. This is an agent operation, not a scheduled checker or controller. The setup task records when implementation and real invocation have actually passed.

## Approval and remaining decisions

The owner approved the setup plan and architecture baseline in the conversation on 2026-09-26, replacing the proposed two-hour cap with unlimited budget. No reliable dollar metering is available. The default one implementation repair remains. Future tasks still need their own approval.

No new product behavior is approved here. The existing coverage baseline, scientific behavior in untested branches, actual supported interpreter matrix, GitHub reviewer requirements, and setup demonstrations must be reported honestly before calling the system ready. See [GitHub setup](agentic-software-delivery-v1.0/GITHUB-SETUP.md) and [setup evidence](tasks/SETUP-001.md).
