# SETUP-001: Doctor public contract for blind test roles

**Version 1.0** Approved setup scope: owner approved the five-step setup plan on 2026-09-26 and specified unlimited budget. This narrow packet records the requested doctor behavior and ordinary CLI interface choices. It contains no implementation solution or history. Read only this packet, your selected skill, test conventions below, and authorized tests/fixtures. Do not read the full setup record, implementation, history, or lessons.

## Observable requirements

- R1: From any working directory, `python /absolute/repo/tools/delivery doctor` starts one installed `codex exec` session rooted at that repository, using `review-work` doctor mode and that repository's `docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md`. It must work when the repository path contains spaces. Do not invoke a shell to interpret arguments.
- R2: Default doctor asks the agent to inspect actual evidence and make an evidenced documentation patch on a docs branch, then wait for owner approval before commit/push/merge. It must use the canonical prompt's procedure. It does not schedule, retry, create parallel agents, or implement the procedure as a custom controller.
- R3: `doctor --check` clearly selects inspection-only mode, passes `--check` intent to the agent, and uses Codex's read-only sandbox. Default mode uses workspace-write. Disable optional Codex memory and multi-agent delegation for both: the installed CLI supports `--disable memories --disable multi_agent` or equivalent `-c features.memories=false -c features.multi_agent=false`. Each invocation is fresh (no resume/fork).
- R4: Forward harness output and exit status so failure cannot appear successful. Missing `codex` gives a readable diagnostic on stderr and nonzero status (127). Missing/unreadable canonical prompt gives a readable error and nonzero status before launching the harness. A failing harness is not retried.
- R5: `--help` and `doctor --help` describe usage and exit 0 without launching a harness. Unknown subcommands/options or absent command fail with usage and exit 2 without launching a harness. Unsupported invocations do not write project files.

## Test conventions and limits

Use pytest with standard-library subprocess/temp-directory facilities. Tests live in `tests/test_delivery.py`; test-only fixtures may be colocated or in `tests/fixtures/delivery/`. Test the actual command and observable process output, status, and filesystem effects. An external fake `codex` executable is allowed only at the harness boundary to observe transport/failure behavior without network or paid agent calls. Do not mock away the launcher path.

Do not mistake argument/prompt assertions for proof of actual agent behavior. A separate real doctor invocation must demonstrate the documentation patch and approval stop. Tests that run before implementation demonstrate absence of the new setup helper, not reproduction of a product bug. A minimal executable scaffold prints a clearly labeled “not implemented” diagnostic and exits nonzero so red tests can reach the real entry point without import/dependency failure.

The prepared environment is `.venv-delivery/bin/python`; run `python -m pytest -q tests/test_delivery.py` through that interpreter. Use Black 23.3.0 at 99 columns. Prefer portable Python fixtures over OS-specific shell behavior. Report useful gaps; do not invent product requirements or edit workflows/coverage settings. The helper's public command contract is the test target; no library numerical behavior is being changed.
