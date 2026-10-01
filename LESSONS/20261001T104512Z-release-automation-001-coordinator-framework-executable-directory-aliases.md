# Framework interpreter paths can canonicalize only the directory

- ID: 20261001T104512Z-release-automation-001-coordinator-framework-executable-directory-aliases
- Date: 2026-10-01T10:45:12Z
- Task/role: RELEASE-AUTOMATION-001 / coordinator
- Status: confirmed
- Observation: A clean framework-Python venv launched through /tmp/PREFIX/bin/python reported /private/tmp/PREFIX/bin/python and the correct canonical venv prefix. Absolute path spelling equality rejected this legitimate selected interpreter.
- Evidence: Non-owned CPython 3.9 diagnostic `reports/release-automation/roles/macos-framework-stdlib-identity-diagnostic.json`, all six alias/control invocations exited zero. Tagged [CPython 3.10.11 launcher](https://raw.githubusercontent.com/python/cpython/v3.10.11/Mac/Tools/pythonw.c) canonicalizes the directory while preserving the executable symlink name. [CI run 36847833267](https://github.com/david-hoffman/dphtools/actions/runs/36847833267) has eleven executable-observer failures, but its actual receipt path values were not retained, so their precise cause remains unproven.
- Lesson: Accept legitimate directory aliases while preserving selected venv directory, executable basename, prefix, nonce, arguments, actual process status and invocation-specific coverage evidence. Resolving the entire executable can collapse separate venv symlinks onto a shared global binary. Keep discriminating controls and include source-free actual/expected paths in mismatch diagnostics. Require new complete verification; do not infer the hosted cause from this independent counterexample.
