# Local verification

**Version 1.0** Setup infrastructure authorized by the owner's 2026-09-28 instruction to verify locally before pushing and have CI verify the submitted result. No new numerical behavior is approved here.

## Public behavior

- `python tools/verification.py fast` checks Black formatting at 99 columns, the retained Flake8 critical-error selection, and NumPy docstring style throughout owned `dphtools` submodules. It uses the invoking interpreter and locates the repository from the script, so invocation does not depend on the caller's current directory.
- `python tools/verification.py full` runs those checks, configured type checks, locked dependency audit, distribution build/install, the complete tests/doctests, and sequential coverage combination/reporting. Installation must consume a distribution produced by that invocation; a stale artifact or separate source rebuild does not verify the build output. It requires 100% measured statements and branches for all owned runtime files, including never-imported files and the verification helper. Only generated `_version.py` remains omitted.
- Missing or invalid mode returns exit 2. A successful check returns 0; any failed tool or report validation returns 1. Every scheduled step is attempted after ordinary child-command failures so test/coverage diagnostics are retained. Failure is never converted into success by a later passing step.
- Importing the Python verification module is inert: it must not launch tools or create run reports. Checking, building, and installation require an explicit command invocation. If the operating system cannot launch one scheduled child process, report that failure, retain diagnostics, and attempt the remaining checks.
- Each invocation creates a fresh directory under `reports/verification/`, identifies it in output, and saves individual tool logs plus `checks.json` with document version `1.0`, mode, Python interpreter, platform, and step names/commands/return codes. Full mode retains `pytest.xml`, `coverage.json`, and `coverage.xml`. Existing reports must not establish success for a new run.
- Full verification rejects missing, malformed, empty, skipped, failed, or errored test results. It rejects missing/malformed coverage reports, absent owned files, disabled branch measurement, exclusions, and any exact missing statement or branch count. Rounded percentages do not establish completeness. Zero instrumentable branch opportunities are allowed and are not a claim about every logical path.
- Coverage must actually instrument executed Python files at every owned directory depth, as well as report never-imported files. For example, executed modules under `dphtools/subpkg/`, `dphtools/utils/subpkg/`, and `tools/subpkg/` must contribute their covered statements. The generated omission applies only to the repository's `dphtools/_version.py`; an identically named handwritten file elsewhere is still measured.
- The tools run as real subprocesses. Their output and failures remain diagnosable, including output bytes that are not valid text in the host's default encoding. Retain the actual child exit status, represent undecodable bytes visibly, and continue the scheduled checks. A local success verifies this interpreter/platform; CI repeats the same command on its configured operating-system matrix.

## Ordinary local Git hooks

- `.githooks/pre-commit` runs fast checks before a commit. It rejects unstaged tracked changes so the files checked correspond to the staged tracked content.
- `.githooks/pre-push` requires a clean working tree, including ordinary untracked files, and requires every non-deletion source revision being pushed to equal the checked-out commit. It then runs full verification once and rejects the push on failure. It must not contact the remote itself.
- Both hooks select `${DPHTOOLS_PYTHON:-python}` as the interpreter. Missing/broken tools fail visibly. Existing hooks/configuration must be inspected before installation and preserved if present.
- After successful verification, recheck the relevant Git state before allowing the operation: commit must still check the same staged content with no unstaged tracked edits; push must still have the same HEAD and a clean tracked/untracked checkout. A save or tool write during checks invalidates that result. This detects ordinary stale results; it is not an atomic snapshot or a guarantee against concurrent changes.
- Hooks are ordinary bypassable feedback, not permission enforcement or certification. Required CI remains. The coordinator does not push a known failing local candidate or bypass the hook.

## Independent test scope

Test the real command/hook entry points. Controlled external-tool stand-ins are appropriate for deterministic failure/report cases; do not replace the verifier under test or mock a numerical algorithm. Temporary fixture repositories must be isolated from this checkout and require no network or remote push. Exercise a real local Git push only to a temporary local bare repository when useful.

These are tests added during setup, not retroactive original test-first evidence. A/B may read this contract, the existing delivery-command tests for platform fixture conventions, and their own tests. They must not inspect the verifier/hook implementation, source history, other sessions, or private gap lists. Copying the executable bytes opaquely into an isolated fixture is allowed. Preserve existing product/doctor tests and all runtime/configuration files.
