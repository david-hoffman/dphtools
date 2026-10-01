# Public Actions caller and fixture interface

This admits external calling conventions for the already approved S4/S5 workflows. It does not supply implementation logic or an expectation oracle. Correct outcomes derive from the separate approved public contract: trusted main identity, frozen source/workflow/artifact bytes, required reports and installations before owner approval, conflict-safe recovery, and completed published checks before finalization. These commands are real Actions callers, not test-only exports.

## Commands

All commands use `python tools/release_workflow.py COMMAND`. The workflow uses Python 3.10.

- `resolve --version VERSION [--resume RUN_ID]`
- `artifact --origin-run RUN_ID --artifact-id ID --artifact-digest DIGEST --workflow-sha SHA`
- `bind --manifest FILE --dist DIR --version VERSION --source-sha SHA --workflow-sha SHA --origin-run RUN_ID`
- `summary`: all `bind` arguments, plus `--artifact-id ID --artifact-digest DIGEST`.
- `tag`: all `bind` arguments, plus `--manifest-digest DIGEST`.
- `finalize`: all `tag` arguments, plus `--receipt FILE`.

`resolve` reads Actions context `GITHUB_REPOSITORY`, `GITHUB_REF`, `GITHUB_EVENT_NAME`, `TRUSTED_WORKFLOW_SHA`, `GITHUB_RUN_ID`, `GH_TOKEN`, and `GITHUB_OUTPUT`. `summary` writes `GITHUB_OUTPUT` and `GITHUB_STEP_SUMMARY`. GitHub transport uses `GH_TOKEN`; `finalize` also reads `GITHUB_RUN_ID`. Fixture tokens are fake; never use real credentials.

Actions output is appended as UTF-8 `key=value` lines. `resolve` keys are `version`, `channel`, `source_sha`, `workflow_sha`, `origin_run`, and `resume` (text `true` or `false`). Recovery additionally returns `artifact_id`, `artifact_digest`. `summary` returns `manifest_digest` (lowercase SHA-256 of manifest bytes) and appends Markdown approval evidence including source, workflow, originating run, version/channel, manifest hash, retained artifact ID/archive hash, and manifest contents. No exact Markdown or success stdout is prescribed.

## External fixtures

Git and gh are ordinary external executables. Git transport reads `rev-parse HEAD`, fetches `--no-tags origin main`, checks `merge-base --is-ancestor SHA FETCH_HEAD`, and reads Git object identities with `rev-parse SHA:PATH`. The latter PATH values are `.github/workflows/make_release.yml`, `tools/release.py`, `tools/release_workflow.py`, and the measurement repair's `tools/release_probe.py`; SHA identifies original/current trusted revision. Object IDs are opaque public metadata. Do not read owner source/history. Isolated Git fixtures or portable fake Git transport may supply known commit/blob identities and real status codes; they must not supply owned release logic results.

GitHub JSON uses argument-list `gh api ENDPOINT --method METHOD --header "X-GitHub-Api-Version: 2026-03-10"`; POST/PATCH JSON may use stdin with `--input -`. Direct HTTPS JSON transport may use `Authorization: Bearer GH_TOKEN`, that API-version header, and `Accept: application/vnd.github+json`. Transport choice is not an expectation oracle.

Endpoint prefix is `repos/david-hoffman/dphtools/`:

| Endpoint | Public response shape |
| --- | --- |
| `actions/runs/RUN_ID` | Original approved run schema: id, repository.full_name, path, head_branch, event, head_sha, status, conclusion, html_url |
| `actions/runs/RUN_ID/artifacts?per_page=100` | Object with artifacts array; records id, name, expired, digest, workflow_run.id, workflow_run.head_sha |
| `actions/artifacts/ID` | Artifact record with those fields |
| `actions/artifacts/ID/zip` | Raw ZIP archive bytes on gh stdout |
| `git/ref/tags/VERSION` | object.type and object.sha, or explicit HTTP 404 |
| `git/refs` POST | Request ref and sha; reference response |
| `releases/tags/VERSION` | Release id, tag_name, prerelease, or explicit HTTP 404 |
| `releases/RELEASE_ID/assets?per_page=100` | Array of asset name and digest records |
| `releases` POST | Request tag_name, target_commitish, name, body, draft, prerelease; release response |
| `releases/RELEASE_ID` PATCH | Request draft=false |

Assets use POST `https://uploads.github.com/repos/david-hoffman/dphtools/releases/RELEASE_ID/assets?name=FILENAME`, Content-Type application/octet-stream and `--input FILE`. Exact upload/PATCH response body is not prescribed. Explicit HTTP 404 is distinct from malformed/failed/ambiguous external state.

Retained artifact name is `release-bundle`; digest syntax is `sha256:` plus archive SHA-256. ZIP member paths have no `bundle/` prefix: `manifest.json`, `dist/FILENAME`, `reports/PLATFORM/checks.json`, `notes.md`. Local extraction is into `bundle/`. Manifest/report/distribution schemas remain the original contract. There is no separate recovery-summary JSON file. Recovery's workflow_sha identifies original run commit; TRUSTED_WORKFLOW_SHA identifies current invocation. Unrelated main progress must not retarget the source; changed workflow/helper bytes require renewed preparation and approval under the original policy.

Explicit local caller layout: report lookup is `MANIFEST.parent / "reports"`, independent of working directory or dist directory. Thus a valid fixture may place its manifest at `bundle/manifest.json`, packages at `bundle/dist/`, and reports at `bundle/reports/PLATFORM/.../checks.json` (exactly one canonical checks.json per platform). `bind`, `summary`, `tag` and `finalize` consume that existing layout. The `artifact` command validates origin/metadata/archive-byte identity only; it does not extract or create a bundle directory. The workflow's separate pinned Actions downloader performs extraction. A controlled end-to-end fixture may extract its own independently prepared archive into this layout; never substitute a fake successful owned command result. No artifact command output file or extraction behavior is prescribed.

`--manifest-digest` identifies manifest bytes separately. The receipt file represents post-publication evidence; no extra acceptance schema is prescribed here. A valid fixture may use repository, version, channel, source_sha, workflow_sha, run_id, files, verification, published, missing. Finalization attaches the two original distributions, manifest and receipt. Hosted ordering and approval are D/setup evidence, not something a receipt or mock can authorize.

## Downloaded registry surface

`python tools/release.py reconcile --manifest FILE --dist DIR --output DIR [--require-complete] [--downloaded DIR]` adds an optional destination for verified published bytes, consumed by post-upload smoke. The destination is absent or empty; retained names match published distribution filenames. Staging still includes only missing originals. Downloaded bytes must match the original manifest. Do not expose corrupt bytes as verified files. Safe partial results need not be atomically erased on a later failure; no new all-or-nothing behavior is prescribed.

Registry JSON entries expose filename, digests.sha256, url. HTTPS fixtures may use standard urllib public request/response/redirect interfaces: Request.full_url; response context manager, geturl(), getcode(), read(); HTTP error code; redirect status, headers (Location), body stream, target URL. A standard HTTPSHandler transport fixture can preserve urllib's real request/error/redirect processing while controlling external IO. The approved rule remains selected official HTTPS registry/file hosts and no arbitrary local/private fetch. Do not impose a particular private handler or owned symbol. Diagnose a legitimate alternate transport as an adapter issue before blaming product.

Primary fixture references verified by coordinator: [Git references](https://docs.github.com/en/rest/git/refs), [Actions artifacts](https://docs.github.com/en/rest/actions/artifacts), [GitHub releases](https://docs.github.com/en/rest/releases/releases), [PyPI JSON](https://docs.pypi.org/api/json/). No live mutation is permitted for tests.


## Additional caller fixture metadata

The executable tools/release_workflow.py defines a no-argument main() callable consumed by its script entry. It reads sys.argv in the same way as the script. A harness caller may load the opaque file as a Python module with the containing tools directory available for imports, then invoke main() with the declared CLI arguments and interpret its returned process status. This names a real production entry, not an implementation oracle or a newly required internal architecture. Test CLI behavior through that entry using ordinary opaque module loading; do not inspect function/source text. Importing an entry must not perform remote writes before the public command is invoked. Expectations for invoked commands remain the approved public command contract. Exact private function names besides this fixture entry are not contractual.

Valid registry metadata may be returned through a standard HTTPS redirect to another path on the same selected official host. Preserve the external standard-library redirect fixture: record requests, respond with an HTTP redirect, serve final bytes, and use existing raw byte/digest outcomes. Unsafe-host rejection remains covered and unchanged. This fixture represents ordinary external HTTP transport; do not return an owned success result.

A successful new preparation requires version availability on the chosen registry; an already-present version must use original-run recovery, not new preparation. Controls differ only in external registry presence; verify no remote write on rejection.

The approved post-upload artifact installation stage is credential-free. Real clean artifact installs must not pass inherited registry/GitHub credentials to candidate installation/probe subprocesses. Use dummy sentinel values only, and observe actual subprocess environments without substituting process results. Preserve public dependency-cache settings and actual clean installed checks. Do not read or print real ambient credentials.


## Installed-probe measurement fixture metadata

The owner approved one additional corrective cycle for honest installed-probe measurement and confirmed implementation PR base codex-main. The prior corrective cycle remains used. Production release sources remain main only. Scope/scenarios/expected numerical behavior remain unchanged.

The actual owned installed probe will be the executable tools/release_probe.py. Its caller is ENVIRONMENT_PYTHON ABSOLUTE_HELPER_PATH VERSION, using the interpreter of the actual new disposable environment. It consumes one version argument, performs no installation/network/publication/credential operation, and returns zero only if the approved actual installed checks pass; nonzero useful diagnostics block smoke. No JSON/wording/internal-function schema is required. Smoke keeps its existing public CLI.

Policy outcomes are unchanged: real separate retained wheel/source installations in clean environments outside the checkout; matching metadata/package versions; dphtools, utils and declared dependency imports beneath the isolated prefix; noninteractive Matplotlib; approved sum0+1+2+3=6; Git executable unavailable for source installation; credential isolation, cleanup and unchanged retained bytes. A helper wrong-version or bad installed-state rejection needs a successful real control first. Don't manufacture success or add a new numerical policy.

The opaque actual tools tree is already copied by accepted worker fixtures. Helper source must never be read for expectations. Run the actual file with the actual installation interpreter and retain ordinary child coverage outside disposable environments. It is the production probe, not a testing duplicate. A raw coverage record can be observed for the declared helper filename and recorded child execution; don't open source, render covered statements, inspect code/AST/disassembly or use statement/branch line listings as an oracle. Full honest100% enforcement remains the unchanged canonical verifier and C/D evidence, not a blind-role guarantee from nonempty records.

Fixture instrumentation must supply the existing locked coverage tool/startup in the actual disposable environment only during instrumented tests, preserving real subprocess results and installed-package/dependency origins. Neither production release runtime nor artifacts gain a coverage dependency or test switch. Instrumentation must not introduce checkout/parent dependency imports, exclude owned files/branches, synthesize line/arc records, or hide setup failure. Use ordinary coverage with existing config/path mapping; retain data for normal combine/report. Unsupported measurement is reported.

Recovery's external git rev-parse SHA:PATH metadata now includes tools/release_probe.py alongside .github/workflows/make_release.yml, tools/release.py and tools/release_workflow.py. Matching helper blobs permit recovery; a changed probe blob must reject without publication or retargeting. Blob identities remain opaque external Git metadata, not source. All other declared transport schemas stay the same.

Test fixture PROCESS_DRIVER observes real environment/install/check/probe processes. It may be corrected for this new actual probe invocation and provision test-only instrumentation, but must continue delegating real processes, independent installed behavior and real fault outcomes. Existing scenario assertions stay; changing a fixture requires fresh B acceptance. Git transport fixture may admit the fourth path and expose matching/conflicting opaque identity controls. No test correction weakens coverage/checks or grants another C cycle.
