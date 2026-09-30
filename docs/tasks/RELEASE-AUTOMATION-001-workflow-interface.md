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

Git and gh are ordinary external executables. Git transport reads `rev-parse HEAD`, fetches `--no-tags origin main`, checks `merge-base --is-ancestor SHA FETCH_HEAD`, and reads Git object identities with `rev-parse SHA:PATH`. The latter PATH values are `.github/workflows/make_release.yml`, `tools/release.py`, `tools/release_workflow.py`; SHA identifies original/current trusted revision. Object IDs are opaque public metadata. Do not read owner source/history. Isolated Git fixtures or portable fake Git transport may supply known commit/blob identities and real status codes; they must not supply owned release logic results.

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

`--manifest-digest` identifies manifest bytes separately. The receipt file represents post-publication evidence; no extra acceptance schema is prescribed here. A valid fixture may use repository, version, channel, source_sha, workflow_sha, run_id, files, verification, published, missing. Finalization attaches the two original distributions, manifest and receipt. Hosted ordering and approval are D/setup evidence, not something a receipt or mock can authorize.

## Downloaded registry surface

`python tools/release.py reconcile --manifest FILE --dist DIR --output DIR [--require-complete] [--downloaded DIR]` adds an optional destination for verified published bytes, consumed by post-upload smoke. The destination is absent or empty; retained names match published distribution filenames. Staging still includes only missing originals. Downloaded bytes must match the original manifest. Do not expose corrupt bytes as verified files. Safe partial results need not be atomically erased on a later failure; no new all-or-nothing behavior is prescribed.

Registry JSON entries expose filename, digests.sha256, url. HTTPS fixtures may use standard urllib public request/response/redirect interfaces: Request.full_url; response context manager, geturl(), getcode(), read(); HTTP error code; redirect status, headers (Location), body stream, target URL. A standard HTTPSHandler transport fixture can preserve urllib's real request/error/redirect processing while controlling external IO. The approved rule remains selected official HTTPS registry/file hosts and no arbitrary local/private fetch. Do not impose a particular private handler or owned symbol. Diagnose a legitimate alternate transport as an adapter issue before blaming product.

Primary fixture references verified by coordinator: [Git references](https://docs.github.com/en/rest/git/refs), [Actions artifacts](https://docs.github.com/en/rest/actions/artifacts), [GitHub releases](https://docs.github.com/en/rest/releases/releases), [PyPI JSON](https://docs.pypi.org/api/json/). No live mutation is permitted for tests.
