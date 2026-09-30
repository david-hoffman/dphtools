# Release operation

Release sources are `main` only. Stable versions publish to PyPI; `aN`, `bN`, and
`rcN` prereleases publish to TestPyPI. Conda is not a release destination. Versions
use unprefixed `MAJOR.MINOR.PATCH` syntax, such as `1.0.0` or `1.0.0rc1`.
These examples do not select the next release version.

The implementation task and its readiness are recorded in
[RELEASE-AUTOMATION-001](tasks/RELEASE-AUTOMATION-001.md). The commands below are the
approved interface being installed by that task. Do not treat the process as
active until its implementation and the activation requirements below are complete.

## Activate once

1. Integrate the reviewed release workflow and helpers into `main`. Confirm that
   `main` is the repository default branch and retains the required `ci-required`
   check. Retire the old tag-triggered publishing workflow before any public tag.
   [GitHub requires a manually dispatched workflow to exist on the default branch.](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
2. Configure both GitHub publication environments, `pypi` and `test-pypi`. Require
   the owner's review, disable administrator bypass, and permit only the branch
   `main`. For this solo-maintainer process, allow the requester to approve their
   own deployment. Read-only inspection on 2026-09-30 found that both environments
   prohibit self-review and have no branch restriction; implementation alone does
   not change those settings.
   [GitHub environment controls](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
3. Register a Trusted Publisher separately on PyPI and TestPyPI for package
   `dphtools`: repository owner `david-hoffman`, repository `dphtools`, workflow
   filename `make_release.yml`, and the matching environment name. Verify the
   identity before declaring publication ready.
   [Publisher registration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
   [publisher operation and TestPyPI](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
4. After confirming their remaining uses, retire the dedicated Anaconda release
   secret and superseded package-registry tokens. Preserve the development
   environment and historical releases. Credential deletion and remote settings
   changes require a separately authorized activation action.
5. Inspect an actual hosted preparation run, its required platform checks, retained
   files, and approval pause. A TestPyPI upload rehearsal requires separate owner
   authorization. Local tests do not establish hosted approval or registry access.

## Prepare and inspect

Choose the exact version and finish the intended consolidation PR into `main`
before preparing its release. Merge approval and publication approval are separate.
Do not create a public version tag to request preparation.

```sh
python tools/delivery release prepare --version 1.0.0 --ref main
python tools/delivery release status --run RUN_ID
```

Preparation freezes the source commit, trusted workflow revision, version, channel,
originating workflow run, and retained artifact identity. It runs canonical full
verification on Linux, macOS, and Windows, builds the final version using a local
tag, and checks fresh installations of the retained wheel and source archive
outside the checkout. Only the identified Linux-built pair is promoted.

Inspect the workflow summary before approving its protected publication job:

- Version, registry destination, source commit, and trusted workflow revision.
- Release notes, compatibility changes, and known verification limits.
- Successful full verification and retained-artifact installation checks on every
  required platform.
- Original run/artifact identity, manifest, filenames, sizes, and SHA-256 digests.

The existing Python >=3.8 declaration exceeds the configured Python 3.10 matrix.
The workflow does not establish compatibility with every declared interpreter.
Python coverage also does not measure shell or YAML execution.

## Approve and publish

Approve the exact displayed bundle through the protected GitHub environment. Until
that approval, the process must not create a remote tag, upload a package, or create
a GitHub Release. A changed version, source, workflow identity, destination, or
bundle requires renewed preparation and approval.

Publication checks the retained bytes and source membership in `main`, creates the
tag at the frozen commit, and uploads only absent original files. It does not
rebuild. Separate jobs without publishing credentials download the published bytes,
compare their digests, and check installations. GitHub Release finalization follows
successful published-file checks. An ordinary later commit on `main` does not
retarget the release.

## Recover a partial run

Retain the originating run ID and bundle. Inspect the failed stage and registry
state before retrying. Resume from that identity:

```sh
python tools/delivery release prepare --version 1.0.0 --resume RUN_ID
```

Matching published filenames and bytes count as completed. Only absent files may
be retried from the retained bundle. A conflicting digest or ambiguous service
response stops recovery. Never overwrite a release, rebuild an expired approved
bundle, force-move its tag, or use generic duplicate suppression as evidence of
success. If original artifacts are unavailable, stop recovery. A new preparation
rejects an existing tag or registry version, as required by the
[approved recovery policy](tasks/RELEASE-AUTOMATION-001.md#publication-permissions-and-recovery).
If either already exists, the owner must choose recovery for that partial release
and a new version before preparing a replacement bundle. New preparation and
approval cannot silently replace the expired bundle for the same version.

If package uploads completed but finalization failed, recovery verifies the
published files and installations before finalizing, without uploading them again.
Native reruns can require another environment review; do not bypass it.

A published installation defect needs an explicit owner recovery decision and a
reviewed corrected version. Git rollback does not remove registry files. Preserve
the failure evidence and describe the actual partial state.

## Installed workflow identity and diagnostics

The top-level `make_release.yml` workflow owns Trusted Publishing. It records the
trusted `github.workflow_sha` separately from the frozen package source. Recovery
replays that original trusted helper revision, the original workflow-dispatch run, and its
unexpired `release-bundle` artifact ID. Each consumer verifies the original archive
SHA-256 digest before downloading by run and artifact ID, then validates manifest,
package, and report digests. Unrelated progress on `main` may continue: the current top-level workflow and
helper files must match the original Git blobs, and the original source must still
belong to `main`. Changed workflow/helper files require an explicit maintenance
and recovery decision; they cannot silently replace approved code.

The optional dispatch `notes` input supplies release notes and compatibility
impact. The CLI's default request leaves a visible owner-review placeholder.
Replace that placeholder through manual preparation before approving publication,
or explicitly review its limits. Do not infer compatibility from commit messages.

Preparation retains full platform reports even when verification fails. Retained
and published installation jobs retain their logs. Publication attempts retain a
per-file receipt when reconciliation completes. Upload errors remain failed or
partial outcomes; recovery rechecks the registry before staging missing files.
Finalization keeps the GitHub Release in draft until its assets are attached. An
existing asset must have the same digest; recovery never overwrites it.

The workflow serializes publication without cancelling partial uploads. The
protected publisher parses metadata and uploads retained bytes; it never installs
or imports the candidate package. Only it receives OpenID Connect (OIDC) permission.
Its GitHub write permission creates the frozen tag. Post-upload installation jobs
have read access only. The finalizer gets GitHub write permission only after every
published installation job passes. Tag creation uses this same dependency graph;
it does not depend on a token-created tag triggering another workflow.

Both environments still need the separately authorized activation changes listed
above: allow requester approval, retain the owner as sole reviewer, keep admin
bypass disabled, restrict deployment to `main`, and register the exact repository,
workflow, and environment with the matching registry. YAML and local tests do not
prove hosted enforcement. No remote setting, secret, tag, or publication changes
are part of installing this implementation.
