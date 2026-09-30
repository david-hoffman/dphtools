#!/usr/bin/env python3
"""Trusted Actions orchestration; never import the candidate library here."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

from release import (
    REPOSITORY,
    WORKFLOW,
    check,
    commit,
    digest,
    fetch,
    gh_api,
    require,
    run_id,
    validate_run,
    version_info,
    verification,
)


def git(*args, cwd=None):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"Git failed: {result.stderr}")
    return result.stdout.strip()


def api_optional(endpoint):
    """Only an explicit HTTP 404 is absence; all other errors stop the run."""
    request = urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "X-GitHub-Api-Version": "2026-03-10",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return None


def output(values):
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            require("\n" not in str(value) and "\r" not in str(value), "Unsafe Actions output")
            stream.write(f"{key}={value}\n")


def main_member(source):
    git("fetch", "--no-tags", "origin", "main")
    git("merge-base", "--is-ancestor", commit(source), "FETCH_HEAD")


def resolve(args):
    require(
        os.environ["GITHUB_REPOSITORY"] == REPOSITORY
        and os.environ["GITHUB_REF"] == "refs/heads/main"
        and os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch",
        "Only main dispatch in the approved repository is eligible",
    )
    workflow_sha = commit(os.environ["TRUSTED_WORKFLOW_SHA"])
    info = version_info(args.version)
    if args.resume:
        identity = run_id(args.resume)
        record = validate_run(gh_api(f"repos/{REPOSITORY}/actions/runs/{identity}"), identity)
        main_member(record["head_sha"])
        for path in (WORKFLOW, "tools/release.py", "tools/release_workflow.py"):
            require(
                git("rev-parse", f"{record['head_sha']}:{path}")
                == git("rev-parse", f"{workflow_sha}:{path}"),
                "Trusted workflow/helper files changed; recovery needs renewed preparation review",
            )
        workflow_sha = record["head_sha"]
        artifacts = gh_api(f"repos/{REPOSITORY}/actions/runs/{identity}/artifacts?per_page=100")
        bundles = [item for item in artifacts["artifacts"] if item["name"] == "release-bundle"]
        require(
            len(bundles) == 1 and not bundles[0]["expired"],
            "Original artifact missing, ambiguous, or expired; never rebuild",
        )
        artifact = bundles[0]
        require(
            type(artifact["id"]) is int
            and artifact["id"] > 0
            and artifact["workflow_run"]["id"] == identity
            and artifact["workflow_run"]["head_sha"] == workflow_sha,
            "Wrong artifact run identity",
        )
        require(
            artifact.get("digest", "").startswith("sha256:"), "Missing retained artifact digest"
        )
        output(
            {
                **info,
                "source_sha": record["head_sha"],
                "workflow_sha": workflow_sha,
                "origin_run": identity,
                "artifact_id": artifact["id"],
                "artifact_digest": artifact["digest"],
                "resume": "true",
            }
        )
    else:
        source = git("rev-parse", "HEAD")
        main_member(source)
        require(
            api_optional(f"repos/{REPOSITORY}/git/ref/tags/{args.version}") is None,
            "Release tag already exists; use original-run recovery",
        )
        host = "pypi.org" if info["channel"] == "pypi" else "test.pypi.org"
        try:
            fetch(f"https://{host}/pypi/dphtools/{args.version}/json", host)
        except urllib.error.HTTPError as error:
            require(error.code == 404, "Registry state is ambiguous")
        else:
            raise ValueError("Registry version already exists; use original-run recovery")
        output(
            {
                **info,
                "source_sha": source,
                "workflow_sha": workflow_sha,
                "origin_run": run_id(os.environ["GITHUB_RUN_ID"]),
                "resume": "false",
            }
        )


def artifact(args):
    identity = run_id(args.origin_run)
    artifact_id = run_id(args.artifact_id)
    record = gh_api(f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}")
    require(
        record["id"] == artifact_id
        and record["name"] == "release-bundle"
        and not record["expired"]
        and record["workflow_run"]["id"] == identity
        and record["workflow_run"]["head_sha"] == commit(args.workflow_sha)
        and record.get("digest") == args.artifact_digest,
        "Wrong original artifact identity",
    )
    result = subprocess.run(
        ["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"],
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, "Original artifact download failed")
    require(
        "sha256:" + digest(result.stdout) == args.artifact_digest,
        "Original artifact archive digest mismatch",
    )


def bind(args):
    payload = check(args.manifest, args.dist)
    require(
        payload["version"] == args.version
        and payload["source_sha"] == args.source_sha
        and payload["workflow_sha"] == args.workflow_sha
        and payload["run_id"] == run_id(args.origin_run),
        "Bundle does not match selected workflow/source/run/version",
    )
    require(
        verification(Path(args.manifest).parent / "reports") == payload["verification"],
        "Retained report digests changed",
    )
    main_member(payload["source_sha"])
    return payload


def summary(args):
    payload = bind(args)
    manifest_digest = digest(Path(args.manifest).read_bytes())
    output({"manifest_digest": manifest_digest})
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
        stream.write("## Retained release bundle awaiting owner approval\n\n")
        stream.write(
            f"Original run: {payload['run_id']}; source: `{payload['source_sha']}`; trusted workflow: `{payload['workflow_sha']}`.\n\n"
        )
        stream.write(
            f"Version: `{payload['version']}`; destination: **{payload['channel']}**; manifest SHA-256: `{manifest_digest}`.\n\n"
        )
        stream.write(
            f"Artifact ID: `{args.artifact_id}`; archive digest: `{args.artifact_digest}`.\n\n"
        )
        stream.write("```json\n" + json.dumps(payload, indent=2) + "\n```\n\n")
        stream.write(
            "Python 3.10 matrix only; declared Python >=3.8 compatibility is broader. Shell/YAML coverage and hosted approval enforcement remain unmeasured.\n"
        )


def tag(args):
    payload = bind(args)
    require(
        digest(Path(args.manifest).read_bytes()) == args.manifest_digest,
        "Approved manifest changed",
    )
    endpoint = f"repos/{REPOSITORY}/git/ref/tags/{payload['version']}"
    existing = api_optional(endpoint)
    if existing is not None:
        require(
            existing["object"]["type"] == "commit"
            and existing["object"]["sha"] == payload["source_sha"],
            "Existing tag targets different source",
        )
    else:
        gh_api(
            f"repos/{REPOSITORY}/git/refs",
            {"ref": "refs/tags/" + payload["version"], "sha": payload["source_sha"]},
        )


def finalize(args):
    payload = bind(args)
    require(
        digest(Path(args.manifest).read_bytes()) == args.manifest_digest,
        "Approved manifest changed",
    )
    tag_record = api_optional(f"repos/{REPOSITORY}/git/ref/tags/{payload['version']}")
    require(
        tag_record is not None
        and tag_record["object"]["type"] == "commit"
        and tag_record["object"]["sha"] == payload["source_sha"],
        "Finalization tag mismatch",
    )
    release = api_optional(f"repos/{REPOSITORY}/releases/tags/{payload['version']}")
    if release is None:
        release = gh_api(
            f"repos/{REPOSITORY}/releases",
            {
                "tag_name": payload["version"],
                "target_commitish": payload["source_sha"],
                "name": payload["version"],
                "body": payload["notes"]
                + f"\n\nOriginal evidence: https://github.com/{REPOSITORY}/actions/runs/{payload['run_id']}\nManifest SHA-256: {args.manifest_digest}\nPublished installation: https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                "draft": True,
                "prerelease": payload["channel"] == "testpypi",
            },
        )
    require(
        release["tag_name"] == payload["version"]
        and release["prerelease"] == (payload["channel"] == "testpypi"),
        "Existing GitHub Release identity mismatch",
    )
    files = [Path(args.dist) / item["filename"] for item in payload["files"]] + [
        Path(args.manifest),
        Path(args.receipt),
    ]
    assets = gh_api(f"repos/{REPOSITORY}/releases/{release['id']}/assets?per_page=100")
    for path in files:
        matches = [asset for asset in assets if asset["name"] == path.name]
        require(len(matches) <= 1, "Ambiguous release asset")
        if matches:
            # Do not replace a previous upload. GitHub's retained asset digest is required.
            require(
                matches[0].get("digest") == "sha256:" + digest(path.read_bytes()),
                "Existing GitHub asset bytes conflict",
            )
            continue
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release['id']}/assets?name={path.name}",
                "--header",
                "Content-Type: application/octet-stream",
                "--input",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(result.returncode == 0, f"GitHub asset upload failed: {result.stderr}")
    gh_result = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{REPOSITORY}/releases/{release['id']}",
            "--input",
            "-",
        ],
        input=json.dumps({"draft": False}),
        capture_output=True,
        text=True,
        check=False,
    )
    require(gh_result.returncode == 0, f"GitHub Release finalization failed: {gh_result.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("resolve", allow_abbrev=False)
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--resume", default="")
    retained = commands.add_parser("artifact", allow_abbrev=False)
    for field in ("origin-run", "artifact-id", "artifact-digest", "workflow-sha"):
        retained.add_argument("--" + field, required=True)
    for name in ("bind", "summary", "tag", "finalize"):
        command = commands.add_parser(name, allow_abbrev=False)
        for field in ("manifest", "dist", "version", "source-sha", "workflow-sha", "origin-run"):
            command.add_argument("--" + field, required=True)
        if name == "summary":
            command.add_argument("--artifact-id", required=True)
            command.add_argument("--artifact-digest", required=True)
        if name in ("tag", "finalize"):
            command.add_argument("--manifest-digest", required=True)
        if name == "finalize":
            command.add_argument("--receipt", required=True)
    args = parser.parse_args()
    try:
        if args.command == "resolve":
            resolve(args)
        elif args.command == "artifact":
            artifact(args)
        elif args.command == "bind":
            bind(args)
        elif args.command == "summary":
            summary(args)
        elif args.command == "tag":
            tag(args)
        else:
            finalize(args)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Release workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
