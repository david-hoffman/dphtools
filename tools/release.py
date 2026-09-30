#!/usr/bin/env python3
"""Validate retained release evidence and distributions without publishing."""

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPOSITORY = "david-hoffman/dphtools"
WORKFLOW = ".github/workflows/make_release.yml"
PLATFORMS = ("ubuntu-24.04", "macos-15", "windows-2025")
STEPS = (
    "format",
    "lint",
    "docstrings",
    "types",
    "audit",
    "build",
    "install",
    "coverage-erase",
    "tests",
    "coverage-combine",
    "coverage-json",
    "coverage-xml",
    "coverage-report",
    "report-validation",
)
DEPENDENCIES = {"numpy", "pandas", "scipy", "matplotlib", "scikit-image"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def version_info(version):
    require(
        matches(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?",
            version,
        ),
        "Invalid canonical release version",
    )
    return {"version": version, "channel": "testpypi" if re.search(r"[a-z]", version) else "pypi"}


def run_id(value):
    require(matches(r"[1-9][0-9]*", value), "Run ID must be a positive decimal integer")
    return int(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def commit(value):
    require(matches(r"[0-9a-f]{40}", value), "Invalid commit identity")
    return value


def safe_path(name):
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and ":" not in name
        and not name.startswith("/")
        and ".." not in PurePosixPath(name).parts,
        "Unsafe archive or file path",
    )


def metadata_check(data, version):
    metadata = BytesParser().parsebytes(data)
    require(metadata.get_all("Name") == ["dphtools"], "Package name must be dphtools")
    require(metadata.get_all("Version") == [version], "Package version mismatch")
    require(metadata.get_all("Requires-Python") == [">=3.8"], "Python requirement changed")
    requirements = metadata.get_all("Requires-Dist", [])
    require(
        len(requirements) == len(DEPENDENCIES) and set(requirements) == DEPENDENCIES,
        "Package dependency declarations changed",
    )


def archive_check(path, version):
    if path.name.endswith(".whl"):
        require(
            matches(
                r"dphtools-"
                + re.escape(version)
                + r"-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl",
                path.name,
            ),
            "Wheel filename mismatch",
        )
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            for name in names:
                safe_path(name)
            require(len(names) == len(set(names)), "Duplicate archive member")
            require(archive.testzip() is None, "Corrupt wheel")
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            require(len(metadata_names) == 1, "Ambiguous wheel metadata")
            metadata_check(archive.read(metadata_names[0]), version)
    else:
        require(path.name == f"dphtools-{version}.tar.gz", "Source archive filename mismatch")
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            for member in members:
                safe_path(member.name)
                require(
                    member.isfile() or member.isdir(),
                    "Source archive contains links or special files",
                )
                require(
                    ".git" not in PurePosixPath(member.name).parts,
                    "Source archive contains Git metadata",
                )
            require(len(names) == len(set(names)), "Duplicate archive member")
            metadata_names = [
                name for name in names if name.count("/") == 1 and name.endswith("/PKG-INFO")
            ]
            require(len(metadata_names) == 1, "Ambiguous source metadata")
            metadata_check(archive.extractfile(metadata_names[0]).read(), version)


def distributions(directory, version):
    paths = sorted(Path(directory).iterdir())
    require(
        len(paths) == 2
        and sum(path.name.endswith(".whl") for path in paths) == 1
        and sum(path.name.endswith(".tar.gz") for path in paths) == 1,
        "Expected exactly one wheel and one source archive",
    )
    records = []
    for path in paths:
        require(path.is_file() and not path.is_symlink(), "Distribution must be a regular file")
        archive_check(path, version)
        content = path.read_bytes()
        records.append({"filename": path.name, "size": len(content), "sha256": digest(content)})
    return records


def verification(directory):
    result = {}
    for platform in PLATFORMS:
        paths = list((Path(directory) / platform).rglob("checks.json"))
        require(len(paths) == 1, f"Missing or ambiguous report for {platform}")
        report = read_json(paths[0])
        require(
            isinstance(report, dict)
            and set(
                (
                    "document_version",
                    "mode",
                    "python",
                    "platform",
                    "steps",
                    "failed",
                    "measurement_limits",
                )
            )
            <= report.keys(),
            "Missing canonical report fields",
        )
        require(
            report["document_version"] == "1.0"
            and report["mode"] == "full"
            and isinstance(report["python"], str)
            and isinstance(report["platform"], str)
            and report["failed"] == []
            and isinstance(report["measurement_limits"], list),
            "Invalid full verification summary",
        )
        steps = report["steps"]
        require(isinstance(steps, list), "Invalid verification steps")
        names = []
        for step in steps:
            require(
                isinstance(step, dict) and {"name", "command", "returncode"} <= step.keys(),
                "Invalid step record",
            )
            require(
                type(step["returncode"]) is int
                and step["returncode"] == 0
                and (
                    step["command"] is None
                    or isinstance(step["command"], list)
                    and all(isinstance(arg, str) for arg in step["command"])
                ),
                "Failed or malformed verification step",
            )
            names.append(step["name"])
        require(sorted(names) == sorted(STEPS), "Missing or duplicate full verification step")
        result[platform] = digest(paths[0].read_bytes())
    return result


def check(manifest, directory):
    payload = read_json(manifest)
    require(
        isinstance(payload, dict)
        and {
            "schema_version",
            "repository",
            "version",
            "channel",
            "source_ref",
            "source_sha",
            "workflow_sha",
            "run_id",
            "files",
            "verification",
            "notes",
        }
        <= payload.keys(),
        "Missing manifest fields",
    )
    require(
        type(payload["schema_version"]) is int
        and payload["schema_version"] == 1
        and payload["repository"] == REPOSITORY
        and payload["source_ref"] == "refs/heads/main",
        "Invalid manifest identity",
    )
    require(
        payload["channel"] == version_info(payload["version"])["channel"],
        "Manifest channel mismatch",
    )
    commit(payload["source_sha"])
    commit(payload["workflow_sha"])
    require(
        type(payload["run_id"]) is int and payload["run_id"] > 0, "Invalid manifest run identity"
    )
    require(isinstance(payload["notes"], str), "Invalid release notes")
    reports = payload["verification"]
    require(
        isinstance(reports, dict)
        and set(reports) == set(PLATFORMS)
        and all(matches(r"[0-9a-f]{64}", value) for value in reports.values()),
        "Invalid verification digests",
    )
    files = payload["files"]
    require(isinstance(files, list) and len(files) == 2, "Invalid manifest files")
    for record in files:
        require(
            isinstance(record, dict) and {"filename", "size", "sha256"} <= record.keys(),
            "Invalid file record",
        )
        safe_path(record["filename"])
        require(
            "/" not in record["filename"]
            and type(record["size"]) is int
            and record["size"] >= 0
            and matches(r"[0-9a-f]{64}", record["sha256"]),
            "Invalid file identity",
        )
    expected = distributions(directory, payload["version"])
    actual = [{key: record[key] for key in ("filename", "size", "sha256")} for record in files]
    require(
        sorted(actual, key=lambda item: item["filename"]) == expected,
        "Retained distribution identity mismatch",
    )
    return payload


def gh_api(endpoint, body=None):
    command = [
        "gh",
        "api",
        endpoint,
        "--method",
        "POST" if body is not None else "GET",
        "--header",
        "X-GitHub-Api-Version: 2026-03-10",
    ]
    if body is not None:
        command.extend(["--input", "-"])
    result = subprocess.run(
        command,
        input=json.dumps(body) if body is not None else "",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    require(result.returncode == 0, f"GitHub API failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def validate_run(record, identity):
    require(
        isinstance(record, dict)
        and record.get("id") == identity
        and type(record.get("id")) is int
        and record.get("repository", {}).get("full_name") == REPOSITORY
        and record.get("path") == WORKFLOW
        and record.get("head_branch") == "main"
        and record.get("event") == "workflow_dispatch",
        "Wrong release workflow run identity",
    )
    commit(record.get("head_sha"))
    require(
        record.get("html_url") == f"https://github.com/{REPOSITORY}/actions/runs/{identity}",
        "Wrong run URL",
    )
    require(
        record.get("status")
        in ("queued", "in_progress", "waiting", "pending", "requested", "completed"),
        "Invalid run status",
    )
    require(
        (
            record["status"] == "completed"
            and record.get("conclusion")
            in (
                "success",
                "failure",
                "cancelled",
                "timed_out",
                "action_required",
                "neutral",
                "skipped",
                "stale",
                "startup_failure",
            )
        )
        or (record["status"] != "completed" and record.get("conclusion") is None),
        "Invalid run conclusion",
    )
    return record


def delivery(args):
    if args.operation == "prepare":
        version_info(args.version)
        require(args.ref == "main", "Release source must be main")
        resume = "" if args.resume is None else str(run_id(args.resume))
        response = gh_api(
            f"repos/{REPOSITORY}/actions/workflows/make_release.yml/dispatches",
            {"ref": "main", "inputs": {"version": args.version, "resume_run": resume}},
        )
        require(
            isinstance(response, dict)
            and type(response.get("workflow_run_id")) is int
            and response["workflow_run_id"] > 0,
            "Invalid dispatch response",
        )
        identity = response["workflow_run_id"]
        require(
            response.get("html_url") == f"https://github.com/{REPOSITORY}/actions/runs/{identity}"
            and response.get("run_url")
            == f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{identity}",
            "Dispatch run URL mismatch",
        )
        print(json.dumps(response))
        return 0
    identity = run_id(args.run)
    response = validate_run(gh_api(f"repos/{REPOSITORY}/actions/runs/{identity}"), identity)
    print(json.dumps(response))
    return int(response["status"] == "completed" and response["conclusion"] != "success")


def allowed_url(url, host):
    parsed = urllib.parse.urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.hostname == host
        and parsed.port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment,
        "Untrusted registry URL",
    )


class RegistryRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        allowed_url(newurl, urllib.parse.urlsplit(request.full_url).hostname)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(url, host):
    allowed_url(url, host)
    with urllib.request.build_opener(RegistryRedirect()).open(url, timeout=60) as response:
        allowed_url(response.geturl(), host)
        require(response.getcode() == 200, "Unexpected registry HTTP status")
        return response.read()


def reconcile(payload, directory, output, complete=False, downloaded=None):
    output = Path(output)
    require(
        not output.exists() or output.is_dir() and not any(output.iterdir()),
        "Staging directory must be empty",
    )
    host, file_host = (
        ("pypi.org", "files.pythonhosted.org")
        if payload["channel"] == "pypi"
        else ("test.pypi.org", "test-files.pythonhosted.org")
    )
    try:
        response = json.loads(
            fetch(f"https://{host}/pypi/dphtools/{payload['version']}/json", host)
        )
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        response = {"info": {"name": "dphtools", "version": payload["version"]}, "urls": []}
    require(
        isinstance(response, dict)
        and isinstance(response.get("urls"), list)
        and response.get("info", {}).get("name") == "dphtools"
        and response.get("info", {}).get("version") == payload["version"],
        "Invalid registry version response",
    )
    expected = {record["filename"]: record for record in payload["files"]}
    published = {}
    for record in response["urls"]:
        require(
            isinstance(record, dict)
            and record.get("filename") in expected
            and record["filename"] not in published,
            "Unexpected or duplicate registry file",
        )
        original = expected[record["filename"]]
        require(
            record.get("digests", {}).get("sha256") == original["sha256"]
            and type(record.get("size")) is int
            and record["size"] == original["size"],
            "Published file identity conflict",
        )
        allowed_url(record.get("url", ""), file_host)
        content = fetch(record["url"], file_host)
        require(
            len(content) == original["size"] and digest(content) == original["sha256"],
            "Published bytes differ from retained bundle",
        )
        published[record["filename"]] = content
    missing = expected.keys() - published.keys()
    require(not complete or not missing, "Publication is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    for filename in sorted(missing):
        shutil.copyfile(Path(directory) / filename, output / filename)
    if downloaded is not None:
        destination = Path(downloaded)
        require(
            not destination.exists() or not any(destination.iterdir()),
            "Download directory must be empty",
        )
        destination.mkdir(parents=True, exist_ok=True)
        for filename, content in published.items():
            (destination / filename).write_bytes(content)
    return {
        **{
            key: payload[key]
            for key in (
                "repository",
                "version",
                "channel",
                "source_sha",
                "workflow_sha",
                "run_id",
                "files",
                "verification",
            )
        },
        "published": sorted(published),
        "missing": sorted(missing),
    }


PROBE = """
from importlib import metadata
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import numpy, pandas, scipy, skimage, dphtools
from dphtools import utils
assert metadata.version('dphtools') == sys.argv[1] == dphtools.__version__, 'installed version mismatch'
for module in (dphtools, utils, numpy, pandas, scipy, matplotlib, skimage):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'import escaped clean environment'
assert utils.bin_ndarray(numpy.arange(4).reshape(2, 2), new_shape=(1, 1), operation='sum').item() == 6, 'array sum mismatch'
print('Installed release smoke passed')
"""


def process(command, cwd, env):
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", check=False
    )
    require(
        result.returncode == 0,
        f"Subprocess failed ({result.returncode}): {result.stdout}\n{result.stderr}",
    )


def smoke(payload, directory):
    env = dict(os.environ)
    for key in list(env):
        if any(
            word in key.upper() for word in ("TOKEN", "PASSWORD", "CREDENTIAL", "SECRET")
        ) or key.upper().startswith(("TWINE_", "PYPI_", "TESTPYPI_", "GH_", "GITHUB_", "AWS_")):
            env.pop(key)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
    ):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_INDEX_URL="https://pypi.org/simple",
        PIP_NO_INPUT="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        MPLBACKEND="Agg",
    )
    for record in payload["files"]:
        artifact = (Path(directory) / record["filename"]).resolve()
        with tempfile.TemporaryDirectory(prefix="dphtools-release-smoke-") as temporary:
            root = Path(temporary)
            environment = root / "environment"
            process([sys.executable, "-m", "venv", str(environment)], root, env)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            install_env = dict(env, PATH=str(root / "no-executables"))
            process([str(python), "-m", "pip", "install", str(artifact)], root, install_env)
            process([str(python), "-m", "pip", "check"], root, env)
            process([str(python), "-c", PROBE, payload["version"]], root, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version", allow_abbrev=False)
    version.add_argument("version")
    manifest = commands.add_parser("manifest", allow_abbrev=False)
    for name in (
        "version",
        "source-sha",
        "workflow-sha",
        "run-id",
        "dist",
        "reports",
        "notes",
        "output",
    ):
        manifest.add_argument("--" + name, required=True)
    for name in ("check", "smoke", "reconcile"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--manifest", required=True)
        command.add_argument("--dist", required=True)
        if name == "reconcile":
            command.add_argument("--output", required=True)
            command.add_argument("--require-complete", action="store_true")
            command.add_argument(
                "--downloaded", help="retain verified registry bytes for post-upload smoke"
            )
    args = parser.parse_args()
    try:
        if args.command == "version":
            print(json.dumps(version_info(args.version)))
        elif args.command == "manifest":
            payload = {
                "schema_version": 1,
                "repository": REPOSITORY,
                **version_info(args.version),
                "source_ref": "refs/heads/main",
                "source_sha": commit(args.source_sha),
                "workflow_sha": commit(args.workflow_sha),
                "run_id": run_id(args.run_id),
                "files": distributions(args.dist, args.version),
                "verification": verification(args.reports),
                "notes": Path(args.notes).read_text(encoding="utf-8"),
            }
            Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            payload = check(args.manifest, args.dist)
            if args.command == "smoke":
                smoke(payload, args.dist)
            elif args.command == "reconcile":
                print(
                    json.dumps(
                        reconcile(
                            payload, args.dist, args.output, args.require_complete, args.downloaded
                        )
                    )
                )
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        print(f"Release validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
