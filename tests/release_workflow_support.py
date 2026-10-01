"""External services only. Owned callers are opaque copies and execute unchanged."""

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import zipfile

from .test_release_cli import run_record
from .test_release_smoke import clean_env
from .test_release_support import (
    Bundle,
    PLATFORMS,
    REPOSITORY,
    RUN_ID,
    SOURCE_SHA,
    WORKFLOW_SHA,
    digest,
    install_executable,
    write_json,
)
from .test_release_version import ROOT, diagnostic, invoke

PREFIX = f"repos/{REPOSITORY}/"
ARTIFACT_ID = 98761

SERVICE = r"""
import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.parse

config_path = Path(os.environ["FIXTURE_CONFIG"])
log_path = Path(os.environ["FIXTURE_LOG"])
def service(method, endpoint, headers, body):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    endpoint = endpoint.removeprefix("https://api.github.com/")
    if endpoint.startswith("https://uploads.github.com/"):
        endpoint = endpoint.removeprefix("https://uploads.github.com/")
    record = {"method": method, "endpoint": endpoint, "headers": headers,
              "body": base64.b64encode(body).decode()}
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    key = method + " " + endpoint
    if key not in config["routes"]:
        return 599, {}, b"Unconfigured external endpoint"
    item = config["routes"][key]
    status = item.get("status", 200)
    if status >= 400:
        return status, item.get("headers", {}), b"Controlled HTTP failure"
    if method == "POST" and endpoint.endswith("/releases"):
        release = {**item["json"], **json.loads(body)}
        release_key = "GET " + endpoint + "/tags/" + release["tag_name"]
        config["routes"][release_key] = {"json": release}
        config_path.write_text(json.dumps(config), encoding="utf-8")
    if method == "PATCH" and endpoint.endswith("/releases/491"):
        release_key = "GET " + endpoint.rsplit("/", 1)[0] + "/tags/1.0.0"
        config["routes"][release_key]["json"].update(json.loads(body))
        config_path.write_text(json.dumps(config), encoding="utf-8")
    if method == "POST" and "/assets?name=" in endpoint:
        name = urllib.parse.parse_qs(urllib.parse.urlsplit(endpoint).query)["name"][0]
        asset_key = "GET " + endpoint.split("?")[0] + "?per_page=100"
        config["routes"][asset_key]["json"].append(
            {"name": name, "digest": "sha256:" + hashlib.sha256(body).hexdigest()})
        config_path.write_text(json.dumps(config), encoding="utf-8")
    if "raw" in item:
        data = base64.b64decode(item["raw"])
    else:
        data = json.dumps(item.get("json", {})).encode()
    return status, item.get("headers", {}), data
"""

GH = r"""
import json
import os
from pathlib import Path
import sys
sys.excepthook = lambda kind, value, traceback: print(f"{kind.__name__}: {value}", file=sys.stderr)
exec(Path(os.environ["FIXTURE_SERVICE"]).read_text(encoding="utf-8"))
args = iter(sys.argv[1:])
assert next(args) == "api"
endpoint = None
method, headers, body = "GET", {}, b""
for arg in args:
    if arg in ("--method", "-X"):
        method = next(args)
    elif arg in ("--header", "-H"):
        name, value = next(args).split(":", 1)
        headers[name.lower()] = value.strip()
    elif arg == "--input":
        filename = next(args)
        body = sys.stdin.buffer.read() if filename == "-" else Path(filename).read_bytes()
    elif not arg.startswith("-") and endpoint is None:
        endpoint = arg
    else:
        raise AssertionError("Undeclared gh option: " + arg)
status, headers, data = service(method, endpoint, headers, body)
if status >= 400:
    print(f"gh: Controlled HTTP failure (HTTP {status})", file=sys.stderr)
    raise SystemExit(1)
sys.stdout.buffer.write(data)
"""

GIT = r"""
import json
import os
from pathlib import Path
import sys
sys.excepthook = lambda kind, value, traceback: print(f"{kind.__name__}: {value}", file=sys.stderr)
config = json.loads(Path(os.environ["FIXTURE_CONFIG"]).read_text(encoding="utf-8"))["git"]
args = sys.argv[1:]
with Path(os.environ["FIXTURE_GIT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")
if args == ["rev-parse", "HEAD"]:
    print(config["head"])
elif args == ["fetch", "--no-tags", "origin", "main"]:
    raise SystemExit(config.get("fetch_status", 0))
elif len(args) == 4 and args[:2] == ["merge-base", "--is-ancestor"] and args[3] == "FETCH_HEAD":
    raise SystemExit(0 if args[2] in config["ancestors"] else 1)
elif len(args) == 2 and args[0] == "rev-parse" and ":" in args[1]:
    sha, path = args[1].split(":", 1)
    assert path in (".github/workflows/make_release.yml", "tools/release.py", "tools/release_workflow.py", "tools/release_probe.py")
    print(config.get("objects", {}).get(args[1], "7" * 40))
else:
    raise AssertionError("Undeclared Git operation: " + repr(args))
"""

BOOTSTRAP = r"""
import base64
from email.message import Message
from io import BytesIO
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import urllib.request
import urllib.response
import warnings
entry, config_file, *args = sys.argv[1:]
settings = json.loads(Path(config_file).read_text(encoding="utf-8"))
os.environ.update(settings["environment"])
warnings.formatwarning = lambda message, category, filename, lineno, line=None: f"{category.__name__}: {message}\n"
sys.excepthook = lambda kind, value, traceback: print(f"{kind.__name__}: {value}", file=sys.stderr)
exec(Path(os.environ["FIXTURE_SERVICE"]).read_text(encoding="utf-8"))
class ControlledHTTPS(urllib.request.HTTPSHandler):
    def https_open(self, request):
        status, headers, body = service(request.get_method(), request.full_url,
                                       {k.lower(): v for k,v in request.header_items()}, request.data or b"")
        message = Message()
        for key, value in headers.items():
            message[key] = value
        response = urllib.response.addinfourl(BytesIO(body), message, request.full_url, status)
        response.msg = "Controlled external response"
        return response
class ControlledHTTP(urllib.request.HTTPHandler):
    http_open = ControlledHTTPS.https_open
urllib.request.HTTPSHandler = ControlledHTTPS
urllib.request.HTTPHandler = ControlledHTTP
urllib.request.install_opener(urllib.request.build_opener(ControlledHTTPS(), ControlledHTTP()))
def blocked(*args, **kwargs):
    raise AssertionError("Fixture forbids live network")
socket.create_connection = blocked
socket.socket.connect = blocked
sys.argv = [entry, *args]
sys.path[0] = str(Path(entry).resolve().parent)
runpy.run_path(entry, run_name="__main__")
"""


class Workflow:
    """Configure external metadata/bytes, never owned command outcomes."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.repo = tmp_path / "opaque caller with spaces"
        shutil.copytree(
            ROOT / "tools", self.repo / "tools", ignore=shutil.ignore_patterns("__pycache__")
        )
        self.entry = self.repo / "tools/release_workflow.py"
        self.outside = tmp_path / "external working directory"
        self.outside.mkdir()
        self.bundle = Bundle(tmp_path / "original bundle")
        self.bundle.reports.rename(self.bundle.root / "reports")
        self.bundle.reports = self.bundle.root / "reports"
        self.bundle.dist.rename(self.bundle.root / "dist")
        self.bundle.dist = self.bundle.root / "dist"
        self.bundle.wheel = self.bundle.dist / self.bundle.wheel.name
        self.bundle.sdist = self.bundle.dist / self.bundle.sdist.name
        self.payload = self.bundle.expected()
        self.payload["workflow_sha"] = SOURCE_SHA
        write_json(self.bundle.manifest, self.payload)
        self.bin = tmp_path / "external executables"
        install_executable(self.bin, "gh", GH)
        install_executable(self.bin, "git", GIT)
        self.log = tmp_path / "transport.jsonl"
        self.git_log = tmp_path / "git.jsonl"
        self.config_file = tmp_path / "external configuration.json"
        self.bootstrap_config = tmp_path / "bootstrap configuration.json"
        self.service_file = tmp_path / "external service.py"
        self.service_file.write_text(SERVICE, encoding="utf-8")
        self.driver = tmp_path / "external bootstrap.py"
        self.driver.write_text(BOOTSTRAP, encoding="utf-8")
        self.output = tmp_path / "actions output.txt"
        self.summary = tmp_path / "approval summary.md"
        self.env = {
            "PATH": str(self.bin),
            "GH_TOKEN": "fixture-not-a-credential",
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_RUN_ID": str(RUN_ID),
            "TRUSTED_WORKFLOW_SHA": SOURCE_SHA,
            "GITHUB_OUTPUT": str(self.output),
            "GITHUB_STEP_SUMMARY": str(self.summary),
            "FIXTURE_CONFIG": str(self.config_file),
            "FIXTURE_LOG": str(self.log),
            "FIXTURE_GIT_LOG": str(self.git_log),
            "FIXTURE_SERVICE": str(self.service_file),
        }
        if os.name == "nt":
            self.env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
        self.config = {
            "routes": {},
            "git": {"head": SOURCE_SHA, "ancestors": [SOURCE_SHA, WORKFLOW_SHA]},
        }
        self.route("actions/runs/" + str(RUN_ID), run_record())
        self.archive = self.archive_bytes()
        self.artifact = {
            "id": ARTIFACT_ID,
            "name": "release-bundle",
            "expired": False,
            "digest": "sha256:" + digest(self.archive),
            "workflow_run": {"id": RUN_ID, "head_sha": SOURCE_SHA},
        }
        self.reset_artifact()
        self.config["routes"]["GET https://pypi.org/pypi/dphtools/1.0.0/json"] = {"status": 404}
        self.route("git/ref/tags/1.0.0", status=404)
        self.route("releases/tags/1.0.0", status=404)
        self.route(
            "git/refs",
            {"ref": "refs/tags/1.0.0", "object": {"type": "commit", "sha": SOURCE_SHA}},
            method="POST",
        )
        self.release = {"id": 491, "tag_name": "1.0.0", "prerelease": False, "draft": True}
        self.route("releases", self.release, method="POST")
        self.route("releases/491/assets?per_page=100", [])
        self.route("releases/491", {}, method="PATCH")
        for path in (
            self.bundle.wheel,
            self.bundle.sdist,
            self.bundle.manifest,
            tmp_path / "receipt.json",
        ):
            self.route("releases/491/assets?name=" + path.name, {}, method="POST")
        self.receipt = tmp_path / "receipt.json"
        write_json(
            self.receipt, {**self.payload, "published": self.payload["files"], "missing": []}
        )

    def route(self, endpoint, payload=None, *, method="GET", status=200, raw=None, headers=None):
        self.config["routes"][method + " " + PREFIX + endpoint] = {
            "status": status,
            "json": payload,
            **({"raw": base64.b64encode(raw).decode()} if raw is not None else {}),
            **({"headers": headers} if headers else {}),
        }

    def archive_bytes(self):
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.bundle.manifest, "manifest.json")
            for path in (self.bundle.wheel, self.bundle.sdist):
                archive.write(path, "dist/" + path.name)
            for platform in PLATFORMS:
                archive.write(
                    self.bundle.reports / platform / "checks.json",
                    f"reports/{platform}/checks.json",
                )
            archive.writestr("notes.md", self.payload["notes"])
        return stream.getvalue()

    def reset_artifact(self):
        self.route(f"actions/runs/{RUN_ID}/artifacts?per_page=100", {"artifacts": [self.artifact]})
        self.route(f"actions/artifacts/{ARTIFACT_ID}", self.artifact)
        self.route(f"actions/artifacts/{ARTIFACT_ID}/zip", raw=self.archive)

    def binding(self):
        return [
            "--manifest",
            self.bundle.manifest,
            "--dist",
            self.bundle.dist,
            "--version",
            "1.0.0",
            "--source-sha",
            SOURCE_SHA,
            "--workflow-sha",
            SOURCE_SHA,
            "--origin-run",
            str(RUN_ID),
        ]

    def artifact_args(self):
        return [
            "--origin-run",
            str(RUN_ID),
            "--artifact-id",
            str(ARTIFACT_ID),
            "--artifact-digest",
            self.artifact["digest"],
            "--workflow-sha",
            SOURCE_SHA,
        ]

    def tag_args(self):
        return [*self.binding(), "--manifest-digest", digest(self.bundle.manifest.read_bytes())]

    def run(self, command, *args):
        write_json(self.config_file, self.config)
        write_json(self.bootstrap_config, {"environment": self.env})
        result = invoke(
            self.driver,
            self.entry,
            self.bootstrap_config,
            command,
            *args,
            cwd=self.outside,
            env=clean_env(),
        )
        self.config = json.loads(self.config_file.read_text(encoding="utf-8"))
        return result

    def calls(self):
        return (
            []
            if not self.log.exists()
            else [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        )

    def clear(self):
        for path in (self.log, self.git_log, self.output, self.summary):
            path.unlink(missing_ok=True)

    def ready(self, command, *args):
        result = self.run(command, *args)
        code = result.returncode
        message = "Valid control: " + diagnostic(result)
        assert code == 0, message
        self.clear()

    def outputs(self):
        return dict(
            line.split("=", 1) for line in self.output.read_text(encoding="utf-8").splitlines()
        )
