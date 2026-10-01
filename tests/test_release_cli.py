"""S1/S2/S4/S5: actual delivery CLI with an external gh executable boundary."""

import json
import os
import re
import shutil
import subprocess

import pytest

from .test_release_support import (
    REPOSITORY,
    RUN_ID,
    SOURCE_SHA,
    install_executable,
    rejected,
    snapshot,
    write_json,
)
from .test_release_version import ROOT, copy_project_context, diagnostic, invoke

RUN_URL = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{RUN_ID}"
DISPATCH = f"repos/{REPOSITORY}/actions/workflows/make_release.yml/dispatches"
STATUS = f"repos/{REPOSITORY}/actions/runs/{RUN_ID}"

FAKE_GH = r"""
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
stdin = sys.stdin.read()
record = {"argv": args, "stdin": stdin}
for index, arg in enumerate(args):
    key, equal, value = arg.partition("=")
    if key not in ("--input", "--field", "-F"):
        continue
    if not equal:
        value = args[index + 1]
    if key == "--input":
        record["request_body"] = stdin if value == "-" else Path(value).read_text(encoding="utf-8")
    elif "=" in value and value.split("=", 1)[1].startswith("@"):
        filename = value.split("=", 1)[1][1:]
        record.setdefault("field_files", {})[filename] = stdin if filename == "-" else Path(filename).read_text(encoding="utf-8")
with Path(os.environ["RELEASE_TEST_GH_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
config = json.loads(Path(os.environ["RELEASE_TEST_GH_CONFIG"]).read_text(encoding="utf-8"))
if args == ["fixture-probe", "literal & $VALUE %VALUE% 'quoted'"]:
    print("external fixture ready")
    raise SystemExit(0)
if not args or args[0] != "api":
    print("Only the approved gh api boundary is available", file=sys.stderr)
    raise SystemExit(97)
response = config["dispatch"] if any("/dispatches" in arg for arg in args) else config["status"]
if config.get("exit", 0):
    print("Controlled GitHub transport failure", file=sys.stderr)
    raise SystemExit(config["exit"])
if isinstance(response, str):
    print(response)
else:
    print(json.dumps(response))
"""


def run_record(**changes):
    record = {
        "id": RUN_ID,
        "repository": {"full_name": REPOSITORY},
        "path": ".github/workflows/make_release.yml",
        "head_branch": "main",
        "event": "workflow_dispatch",
        "head_sha": SOURCE_SHA,
        "status": "completed",
        "conclusion": "success",
        "html_url": RUN_URL,
    }
    record.update(changes)
    return record


class Delivery:
    def __init__(self, tmp_path):
        self.repo = tmp_path / "opaque repo & $VALUE %VALUE% 'quoted'"
        shutil.copytree(
            ROOT / "tools", self.repo / "tools", ignore=shutil.ignore_patterns("__pycache__")
        )
        copy_project_context(self.repo)
        self.entry = self.repo / "tools/delivery"
        self.outside = tmp_path / "external caller"
        self.outside.mkdir()
        self.bin = tmp_path / "external tools"
        self.executable = install_executable(self.bin, "gh", FAKE_GH)
        self.log = tmp_path / "gh.jsonl"
        self.config_file = tmp_path / "gh-config.json"
        self.config = {
            "dispatch": {"workflow_run_id": RUN_ID, "run_url": API_URL, "html_url": RUN_URL},
            "status": run_record(),
        }
        self.env = dict(
            os.environ,
            PATH=str(self.bin),
            RELEASE_TEST_GH_LOG=str(self.log),
            RELEASE_TEST_GH_CONFIG=str(self.config_file),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
        )
        if os.name == "nt":
            self.env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
        self.save()

    def save(self):
        write_json(self.config_file, self.config)

    def run(self, *args):
        self.save()
        return invoke(self.entry, "release", *args, cwd=self.outside, env=self.env)

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def clear(self):
        self.log.unlink(missing_ok=True)

    def ready(self, command):
        args = (
            ("prepare", "--version", "1.0.0")
            if command == "prepare"
            else ("status", "--run", str(RUN_ID))
        )
        result = self.run(*args)
        assert result.returncode == 0, "Approved command precondition: " + diagnostic(result)
        assert RUN_URL in result.stdout, diagnostic(result)
        self.clear()


@pytest.fixture
def delivery(tmp_path):
    command = Delivery(tmp_path)
    probe = subprocess.run(
        [str(command.executable), "fixture-probe", "literal & $VALUE %VALUE% 'quoted'"],
        cwd=command.outside,
        env=command.env,
        input="probe input",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert probe.returncode == 0, diagnostic(probe)
    assert probe.stdout.strip() == "external fixture ready"
    assert command.calls() == [
        {"argv": ["fixture-probe", "literal & $VALUE %VALUE% 'quoted'"], "stdin": "probe input"}
    ]
    command.clear()
    return command


def api_request(call):
    """Decode standard gh api arguments; do not constrain field transport syntax."""
    args = iter(call["argv"])
    assert next(args) == "api", "No approval, upload, release, tag, or workflow-list commands"
    method, endpoint, headers, fields, payload = None, None, {}, {}, None
    for arg in args:
        key, equal, value = arg.partition("=")
        if key in (
            "--method",
            "-X",
            "--header",
            "-H",
            "--field",
            "-F",
            "--raw-field",
            "-f",
            "--input",
            "--hostname",
        ):
            if not equal:
                value = next(args)
            if key in ("--method", "-X"):
                method = value
            elif key in ("--header", "-H"):
                name, value = value.split(":", 1)
                headers[name.lower()] = value.strip()
            elif key == "--input":
                payload = json.loads(call["request_body"])
            elif key == "--hostname":
                assert (
                    value == "github.com"
                ), "Dispatch and status must target the approved service"
            else:
                name, value = value.split("=", 1)
                if key in ("--field", "-F"):
                    if value.startswith("@"):
                        value = call["field_files"][value[1:]]
                    elif value in ("true", "false", "null"):
                        value = json.loads(value)
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                if name.startswith("inputs["):
                    fields.setdefault("inputs", {})[name[7:-1]] = value
                else:
                    fields[name] = value
        elif arg.startswith("-"):
            pytest.fail(f"Unaccounted gh api option in boundary fixture: {arg}")
        else:
            assert endpoint is None, "More than one API endpoint"
            endpoint = arg.removeprefix("https://api.github.com/").lstrip("/")
    body = fields if payload is None else payload
    return method or ("POST" if body else "GET"), endpoint, headers, body


def requests(delivery):
    return [api_request(call) for call in delivery.calls()]


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0a1", "1.0.0b2", "1.0.0rc1"])
@pytest.mark.parametrize("explicit_main", [False, True])
def test_prepare_dispatches_main_and_returns_exact_identity(delivery, version, explicit_main):
    before = snapshot(delivery.repo)
    result = delivery.run(
        "prepare", "--version", version, *(["--ref", "main"] if explicit_main else [])
    )
    assert result.returncode == 0, diagnostic(result)
    assert RUN_URL in result.stdout, diagnostic(result)
    calls = requests(delivery)
    posts = [call for call in calls if call[0] == "POST"]
    assert len(posts) == 1
    method, endpoint, headers, body = posts[0]
    assert endpoint == DISPATCH
    assert headers["x-github-api-version"] == "2026-03-10"
    assert body == {"ref": "main", "inputs": {"version": version, "resume_run": ""}}
    assert all(call[1] in (DISPATCH, STATUS) for call in calls), "Never infer the newest run"
    assert all(call[0] == "GET" for call in calls if call[1] == STATUS)
    assert snapshot(delivery.repo) == before, "Preparation must not write local release state"


@pytest.mark.parametrize("resume", ["1", "70931"])
def test_resume_dispatches_original_run_identity(delivery, resume):
    result = delivery.run("prepare", "--version", "1.0.0", "--resume", resume)
    assert result.returncode == 0, diagnostic(result)
    assert RUN_URL in result.stdout
    posts = [call for call in requests(delivery) if call[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1] == DISPATCH
    assert posts[0][3] == {"ref": "main", "inputs": {"version": "1.0.0", "resume_run": resume}}
    assert all(call[1] in (DISPATCH, STATUS) for call in requests(delivery))


@pytest.mark.parametrize(
    "option,value",
    [
        ("--ref", "feature"),
        ("--ref", "refs/heads/main"),
        ("--ref", "main; echo unsafe"),
        ("--version", "v1.0.0"),
        ("--version", "1.0.0.dev1"),
        ("--version", "1.0.0+local"),
        ("--version", "1.0.0.post1"),
        ("--version", "01.0.0"),
        ("--resume", "0"),
        ("--resume", "-1"),
        ("--resume", "1.5"),
        ("--resume", "+1"),
        ("--resume", "1;echo unsafe"),
    ],
)
def test_prepare_invalid_inputs_never_dispatch(delivery, option, value):
    delivery.ready("prepare")
    args = ["prepare"] + ([] if option == "--version" else ["--version", "1.0.0"])
    before = snapshot(delivery.repo)
    rejected(delivery.run(*args, option, value))
    assert delivery.calls() == []
    assert snapshot(delivery.repo) == before


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        {},
        [],
        {"workflow_run_id": 0, "run_url": API_URL, "html_url": RUN_URL},
        {"workflow_run_id": True, "run_url": API_URL, "html_url": RUN_URL},
        {"workflow_run_id": RUN_ID, "run_url": API_URL, "html_url": RUN_URL + "9"},
        {
            "workflow_run_id": RUN_ID,
            "run_url": API_URL.replace(REPOSITORY, "other/repo"),
            "html_url": RUN_URL,
        },
        {
            "workflow_run_id": RUN_ID,
            "run_url": API_URL,
            "html_url": RUN_URL.replace("github.com", "example.invalid"),
        },
    ],
)
def test_dispatch_malformed_or_wrong_identity_is_failure(delivery, response):
    delivery.ready("prepare")
    delivery.config["dispatch"] = response
    rejected(delivery.run("prepare", "--version", "1.0.0"))
    calls = requests(delivery)
    assert sum(call[0] == "POST" and call[1] == DISPATCH for call in calls) == 1
    assert all(call[1] in (DISPATCH, STATUS) for call in calls)


@pytest.mark.parametrize("command", ["prepare", "status"])
@pytest.mark.parametrize("fault", ["missing-executable", "api-failure"])
def test_external_failure_is_not_success(delivery, command, fault):
    delivery.ready(command)
    if fault == "missing-executable":
        delivery.executable.unlink()
    else:
        delivery.config["exit"] = 19
    args = (
        ("prepare", "--version", "1.0.0")
        if command == "prepare"
        else ("status", "--run", str(RUN_ID))
    )
    result = delivery.run(*args)
    rejected(result)
    if fault == "missing-executable":
        assert delivery.calls() == []
    else:
        assert len(delivery.calls()) == 1


@pytest.mark.parametrize(
    "status,conclusion",
    [
        ("queued", None),
        ("in_progress", None),
        ("waiting", None),
        ("completed", "success"),
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("completed", "timed_out"),
    ],
)
def test_status_truthfully_observes_only_exact_run(delivery, status, conclusion):
    delivery.config["status"] = run_record(status=status, conclusion=conclusion)
    before = snapshot(delivery.repo)
    result = delivery.run("status", "--run", str(RUN_ID))
    output = result.stdout + result.stderr
    assert RUN_URL in output, diagnostic(result)
    assert status in output, diagnostic(result)
    if conclusion is not None:
        assert conclusion in output, diagnostic(result)
    if status == "completed":
        assert (result.returncode == 0) is (conclusion == "success"), diagnostic(result)
    else:
        # A successful read of a pending run may be described as "success";
        # it must not claim completed release success. Output wording is free.
        assert not re.search(
            r"completed\s+successfully|conclusion\s*[:=]\s*[\"']?success", output, re.I
        ), diagnostic(result)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict):
                assert payload.get("conclusion") != "success"
    calls = requests(delivery)
    assert len(calls) == 1
    assert calls[0][0:2] == ("GET", STATUS)
    assert calls[0][2]["x-github-api-version"] == "2026-03-10"
    assert calls[0][3] == {}
    assert snapshot(delivery.repo) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"id": RUN_ID + 1},
        {"repository": {"full_name": "other/repo"}},
        {"path": ".github/workflows/other.yml"},
        {"head_branch": "feature"},
        {"event": "pull_request"},
        {"head_sha": "short"},
        {"head_sha": SOURCE_SHA.upper()},
        {"html_url": RUN_URL.replace(REPOSITORY, "other/repo")},
        {"status": "completed", "conclusion": None},
        {"status": "unknown-state"},
    ],
)
def test_status_rejects_unrelated_or_malformed_run(delivery, changes):
    delivery.ready("status")
    delivery.config["status"] = run_record(**changes)
    rejected(delivery.run("status", "--run", str(RUN_ID)))
    calls = requests(delivery)
    assert len(calls) == 1 and calls[0][0:2] == ("GET", STATUS)


@pytest.mark.parametrize("response", ["not JSON", {}, [], {"id": RUN_ID}])
def test_status_rejects_incomplete_response(delivery, response):
    delivery.ready("status")
    delivery.config["status"] = response
    rejected(delivery.run("status", "--run", str(RUN_ID)))
    assert [call[0:2] for call in requests(delivery)] == [("GET", STATUS)]


@pytest.mark.parametrize("run_id", ["0", "-2", "2.5", "+3", "", "1 && unsafe"])
def test_status_invalid_identity_makes_no_external_call(delivery, run_id):
    delivery.ready("status")
    rejected(delivery.run("status", "--run", run_id))
    assert delivery.calls() == []


def test_gh_fixture_captures_json_before_cleanup_and_preserves_field_types(delivery, tmp_path):
    """Fixture self-check only: gh input files and documented typed/raw fields."""
    request_file = tmp_path / "request body with spaces.json"
    body = {"ref": "main", "inputs": {"version": "1.0.0", "resume_run": "70931"}}
    write_json(request_file, body)
    result = subprocess.run(
        [str(delivery.executable), "api", DISPATCH, "--input", str(request_file)],
        cwd=delivery.outside,
        env=delivery.env,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, diagnostic(result)
    request_file.unlink()
    assert api_request(delivery.calls()[0])[3] == body
    delivery.clear()
    result = subprocess.run(
        [
            str(delivery.executable),
            "api",
            DISPATCH,
            "-f",
            "ref=main",
            "-F",
            "inputs[resume_run]=70931",
        ],
        cwd=delivery.outside,
        env=delivery.env,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, diagnostic(result)
    assert api_request(delivery.calls()[0])[3] == {"ref": "main", "inputs": {"resume_run": 70931}}
