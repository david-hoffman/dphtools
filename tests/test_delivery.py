"""Black-box tests for the approved SETUP-001 doctor command.

R1/R2: one fresh exec, repository rooting, canonical procedure and skill transport.
R3: check intent, sandbox and disabled features, including repeated invocations.
R4: output/status forwarding and preflight failures without retries.
R5: help, invalid invocations and preservation of project files.

The fake harness proves process transport only. It cannot prove that a real agent
inspects evidence, creates the requested documentation patch/branch, or stops for
owner approval. Those behaviors require the separate real doctor demonstration.
"""

from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import zipfile

import pytest

ENTRY_POINT = Path(__file__).resolve().parents[1] / "tools" / "delivery"
PROMPT_PATH = Path("docs/agentic-software-delivery-v1.0/DOCTOR-PROMPT.md")
PROMPT_TEXT = """# Canonical doctor procedure fixture: SETUP001-procedure-7182

Use review-work in doctor mode. Inspect actual repository evidence. Make an
evidenced documentation patch on a docs branch. Wait for owner approval before
commit, push, or merge. Do not change product code. In --check mode inspect only.
"""
HARNESS_STDOUT = "fake codex stdout: evidence inspected\n"
HARNESS_STDERR = "fake codex stderr: diagnostic retained\n"
FAKE_CODEX = r"""
import json
import os
from pathlib import Path
import sys

record = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "stdin": sys.stdin.read(),
}
with Path(os.environ["DELIVERY_TEST_RECORD"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
sys.stdout.write(os.environ["DELIVERY_TEST_STDOUT"])
sys.stdout.flush()
sys.stderr.write(os.environ["DELIVERY_TEST_STDERR"])
sys.stderr.flush()
sys.exit(int(os.environ.get("DELIVERY_TEST_EXIT", "0")))
"""


def _install_fake_codex(bin_dir):
    """Create an external test executable, without a cmd/bat argument-reparse layer."""
    bin_dir.mkdir()
    if os.name == "nt":
        # Reuse pytest's installed native console launcher, replacing only its ZIP
        # payload. This needs no compiler, extra package, shell, or checked-in EXE.
        wrapper = Path(sysconfig.get_path("scripts")) / "pytest.exe"
        assert wrapper.is_file(), "Windows fixture needs pytest's installed console launcher"
        with zipfile.ZipFile(wrapper) as archive:
            payload_offset = min(info.header_offset for info in archive.infolist())
        # Build the ZIP separately so its offsets are relative to the archive.
        # Appending with ZipFile(executable, "a") includes the EXE prefix in those
        # offsets, preventing distlib's launcher from finding its interpreter line.
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("__main__.py", FAKE_CODEX)
        executable = bin_dir / "codex.exe"
        with wrapper.open("rb") as source, executable.open("wb") as target:
            target.write(source.read(payload_offset))
            target.write(payload.getvalue())
    else:
        # env plus a private interpreter symlink also handles spaces in the real
        # interpreter's path, which a direct Python shebang would not handle.
        (bin_dir / "python3").symlink_to(sys.executable)
        executable = bin_dir / "codex"
        executable.write_text("#!/usr/bin/env python3\n" + FAKE_CODEX, encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


class DoctorCommand:
    def __init__(self, tmp_path):
        self.repo = tmp_path / "repo space & $DELIVERY_TEST_TOKEN %DELIVERY_TEST_TOKEN% 'quoted'"
        self.entry_point = self.repo / "tools" / "delivery"
        self.entry_point.parent.mkdir(parents=True)
        # Opaque artifact copy: do not import or inspect the implementation.
        shutil.copy2(ENTRY_POINT, self.entry_point)
        self.prompt = self.repo / PROMPT_PATH
        self.prompt.parent.mkdir(parents=True)
        self.prompt.write_text(PROMPT_TEXT, encoding="utf-8")
        (self.repo / "owner-notes.txt").write_text("Preserve owner data.\n", encoding="utf-8")
        self.outside = tmp_path / "unrelated working directory"
        self.outside.mkdir()
        self.bin_dir = tmp_path / "fake harness bin"
        self.fake_codex = _install_fake_codex(self.bin_dir)
        self.record = tmp_path / "harness-calls.jsonl"
        self.env = os.environ.copy()
        self.env.update(
            PATH=str(self.bin_dir),
            PYTHONIOENCODING="utf-8",
            DELIVERY_TEST_RECORD=str(self.record),
            DELIVERY_TEST_STDOUT=HARNESS_STDOUT,
            DELIVERY_TEST_STDERR=HARNESS_STDERR,
            DELIVERY_TEST_EXIT="0",
            DELIVERY_TEST_TOKEN="ARGUMENT_WAS_EXPANDED_BY_A_SHELL",
        )
        if os.name == "nt":
            self.env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"

    def run(self, *args, cwd=None, env=None):
        return subprocess.run(
            [sys.executable, str(self.entry_point), *args],
            cwd=self.outside if cwd is None else cwd,
            env=self.env if env is None else env,
            input="",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def doctor(tmp_path):
    command = DoctorCommand(tmp_path)
    # Distinguish a broken fake executable from a failure of the real launcher.
    probe_args = ["fixture probe", "literal & $DELIVERY_TEST_TOKEN %DELIVERY_TEST_TOKEN%"]
    probe = subprocess.run(
        [str(command.fake_codex), *probe_args],
        cwd=command.outside,
        env=command.env,
        input="fixture stdin\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert probe.returncode == 0, f"Fake harness setup failed: {_result_detail(probe)}"
    assert probe.stdout == HARNESS_STDOUT
    assert probe.stderr == HARNESS_STDERR
    assert command.calls() == [
        {"argv": probe_args, "cwd": str(command.outside), "stdin": "fixture stdin\n"}
    ]
    command.record.unlink()
    return command


@pytest.fixture
def unreadable_prompt(doctor):
    """Keep a regular file present while the OS denies reads; never skip a failed setup."""
    with ExitStack() as cleanup:
        if os.name == "posix":
            original_mode = stat.S_IMODE(doctor.prompt.stat().st_mode)
            cleanup.callback(doctor.prompt.chmod, original_mode)
            doctor.prompt.chmod(0)
        elif os.name == "nt":
            import ctypes
            from ctypes import wintypes

            # An exclusive handle denies other opens, including the child launcher's
            # read, without editing ACLs or depending on administrator privileges.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create_file.restype = wintypes.HANDLE
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            # GENERIC_READ, no sharing, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL.
            handle = create_file(str(doctor.prompt), 0x80000000, 0, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                pytest.fail(f"Unreadable-file fixture could not lock prompt: {ctypes.WinError()}")
            cleanup.callback(close_handle, handle)
        else:
            pytest.fail(f"Portability blocker: no unreadable-file fixture for {os.name}")

        assert doctor.prompt.is_file(), "Unreadable prompt must remain a regular file"
        try:
            doctor.prompt.read_bytes()
        except PermissionError:
            yield doctor
        else:
            pytest.fail(
                "Portability blocker: this process can still read the protected regular prompt "
                "file (for example, POSIX root bypasses mode bits); required validation cannot run"
            )


def _result_detail(result):
    return f"exit={result.returncode}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"


def _one_call(doctor, result):
    calls = doctor.calls()
    assert (
        len(calls) == 1
    ), f"Expected one Codex launch; got {len(calls)}. {_result_detail(result)}"
    return calls[0]


def _options_and_request(call):
    """Accept only the small Codex option set needed by this launcher contract."""
    aliases = {"-c": "--config", "-s": "--sandbox", "-C": "--cd"}
    value_options = {"--config", "--sandbox", "--cd", "--disable"}
    options = []
    positional = []
    args = iter(call["argv"])
    for arg in args:
        if arg == "--":
            positional.extend(args)
            break
        key, equals, value = arg.partition("=")
        key = aliases.get(key, key)
        if key in value_options:
            if not equals:
                value = next(args, None)
                assert value is not None and not value.startswith(
                    "-"
                ), f"Missing value for Codex option {key}: {value!r}"
            options.append((key, value))
        elif arg == "--skip-git-repo-check":
            options.append((arg, None))
        elif arg.startswith("-") and arg != "-":
            pytest.fail(f"Unsupported Codex option at the process boundary: {arg!r}")
        else:
            positional.append(arg)
    assert positional and positional[0] == "exec", f"Expected codex exec: {call['argv']!r}"
    assert len(positional) <= 2, f"Expected one fresh exec request: {positional!r}"
    assert positional[1:] not in (["resume"], ["fork"]), "Doctor must start a fresh session"
    prompt = positional[1] if len(positional) == 2 and positional[1] != "-" else ""
    request = prompt + "\n" + call["stdin"]
    assert request.strip(), "Doctor must supply an agent request through argv or stdin"
    return options, request


def _assert_session(call, repo, *, check):
    options, request = _options_and_request(call)
    values = dict(options)
    root = Path(values.get("--cd", call["cwd"]))
    if not root.is_absolute():
        root = Path(call["cwd"]) / root
    assert root.resolve() == repo.resolve(), f"Codex session is rooted at {root}, not {repo}"

    config = {}
    disabled = set()
    for key, value in options:
        if key == "--config":
            name, separator, setting = value.partition("=")
            assert separator, f"Invalid config override: {value!r}"
            config[name.strip()] = setting.strip().strip("\"'")
        elif key == "--disable":
            disabled.add(value)
    sandbox = values.get("--sandbox", config.get("sandbox_mode"))
    expected_sandbox = "read-only" if check else "workspace-write"
    assert sandbox == expected_sandbox, f"Expected {expected_sandbox} sandbox, got {sandbox!r}"
    for feature in ("memories", "multi_agent"):
        setting = config.get(f"features.{feature}")
        assert setting != "true", f"Forbidden feature enabled: {feature}"
        assert feature in disabled or setting == "false", f"Codex must disable {feature}"

    request = request.replace("\r\n", "\n")
    assert "review-work" in request, "Agent request must select the review-work skill"
    assert "doctor" in request.lower(), "Agent request must select doctor mode"
    normalized_request = request.replace("\\", "/")
    absolute_prompt = re.escape((repo / PROMPT_PATH).as_posix())
    relative_prompt = re.escape(PROMPT_PATH.as_posix())
    # Sentence punctuation may follow the path; a filename suffix such as .backup
    # must not turn a different file into an accepted canonical reference.
    path_end = r"(?=$|[\s`'\"),;:!?\]}]|\.(?=$|[\s`'\"),;:!?\]}]))"
    reference = re.search(
        rf"(?<![\w/.:~-])(?:{absolute_prompt}|(?:\./)?{relative_prompt}){path_end}",
        normalized_request,
    )
    assert (
        PROMPT_TEXT.strip() in request or reference
    ), "Agent request must carry the canonical procedure or reference its repository path"
    # The canonical procedure itself describes both modes. Only the surrounding
    # request selects which mode the agent must execute for this invocation.
    mode_request = request.replace(PROMPT_TEXT.strip(), "")
    assert (
        "--check" in mode_request
    ) is check, "Explicit agent mode must match the doctor command"


def _snapshot(repo):
    """Compare filesystem effects, hashing the opaque executable without inspecting it."""
    snapshot = {}
    for path in repo.rglob("*"):
        relative = path.relative_to(repo).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            snapshot[relative] = ("directory",)
        else:
            snapshot[relative] = (
                "file",
                stat.S_IMODE(path.stat().st_mode),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snapshot


@pytest.mark.parametrize("check", [False, True], ids=["default", "check"])
@pytest.mark.parametrize("working_directory", ["outside", "repo", "nested"])
def test_doctor_transports_one_session_from_any_working_directory(
    doctor, check, working_directory
):
    """R1-R3: shell-sensitive paths, canonical procedure, mode and fresh exec transport."""
    nested = doctor.repo / "nested caller"
    nested.mkdir()
    cwd = {"outside": doctor.outside, "repo": doctor.repo, "nested": nested}[working_directory]
    args = ("doctor", "--check") if check else ("doctor",)
    result = doctor.run(*args, cwd=cwd)

    assert result.returncode == 0, _result_detail(result)
    _assert_session(_one_call(doctor, result), doctor.repo, check=check)
    assert HARNESS_STDOUT in result.stdout, _result_detail(result)
    assert HARNESS_STDERR in result.stderr, _result_detail(result)


def test_successive_invocations_each_start_a_fresh_session(doctor):
    """R3: a second invocation cannot reuse/resume/fork the previous session."""
    for check in (False, True):
        result = doctor.run("doctor", *(["--check"] if check else []))
        assert result.returncode == 0, _result_detail(result)
    calls = doctor.calls()
    assert len(calls) == 2, f"Expected exactly two independent exec sessions: {calls!r}"
    for call, check in zip(calls, (False, True)):
        _assert_session(call, doctor.repo, check=check)


@pytest.mark.parametrize("check", [False, True], ids=["default", "check"])
@pytest.mark.parametrize("exit_code", [1, 37])
def test_harness_failure_forwards_both_streams_and_status_without_retry(doctor, check, exit_code):
    """R4: failures remain failures, including a nonstandard nonzero exit status."""
    doctor.env["DELIVERY_TEST_EXIT"] = str(exit_code)
    result = doctor.run("doctor", *(["--check"] if check else []))

    assert result.returncode == exit_code, _result_detail(result)
    assert HARNESS_STDOUT in result.stdout, _result_detail(result)
    assert HARNESS_STDERR in result.stderr, _result_detail(result)
    _assert_session(_one_call(doctor, result), doctor.repo, check=check)


@pytest.mark.parametrize("check", [False, True], ids=["default", "check"])
def test_missing_codex_has_readable_stderr_and_exit_127(doctor, check, tmp_path):
    """R4: missing executable is a preflight error, not success or an agent launch."""
    empty_path = tmp_path / "empty executable search path"
    empty_path.mkdir()
    env = dict(doctor.env, PATH=str(empty_path))
    result = doctor.run("doctor", *(["--check"] if check else []), env=env)

    assert result.returncode == 127, _result_detail(result)
    diagnostic = result.stderr.lower()
    assert "codex" in diagnostic, _result_detail(result)
    assert "traceback" not in diagnostic, _result_detail(result)
    assert doctor.calls() == []


@pytest.mark.parametrize("check", [False, True], ids=["default", "check"])
@pytest.mark.parametrize("prompt_state", ["missing", "directory"])
def test_unavailable_canonical_prompt_fails_before_harness_launch(doctor, check, prompt_state):
    """R4: reject missing prompts and non-file paths before starting the harness."""
    doctor.prompt.unlink()
    if prompt_state == "directory":
        doctor.prompt.mkdir()
    result = doctor.run("doctor", *(["--check"] if check else []))

    assert result.returncode != 0, _result_detail(result)
    diagnostic = (result.stderr + result.stdout).lower()
    assert "prompt" in diagnostic, _result_detail(result)
    assert "traceback" not in diagnostic, _result_detail(result)
    assert doctor.calls() == [], "An unavailable canonical prompt must prevent the agent launch"


@pytest.mark.parametrize("check", [False, True], ids=["default", "check"])
def test_unreadable_regular_prompt_fails_before_harness_launch(unreadable_prompt, check):
    """R4: an existing regular file that cannot be read is a diagnosed preflight failure."""
    doctor = unreadable_prompt
    result = doctor.run("doctor", *(["--check"] if check else []))

    assert doctor.calls() == [], "An unreadable regular prompt must prevent the agent launch"
    assert result.returncode != 0, _result_detail(result)
    diagnostic = (result.stderr + result.stdout).lower()
    assert "prompt" in diagnostic, _result_detail(result)
    assert "traceback" not in diagnostic, _result_detail(result)


@pytest.mark.parametrize("args", [("--help",), ("doctor", "--help")], ids=["root", "doctor"])
@pytest.mark.parametrize(
    "missing_dependencies", [False, True], ids=["ready", "missing-dependencies"]
)
def test_help_describes_usage_without_launching_codex(doctor, args, missing_dependencies):
    """R5: both documented help entry points are successful and have no agent cost."""
    if missing_dependencies:
        doctor.fake_codex.unlink()
        doctor.prompt.unlink()
    result = doctor.run(*args)

    assert result.returncode == 0, _result_detail(result)
    output = (result.stdout + result.stderr).lower()
    assert "usage" in output and "doctor" in output, _result_detail(result)
    if args[0] == "doctor":
        assert "--check" in output, _result_detail(result)
    assert doctor.calls() == []


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("unknown-command",),
        ("--unknown-option",),
        ("doctor", "--unknown-option"),
        ("doctor", "--check", "--unknown-option"),
        ("doctor", "unexpected-positional"),
        ("doctor", "--check=yes"),
        ("--check",),
    ],
    ids=[
        "absent-command",
        "unknown-command",
        "unknown-root-option",
        "unknown-doctor-option",
        "unknown-check-option",
        "extra-positional",
        "check-with-value",
        "check-without-command",
    ],
)
@pytest.mark.parametrize(
    "missing_dependencies", [False, True], ids=["ready", "missing-dependencies"]
)
def test_invalid_invocations_show_usage_without_harness_or_project_writes(
    doctor, args, missing_dependencies
):
    """R5: parsing errors must remain cheap and leave all project files intact."""
    if missing_dependencies:
        doctor.fake_codex.unlink()
        doctor.prompt.unlink()
    before = _snapshot(doctor.repo)
    result = doctor.run(*args)
    after = _snapshot(doctor.repo)

    assert doctor.calls() == [], "Unsupported invocation launched the agent"
    assert after == before, "Unsupported invocation changed the project filesystem"
    assert result.returncode == 2, _result_detail(result)
    assert "usage" in (result.stdout + result.stderr).lower(), _result_detail(result)
