"""Independent command-boundary tests for LOCAL-VERIFICATION-CONTRACT v1.0.

The actual verifier and Git hooks are copied opaquely. External Python tools are
controlled subprocesses; their reports are fixtures, not numerical coverage proof.
"""

from io import BytesIO
from importlib.metadata import distribution
import codecs
import json
import os
from pathlib import Path
import shutil
import shlex
import stat
import subprocess
import sys
import sysconfig
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCEFREE = r"""
import linecache
import sys
import traceback
import warnings
linecache.getline = lambda *args, **kwargs: ""
linecache.getlines = lambda *args, **kwargs: []
def hook(kind, error, trace):
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SyntaxError):
            current.text = None
        current = current.__cause__ or current.__context__
    traceback.print_exception(kind, error, trace)
sys.excepthook = hook
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{filename}:{lineno}: {category.__name__}: {message}\n"
)
"""

# Keep coverage.py's installed API available for subprocess measurement. Only its
# command entry point is controlled; importing it must not run the fake CLI.
COVERAGE_API = r"""
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
import sys

fixture_package = Path(__file__).resolve().parent
search = [entry for entry in sys.path if Path(entry).resolve() != fixture_package.parent]
spec = PathFinder.find_spec("coverage", search)
if spec is None or spec.loader is None:
    raise ImportError("Fixture requires installed coverage.py for subprocess measurement")
real_coverage = module_from_spec(spec)
sys.modules["coverage"] = real_coverage
spec.loader.exec_module(real_coverage)
real_coverage.__path__.insert(0, str(fixture_package))
"""

# Each module is a genuine child command. No import or monkeypatch of the verifier.
TOOL = r"""
import json
import os
from pathlib import Path
import sys

name = Path(__file__).stem
args = sys.argv[1:]
record = {"tool": name, "args": args, "cwd": os.getcwd(), "python": sys.executable}
with Path(os.environ["VERIFICATION_TEST_CALLS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
print(f"fixture {name} stdout")
print(f"fixture {name} stderr", file=sys.stderr)
if name == "pytest" and os.environ.get("VERIFICATION_TEST_REAL_COVERAGE"):
    import dphtools

def option(*names):
    for index, arg in enumerate(args):
        for flag in names:
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
            if arg == flag and index + 1 < len(args):
                return args[index + 1]
    return None

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")

scenario = os.environ.get("VERIFICATION_TEST_REPORT", "valid")
junit = option("--junitxml", "--junit-xml")
if junit and scenario != "missing-tests":
    cases = {
        "valid": '<testsuite tests="1" failures="0" errors="0" skipped="0">'
                 '<testcase classname="fixture" name="passes"/></testsuite>',
        "empty-tests": '<testsuite tests="0" failures="0" errors="0" skipped="0"/>',
        "empty-positive-count": '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
        "skipped-tests": '<testsuite tests="1" failures="0" errors="0" skipped="1">'
                         '<testcase name="skipped"><skipped/></testcase></testsuite>',
        "failed-tests": '<testsuite tests="1" failures="1" errors="0" skipped="0">'
                        '<testcase name="failed"><failure/></testcase></testsuite>',
        "errored-tests": '<testsuite tests="1" failures="0" errors="1" skipped="0">'
                         '<testcase name="errored"><error/></testcase></testsuite>',
        "malformed-tests": '<testsuites broken',
    }
    write(junit, cases.get(scenario, cases["valid"]))

if name == "build" and not set(args) & {"--version", "-V", "--help", "-h"}:
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("project", nargs="?", default=".")
    for short, long in (("-s", "--sdist"), ("-w", "--wheel"), ("-n", "--no-isolation"),
                        ("-x", "--skip-dependency-check"), ("-v", "--verbose")):
        parser.add_argument(short, long, action="store_true")
    parser.add_argument("-o", "--outdir")
    parser.add_argument("--installer")
    parser.add_argument("-C", "--config-setting", action="append")
    build_args = parser.parse_args(args)
    if Path(build_args.project).resolve() != Path.cwd():
        raise ValueError("Build must consume the fixture repository")
    output = option("--outdir", "-o") or "dist"
    artifacts = [Path(output) / "fixture-1.0-py3-none-any.whl",
                 Path(output) / "fixture-1.0.tar.gz"]
    for artifact in artifacts:
        write(artifact, "controlled distribution from this build invocation")
    write(os.environ["VERIFICATION_TEST_BUILD_ARTIFACTS"],
          json.dumps([str(artifact.resolve()) for artifact in artifacts]))

if name == "coverage" and "json" in args and scenario != "missing-coverage":
    summary = {"covered_lines": 1, "num_statements": 1, "percent_covered": 100.0,
               "percent_covered_display": "100", "missing_lines": 0,
               "excluded_lines": 0, "num_branches": 0, "num_partial_branches": 0,
               "covered_branches": 0, "missing_branches": 0}
    files = {}
    owned = list(Path("dphtools").rglob("*.py")) + [Path("tools/verification.py"),
                                                    Path("tools/delivery")]
    for path in owned:
        if path.as_posix() == "dphtools/_version.py":
            continue
        files[path.as_posix()] = {"executed_lines": [1], "summary": dict(summary),
                                 "missing_lines": [], "excluded_lines": [],
                                 "executed_branches": [], "missing_branches": []}
    target = files["dphtools/never_imported.py"]
    if scenario == "absent-owned":
        del files["dphtools/never_imported.py"]
    elif scenario == "absent-helper":
        del files["tools/verification.py"]
    elif scenario == "excluded-coverage":
        target["excluded_lines"] = [2]
        target["summary"]["excluded_lines"] = 1
    elif scenario == "missing-statement":
        target["missing_lines"] = [2]
        target["summary"].update(covered_lines=9999, num_statements=10000,
                                 missing_lines=1, percent_covered=99.99)
    elif scenario == "missing-branch":
        target["missing_branches"] = [[1, 2]]
        target["summary"].update(num_branches=10000, covered_branches=9999,
                                 missing_branches=1, num_partial_branches=1,
                                 percent_covered=99.99)
    totals = {key: sum(value["summary"][key] for value in files.values())
              for key in summary if not key.startswith("percent")}
    totals.update(percent_covered=100.0, percent_covered_display="100")
    data = {"meta": {"format": 3, "version": "7.16.1", "branch_coverage":
                     scenario != "no-branches", "show_contexts": False},
            "files": files, "totals": totals}
    write(option("-o", "--output") or "coverage.json",
          "{" if scenario == "malformed-coverage" else json.dumps(data))
if name == "coverage" and "xml" in args and scenario != "missing-coverage-xml":
    count = len(list(Path("dphtools").rglob("*.py"))) + 1
    branches = 'branches-valid="10000" branches-covered="9999"' if (
        scenario == "xml-missing-branch"
    ) else 'branches-valid="0" branches-covered="0"'
    write(option("-o", "--output") or "coverage.xml",
          '<coverage broken' if scenario == "malformed-coverage-xml" else
          f'<coverage branch-rate="1" line-rate="1" version="fixture" '
          f'lines-valid="{count}" lines-covered="{count}" {branches}>'
          '<packages/></coverage>')

failure = os.environ.get("VERIFICATION_TEST_FAIL", "")
label = name + (":" + args[0] if name == "coverage" and args else "")
sys.exit(23 if failure in (name, label) else 0)
"""


def _executable(path, body):
    """Use a native Windows launcher or exec the exact selected POSIX interpreter."""
    if os.name == "nt":
        wrapper = Path(sysconfig.get_path("scripts")) / "pytest.exe"
        assert wrapper.is_file(), "Fixture requires pytest's installed Windows launcher"
        with zipfile.ZipFile(wrapper) as archive:
            offset = min(info.header_offset for info in archive.infolist())
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("__main__.py", body)
        path = path.with_suffix(".exe")
        with wrapper.open("rb") as source, path.open("wb") as target:
            target.write(source.read(offset))
            target.write(payload.getvalue())
    else:
        # This shell/Python preamble handles spaces while retaining the venv path.
        # Resolving or relinking sys.executable can change Python 3.10's prefix.
        preamble = "#!/bin/sh\n'''exec' " + shlex.quote(sys.executable) + " \"$0\" \"$@\"\n' '''\n"
        path.write_text(preamble + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _prove_executable_runtime(command):
    """Validate the fixture launcher and real coverage API before hook evidence."""
    body = r"""
import json
import sys
import coverage
from coverage import Coverage
print(json.dumps({"python": sys.executable, "prefix": sys.prefix,
                  "coverage_file": coverage.__file__, "coverage_api": callable(Coverage)}))
"""
    executable = _executable(command.bin / "runtime probe", body)
    result = subprocess.run(
        [str(executable)],
        cwd=command.outside,
        env=command.env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, f"Executable fixture preflight failed: {_detail(result)}"
    observed = json.loads(result.stdout)
    assert Path(observed["python"]) == Path(sys.executable), observed
    assert Path(observed["prefix"]).resolve() == Path(sys.prefix).resolve(), observed
    installed = distribution("coverage").locate_file("coverage/__init__.py")
    assert Path(observed["coverage_file"]).resolve() == Path(installed).resolve(), observed
    assert observed["coverage_api"] is True, observed
    command.record.with_name("interpreter-probe.json").write_text(result.stdout, encoding="utf-8")


def _detail(result):
    return f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"


class VerificationCommand:
    def __init__(self, tmp_path):
        self.repo = tmp_path / "repo space & $TOKEN %TOKEN% 'quoted'"
        self.entry = self.repo / "tools" / "verification.py"
        self.entry.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "tools" / "verification.py", self.entry)
        self.outside = tmp_path / "unrelated working directory"
        self.outside.mkdir()
        self.modules = tmp_path / "controlled tools"
        self.modules.mkdir()
        (self.modules / "sitecustomize.py").write_text(SOURCEFREE, encoding="utf-8")
        for name in (
            "black",
            "flake8",
            "pydocstyle",
            "mypy",
            "pip_audit",
            "build",
            "pip",
            "pytest",
        ):
            (self.modules / f"{name}.py").write_text(TOOL, encoding="utf-8")
        coverage_package = self.modules / "coverage"
        coverage_package.mkdir()
        (coverage_package / "__init__.py").write_text(COVERAGE_API, encoding="utf-8")
        (coverage_package / "__main__.py").write_text(
            TOOL.replace("name = Path(__file__).stem", 'name = "coverage"'), encoding="utf-8"
        )
        self.record = tmp_path / "tool-calls.jsonl"
        self.bin = tmp_path / "fixture bin"
        self.bin.mkdir()
        self.env = dict(
            os.environ,
            PYTHONPATH=str(self.modules),
            PYTHONIOENCODING="utf-8",
            PYTHONDONTWRITEBYTECODE="1",
            COVERAGE_FILE=str(self.repo / ".coverage"),
            VERIFICATION_TEST_CALLS=str(self.record),
            VERIFICATION_TEST_BUILD_ARTIFACTS=str(tmp_path / "built-distributions.json"),
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
        )
        # Retain inherited subprocess measurement; isolate only the fixture's own
        # coverage CLI data so its erase/combine commands cannot affect the parent.
        self.env.pop("PYTEST_ADDOPTS", None)
        for relative in (
            "dphtools/__init__.py",
            "dphtools/never_imported.py",
            "dphtools/sub/__init__.py",
            "dphtools/sub/deep.py",
            "dphtools/_version.py",
        ):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('"""An owned fixture module."""\nVALUE = 1\n', encoding="utf-8")
        (self.repo / "pyproject.toml").write_text(
            '[tool.black]\nline-length = 99\n[tool.pydocstyle]\nconvention = "numpy"\n'
            '[tool.coverage.run]\nbranch = true\nsource = ["dphtools", "tools"]\n'
            'omit = ["dphtools/_version.py"]\n',
            encoding="utf-8",
        )
        (self.repo / "requirements-dev.lock").write_text("fixture==1.0\n", encoding="utf-8")
        (self.repo / "tools" / "delivery").write_text(
            '#!/usr/bin/env python\n"""Owned delivery fixture."""\n', encoding="utf-8"
        )

    def run(self, *args):
        return subprocess.run(
            [sys.executable, str(self.entry), *args],
            cwd=self.outside,
            env=self.env,
            input="",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]

    def reports(self):
        return sorted((self.repo / "reports" / "verification").glob("*/checks.json"))


@pytest.fixture
def verifier(tmp_path):
    command = VerificationCommand(tmp_path)
    probe = subprocess.run(
        [sys.executable, "-m", "black", "fixture-probe"],
        cwd=command.repo,
        env=command.env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert probe.returncode == 0, f"External-tool fixture failed: {_detail(probe)}"
    assert command.calls()[0]["tool"] == "black"
    command.record.unlink()
    return command


@pytest.mark.parametrize("args", [(), ("invalid",), ("fast", "unexpected")])
def test_invalid_mode_is_usage_error(verifier, args):
    result = verifier.run(*args)
    assert result.returncode == 2, _detail(result)
    assert verifier.calls() == []


def test_fast_runs_all_tools_from_script_repository(verifier):
    result = verifier.run("fast")
    assert result.returncode == 0, _detail(result)
    calls = verifier.calls()
    assert [call["tool"] for call in calls] == ["black", "flake8", "pydocstyle"]
    assert all(Path(call["cwd"]).resolve() == verifier.repo.resolve() for call in calls)
    assert all(Path(call["python"]) == Path(sys.executable) for call in calls)
    black_args = calls[0]["args"]
    assert "--check" in black_args
    assert "--line-length=99" in black_args or any(
        pair == ["--line-length", "99"]
        for pair in [black_args[index : index + 2] for index in range(len(black_args))]
    ), calls[0]
    _check_manifest(verifier, result, "fast")


def test_full_success_orders_tools_and_retains_reports(verifier):
    result = verifier.run("full")
    assert result.returncode == 0, _detail(result)
    calls = verifier.calls()
    assert _sequence(calls) == FULL_SEQUENCE
    assert all(Path(call["python"]) == Path(sys.executable) for call in calls)
    assert all(Path(call["cwd"]).resolve() == verifier.repo.resolve() for call in calls)
    audit = next(call for call in calls if call["tool"] == "pip_audit")
    assert any(arg.endswith(".lock") for arg in audit["args"]), audit
    tests = next(call for call in calls if call["tool"] == "coverage" and "run" in call["args"])
    assert "pytest" in tests["args"] and "--doctest-modules" in tests["args"]
    assert {"dphtools", "tests"} <= set(tests["args"])
    report = _check_manifest(verifier, result, "full")
    assert {"pytest.xml", "coverage.json", "coverage.xml"} <= {
        path.name for path in report.iterdir()
    }
    coverage = json.loads((report / "coverage.json").read_text(encoding="utf-8"))
    assert "dphtools/never_imported.py" in coverage["files"]
    assert "tools/verification.py" in coverage["files"]
    assert "dphtools/_version.py" not in coverage["files"]
    assert coverage["totals"]["num_branches"] == 0
    _assert_full_operations(verifier, calls)


def _assert_full_operations(verifier, calls):
    """Checking, building and installing must do work on the owned fixture inputs."""
    by_tool = {call["tool"]: call for call in calls}
    informational = {"--version", "-V", "--help", "-h", "--help-extra"}
    for name in ("mypy", "build", "pip"):
        assert not informational.intersection(by_tool[name]["args"]), by_tool[name]
    mypy = by_tool["mypy"]
    # Existing configured owned targets; no additional type-check target is introduced.
    targets = {
        (Path(mypy["cwd"]) / arg).resolve() for arg in mypy["args"] if not arg.startswith("-")
    }
    assert {
        (verifier.repo / name).resolve()
        for name in ("dphtools", "tools/delivery", "tools/verification.py")
    } <= targets, mypy
    built = Path(verifier.env["VERIFICATION_TEST_BUILD_ARTIFACTS"])
    assert built.is_file(), "Build did not produce fixture distributions"
    artifacts = {Path(name).resolve() for name in json.loads(built.read_text(encoding="utf-8"))}
    assert artifacts and all(path.is_file() for path in artifacts)
    pip = by_tool["pip"]
    assert "install" in pip["args"], f"Expected pip install operation: {pip}"
    install_args = pip["args"][pip["args"].index("install") + 1 :]
    inputs = {
        (Path(pip["cwd"]) / arg).resolve() for arg in install_args if not arg.startswith("-")
    }
    assert (
        inputs & artifacts
    ), f"Installation did not consume a distribution from this build: {pip}"


@pytest.mark.parametrize("failed", ["black", "flake8", "pydocstyle"])
def test_fast_attempts_later_tools_after_failure(verifier, failed):
    verifier.env["VERIFICATION_TEST_FAIL"] = failed
    result = verifier.run("fast")
    assert result.returncode == 1, _detail(result)
    assert [call["tool"] for call in verifier.calls()] == ["black", "flake8", "pydocstyle"]
    _check_manifest(verifier, result, "fast", failed)


@pytest.mark.parametrize(
    "scenario",
    [
        "empty-tests",
        "empty-positive-count",
        "skipped-tests",
        "failed-tests",
        "errored-tests",
        "malformed-tests",
        "missing-tests",
        "missing-coverage",
        "malformed-coverage",
        "absent-owned",
        "absent-helper",
        "no-branches",
        "excluded-coverage",
        "missing-statement",
        "missing-branch",
        "missing-coverage-xml",
        "malformed-coverage-xml",
        "xml-missing-branch",
    ],
)
def test_full_rejects_invalid_reports_even_when_all_tools_pass(verifier, scenario):
    verifier.env["VERIFICATION_TEST_REPORT"] = scenario
    result = verifier.run("full")
    assert result.returncode == 1, _detail(result)
    assert _sequence(verifier.calls()) == FULL_SEQUENCE, _detail(result)
    _check_manifest(verifier, result, "full")


FULL_SEQUENCE = [
    "black",
    "flake8",
    "pydocstyle",
    "mypy",
    "pip_audit",
    "build",
    "pip",
    "coverage:erase",
    "coverage:run",
    "coverage:combine",
    "coverage:json",
    "coverage:xml",
    "coverage:report",
]


def _sequence(calls):
    return [
        call["tool"] + (":" + call["args"][0] if call["tool"] == "coverage" else "")
        for call in calls
    ]


def _check_manifest(verifier, result, mode, failed=None):
    paths = verifier.reports()
    assert paths, _detail(result)
    # Every invocation must identify its new report directory, independent of its name.
    paths = [path for path in paths if str(path.parent) in result.stdout + result.stderr]
    assert len(paths) == 1, _detail(result)
    report = paths[0].parent
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["document_version"] == "1.0"
    assert data["mode"] == mode
    assert Path(data["python"]) == Path(sys.executable)
    assert isinstance(data["platform"], str) and data["platform"]
    steps = [step for step in data["steps"] if step["command"] is not None]
    calls = verifier.calls()
    assert len(steps) == len(calls)
    assert len({step["name"] for step in steps}) == len(steps)
    logs = [path.read_text(encoding="utf-8") for path in report.glob("*.log")]
    assert len(logs) >= len(steps), "Each tool must have an individual diagnostic log"
    for step, call, label in zip(steps, calls, _sequence(calls)):
        assert step["name"]
        assert step["command"] == [sys.executable, "-m", call["tool"], *call["args"]]
        expected = 23 if failed in (call["tool"], label) else 0
        assert step["returncode"] == expected
        assert any(
            f"fixture {call['tool']} stdout" in log and f"fixture {call['tool']} stderr" in log
            for log in logs
        )
    return report


@pytest.mark.parametrize("failed", FULL_SEQUENCE)
def test_full_failure_stays_failed_and_attempts_all_later_steps(verifier, failed):
    verifier.env["VERIFICATION_TEST_FAIL"] = failed
    result = verifier.run("full")
    assert result.returncode == 1, _detail(result)
    assert _sequence(verifier.calls()) == FULL_SEQUENCE
    _check_manifest(verifier, result, "full", failed)


def test_full_cannot_reuse_previous_success_reports(verifier):
    first = verifier.run("full")
    assert first.returncode == 0, _detail(first)
    old = _check_manifest(verifier, first, "full")
    before = {path.name: path.read_bytes() for path in old.iterdir() if path.is_file()}
    verifier.record.unlink()
    verifier.env["VERIFICATION_TEST_REPORT"] = "missing-tests"
    second = verifier.run("full")
    assert second.returncode == 1, _detail(second)
    new = _check_manifest(verifier, second, "full")
    assert new != old
    assert not (new / "pytest.xml").exists()
    assert {path.name: path.read_bytes() for path in old.iterdir() if path.is_file()} == before


def test_fast_invocations_have_distinct_report_directories(verifier):
    directories = []
    for _ in range(2):
        result = verifier.run("fast")
        assert result.returncode == 0, _detail(result)
        directories.append(_check_manifest(verifier, result, "fast"))
        verifier.record.unlink()
    assert directories[0] != directories[1]


def _real_configuration(verifier):
    """Use configuration through tools without inspecting its bytes."""
    (verifier.repo / "pyproject.toml").unlink()
    copied = []
    for name in ("setup.cfg", "pyproject.toml", "tox.ini", ".pydocstyle", ".pydocstyle.ini"):
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, verifier.repo / name)
            copied.append(name)
    assert copied, "Contract/configuration blocker: no conventional tool configuration found"


def test_real_docstring_tool_reaches_nested_modules_with_repository_configuration(verifier):
    _real_configuration(verifier)
    (verifier.modules / "pydocstyle.py").unlink()
    good = verifier.run("fast")
    assert good.returncode == 0, f"Valid docstring fixture failed: {_detail(good)}"
    verifier.record.unlink()
    nested = verifier.repo / "dphtools" / "sub" / "deep.py"
    nested.write_text(
        '''"""An owned fixture module."""


def scale(value):
    """Scale a value.

    Parameters
    value : float
        Value to scale.

    Returns
    -------
    float
        Scaled value.
    """
    return value
''',
        encoding="utf-8",
    )
    bad = verifier.run("fast")
    assert bad.returncode == 1, _detail(bad)
    diagnostic = bad.stdout + bad.stderr
    assert "deep.py" in diagnostic and "D407" in diagnostic, _detail(bad)


def test_real_coverage_reports_never_imported_owned_sources(verifier):
    _real_configuration(verifier)
    shutil.rmtree(verifier.modules / "coverage")
    verifier.env["VERIFICATION_TEST_REAL_COVERAGE"] = "1"
    result = verifier.run("full")
    # Only the package initializer ran. Its unimported siblings must make this fail.
    assert result.returncode == 1, _detail(result)
    reports = verifier.reports()
    assert len(reports) == 1, _detail(result)
    data = json.loads((reports[0].parent / "coverage.json").read_text(encoding="utf-8"))
    assert data["meta"]["branch_coverage"] is True
    files = {name.replace("\\", "/"): value for name, value in data["files"].items()}
    for name in ("dphtools/never_imported.py", "dphtools/sub/deep.py", "tools/verification.py"):
        assert name in files, f"Never-imported owned file absent from real measurement: {name}"
        assert files[name]["summary"]["missing_lines"] > 0
    assert "dphtools/_version.py" not in files


def test_real_lint_configuration_retains_critical_errors(verifier):
    _real_configuration(verifier)
    (verifier.modules / "flake8.py").unlink()
    target = verifier.repo / "dphtools" / "sub" / "deep.py"
    target.write_text('VALUE = "' + "x" * 120 + '"\n', encoding="utf-8")
    (verifier.repo / "tests").mkdir()
    for name in ("setup.py", "versioneer.py"):
        (verifier.repo / name).write_text('"""Tooling fixture."""\n', encoding="utf-8")

    def lint_fixture():
        return subprocess.run(
            [sys.executable, "-m", "flake8", "dphtools"],
            cwd=verifier.repo,
            env=verifier.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )

    style_only = lint_fixture()
    assert style_only.returncode == 0, _detail(style_only)
    public_style = verifier.run("fast")
    assert public_style.returncode == 0, _detail(public_style)
    assert _sequence(verifier.calls()) == ["black", "pydocstyle"]
    verifier.record.unlink()
    target.write_text("VALUE = undefined_fixture_name\n", encoding="utf-8")
    critical = lint_fixture()
    assert critical.returncode != 0, _detail(critical)
    assert "F821" in critical.stdout + critical.stderr, _detail(critical)
    public_critical = verifier.run("fast")
    assert public_critical.returncode == 1, _detail(public_critical)
    assert "deep.py" in public_critical.stdout + public_critical.stderr, _detail(public_critical)
    assert "F821" in public_critical.stdout + public_critical.stderr, _detail(public_critical)
    assert _sequence(verifier.calls()) == ["black", "pydocstyle"]


def test_undecodable_child_output_retains_failure_diagnostics_and_later_checks(verifier):
    """Undecodable bytes cannot replace a child status or interrupt scheduled checks."""
    # Python's UTF-8 mode makes the subprocess default deterministic on all three
    # supported platforms, independently of the test runner's locale/code page.
    verifier.env["PYTHONUTF8"] = "1"
    encoding_probe = subprocess.run(
        [sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
        cwd=verifier.outside,
        env=verifier.env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert encoding_probe.returncode == 0, _detail(encoding_probe)
    encoding = encoding_probe.stdout.strip()
    assert codecs.lookup(encoding).name == "utf-8", "UTF-8-mode fixture did not take effect"
    prefix = r"""
import sys
sys.stdout.buffer.write(b"BYTE_STDOUT_START[\xff]BYTE_STDOUT_END\n")
sys.stdout.buffer.flush()
sys.stderr.buffer.write(b"BYTE_STDERR_START[\xff]BYTE_STDERR_END\n")
sys.stderr.buffer.flush()
"""
    (verifier.modules / "black.py").write_text(prefix + TOOL, encoding="utf-8")
    verifier.env["VERIFICATION_TEST_FAIL"] = "black"
    probe = subprocess.run(
        [sys.executable, "-m", "black", "byte-fixture-probe"],
        cwd=verifier.repo,
        env=verifier.env,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert probe.returncode == 23, f"Byte-output fixture failed: {probe!r}"
    for stream in (probe.stdout, probe.stderr):
        assert b"\xff" in stream
        with pytest.raises(UnicodeDecodeError):
            stream.decode(encoding)
    verifier.record.unlink()

    result = verifier.run("fast")
    assert result.returncode == 1, _detail(result)
    assert _sequence(verifier.calls()) == ["black", "flake8", "pydocstyle"], _detail(result)
    report = _check_manifest(verifier, result, "fast", failed="black")
    logs = "\n".join(path.read_text(encoding="utf-8") for path in report.glob("*.log"))
    for destination in (result.stdout + result.stderr, logs):
        for stream in ("STDOUT", "STDERR"):
            begin, end = f"BYTE_{stream}_START[", f"]BYTE_{stream}_END"
            assert begin in destination and end in destination, _detail(result)
            representation = destination.split(begin, 1)[1].split(end, 1)[0]
            assert representation.strip(), "Undecodable bytes were silently discarded"


def test_real_coverage_instruments_executed_owned_modules_at_every_depth(verifier):
    """Executing an owned nested statement must produce measured coverage."""
    _real_configuration(verifier)
    shutil.rmtree(verifier.modules / "coverage")
    values = {
        "dphtools/control.py": 66,
        "dphtools/subpkg/probe.py": 11,
        "dphtools/utils/subpkg/probe.py": 22,
        "tools/subpkg/probe.py": 33,
        "dphtools/subpkg/_version.py": 44,
        "dphtools/_version.py": 55,
    }
    for relative, value in values.items():
        target = verifier.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # A single assignment is exactly one instrumentable statement, no branches.
        target.write_text(f"FIXTURE_VALUE = {value}\n", encoding="utf-8")
        for directory in target.parents:
            if directory == verifier.repo:
                break
            initializer = directory / "__init__.py"
            if not initializer.exists():
                initializer.write_text("", encoding="utf-8")
    receipt = verifier.record.with_name("executed-nested-modules.json")
    verifier.env["VERIFICATION_TEST_EXECUTE_MODULES"] = json.dumps(values)
    verifier.env["VERIFICATION_TEST_EXECUTION_RECEIPT"] = str(receipt)
    execution = r"""
import json
import os
from pathlib import Path
import runpy
values = json.loads(os.environ["VERIFICATION_TEST_EXECUTE_MODULES"])
executed = {name: runpy.run_path(name)["FIXTURE_VALUE"] for name in values}
Path(os.environ["VERIFICATION_TEST_EXECUTION_RECEIPT"]).write_text(
    json.dumps(executed), encoding="utf-8"
)
"""
    (verifier.modules / "pytest.py").write_text(execution + TOOL, encoding="utf-8")
    result = verifier.run("full")
    assert (
        receipt.is_file()
    ), f"Nested-module execution fixture did not complete: exit={result.returncode}"
    assert json.loads(receipt.read_text(encoding="utf-8")) == values
    reports = verifier.reports()
    assert len(reports) == 1, f"Expected a completed full report: exit={result.returncode}"
    data = json.loads((reports[0].parent / "coverage.json").read_text(encoding="utf-8"))
    assert data["meta"]["branch_coverage"] is True
    summaries = {
        name.replace("\\", "/"): entry["summary"] for name, entry in data["files"].items()
    }
    assert "dphtools/_version.py" not in summaries
    expected = {
        name: {"num_statements": 1, "covered_lines": 1, "missing_lines": 0}
        for name in values
        if name != "dphtools/_version.py"
    }
    measured = {
        name: {key: summaries[name][key] for key in counts} if name in summaries else None
        for name, counts in expected.items()
    }
    # Report only numeric summaries, never source or missing-line/branch listings.
    assert measured == expected, f"Executed fixture statements were not measured: {measured}"


def test_importing_verifier_does_not_run_checks_or_create_reports(verifier):
    """An ordinary module import must be inert even when the caller has mode-like args."""
    driver = r"""
import importlib.util
import sys
path = sys.argv[1]
sys.argv = [path, "full"]
spec = importlib.util.spec_from_file_location("opaque_verification_fixture", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print("opaque module import completed")
"""
    result = subprocess.run(
        [sys.executable, "-c", driver, str(verifier.entry)],
        cwd=verifier.outside,
        env=verifier.env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, _detail(result)
    assert "opaque module import completed" in result.stdout, _detail(result)
    assert verifier.calls() == [], "Importing the verifier launched verification tools"
    assert not list(
        (verifier.repo / "reports" / "verification").rglob("*")
    ), "Importing the verifier created run reports"


def test_os_launch_failure_is_diagnosed_and_later_checks_still_run(verifier):
    """A denied OS launch creates no tool process and cannot abort the remaining checks."""
    receipt = verifier.record.with_name("denied-launch.json")
    verifier.env["VERIFICATION_TEST_DENIED_LAUNCH"] = str(receipt)
    fault = r"""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

def deny_black_launch(event, args):
    if event != "subprocess.Popen":
        return
    argv = args[1]
    command_prefix = [sys.executable, "-m", "black"]
    if isinstance(argv, str):
        # Windows audits the serialized command line, not an argv sequence.
        prefix = subprocess.list2cmdline(command_prefix)
        is_black = argv == prefix or argv.startswith(prefix + " ")
    elif isinstance(argv, (list, tuple)):
        is_black = list(argv[:3]) == command_prefix
    else:
        return
    if is_black:
        Path(os.environ["VERIFICATION_TEST_DENIED_LAUNCH"]).write_text(
            json.dumps({"command": argv}), encoding="utf-8"
        )
        raise PermissionError(errno.EACCES, "fixture scheduled launch unavailable")

sys.addaudithook(deny_black_launch)
"""
    # The fault is confined to the child fixture's OS/subprocess audit boundary.
    # No verifier function, subprocess implementation, or host executable is changed.
    (verifier.modules / "sitecustomize.py").write_text(SOURCEFREE + fault, encoding="utf-8")
    result = verifier.run("fast")
    assert receipt.is_file(), f"OS launch-fault fixture was not exercised: {_detail(result)}"
    denied = json.loads(receipt.read_text(encoding="utf-8"))["command"]
    assert result.returncode == 1, _detail(result)
    assert _sequence(verifier.calls()) == ["flake8", "pydocstyle"], _detail(result)
    assert all(Path(call["python"]) == Path(sys.executable) for call in verifier.calls())
    reports = verifier.reports()
    assert len(reports) == 1, _detail(result)
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    assert data["document_version"] == "1.0" and data["mode"] == "fast"
    steps = [step for step in data["steps"] if step["command"] is not None]
    assert len(steps) == 3
    reported_command = steps[0]["command"]
    if isinstance(denied, str):
        reported_command = subprocess.list2cmdline(reported_command)
    assert reported_command == denied
    assert isinstance(steps[0]["returncode"], int) and steps[0]["returncode"] != 0
    for step, call in zip(steps[1:], verifier.calls()):
        assert step["command"] == [sys.executable, "-m", call["tool"], *call["args"]]
        assert step["returncode"] == 0
    diagnostic = "fixture scheduled launch unavailable"
    assert diagnostic in result.stdout + result.stderr, _detail(result)
    assert any(
        diagnostic in path.read_text(encoding="utf-8") for path in reports[0].parent.glob("*.log")
    ), "Launch-failure diagnostics were not retained in tool logs"
