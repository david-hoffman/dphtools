"""Test-only provisioning for real disposable-interpreter coverage.

No owned source is inspected. Coverage reads the existing configuration normally;
this fixture never creates line/arc records or substitutes process results.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import uuid

SAFE_STARTUP = r"""
import os
import sys
import warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: f"{category.__name__}: {message}\n"
sys.excepthook = lambda kind, value, traceback: print(f"{kind.__name__}: {value}", file=sys.stderr)
try:
    # This disposable child's explicit file config/data must win over the
    # serialized configuration inherited from a parent subprocess patch.
    # Ordinary startup reapplies subprocess measurement from the same config.
    if os.environ.get("COVERAGE_PROCESS_START"):
        os.environ.pop("COVERAGE_PROCESS_CONFIG", None)
    import coverage
    coverage.process_startup()
except Exception as error:
    # site normally prints and ignores .pth exceptions. Fail this real setup
    # visibly without rendering startup/library source or continuing unmeasured.
    print(f"Coverage startup failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    os._exit(71)
"""


def settings(root, project):
    """Select coverage's ordinary config and its hashed locked tool requirement."""
    import coverage

    config = os.environ.get("COVERAGE_PROCESS_START") or coverage.Coverage().config.config_file
    assert (
        config and Path(config).is_file()
    ), "Measurement requires existing coverage configuration"
    match = re.search(
        r"^coverage(?:\[[^\]]+\])?==[^\n]*(?:\n[ \t]+[^\n]*)*",
        (project / "requirements-dev.lock").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "Locked coverage requirement was not found"
    requirement = match.group(0)
    assert "--hash=sha256:" in requirement, "Coverage provisioning must use locked hashes"
    lock = root / "coverage-tool-requirement.txt"
    lock.write_text(requirement, encoding="utf-8")
    # Keep child data where ordinary combine can find it during canonical tests.
    # Standalone measurement uses its own retained fixture directory.
    active = coverage.Coverage.current() is not None or bool(
        os.environ.get("COVERAGE_PROCESS_START")
    )
    data = os.environ.get("COVERAGE_FILE") or str(
        project / ".coverage" if active else root / ".coverage"
    )
    return {
        "config": str(Path(config).resolve()),
        "requirement": str(lock),
        "data": str(Path(data).absolute()),
        "version": re.search(r"==([^\s\\]+)", requirement).group(1),
    }


def run(real_popen, argv, cwd, env, timeout=120):
    process = real_popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(timeout=timeout)
    # Never return raw child source/traceback streams as assertion operands.
    message = []
    for label, stream in (("stdout", stdout), ("stderr", stderr)):
        lines = stream.decode("utf-8", errors="replace").splitlines()
        if any("Traceback (most recent call last)" in line for line in lines):
            lines = lines[-1:]
        else:
            lines = [line for line in lines if line and not line[0].isspace()]
        message.extend(label + ": " + line for line in lines[-2:])
    return process.returncode, stdout, message


def provision(real_popen, python, env, outside, measurement, record):
    """Install ordinary locked coverage into this interpreter, then validate startup."""
    code, _, message = run(
        real_popen,
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--require-hashes",
            "-r",
            measurement["requirement"],
        ],
        outside,
        env,
    )
    record(
        {
            "event": "coverage-setup",
            "python": str(python),
            "returncode": code,
            "diagnostic": message,
        }
    )
    assert code == 0, "Locked coverage setup failed: " + " | ".join(message)
    code, stdout, message = run(
        real_popen,
        [
            str(python),
            "-c",
            "import json, sysconfig; print(json.dumps(sysconfig.get_path('purelib')))",
        ],
        outside,
        env,
    )
    assert code == 0, "Actual interpreter path lookup failed: " + " | ".join(message)
    site = Path(json.loads(stdout))
    # The installation prefix is lexical: a venv python may symlink to its parent.
    prefix = python.absolute().parent.parent
    assert site.resolve().is_relative_to(
        prefix.resolve()
    ), "Coverage setup escaped disposable prefix"
    (site / "release_fixture_startup.py").write_text(SAFE_STARTUP, encoding="utf-8")
    # Run before coverage 7.16.1's a1_coverage.pth: otherwise it starts with
    # inherited serialized settings and hides this fixture's config errors.
    (site / "00_release_fixture_coverage.pth").write_text(
        "import release_fixture_startup\n", encoding="utf-8"
    )
    data_base = measurement["data"] + ".clean-" + uuid.uuid4().hex
    measured_env = dict(env, COVERAGE_PROCESS_START=measurement["config"], COVERAGE_FILE=data_base)
    code, stdout, message = run(
        real_popen,
        [
            str(python),
            "-c",
            "import coverage, json, sys; from pathlib import Path; "
            "assert coverage.Coverage.current() is not None; "
            "assert Path(coverage.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
            "print(json.dumps({'prefix': sys.prefix, 'coverage_origin': coverage.__file__, 'version': coverage.__version__}))",
        ],
        outside,
        measured_env,
    )
    record(
        {
            "event": "coverage-startup",
            "data_base": data_base,
            "python": str(python),
            "returncode": code,
            "diagnostic": message,
            **({"result": json.loads(stdout)} if code == 0 else {}),
        }
    )
    assert code == 0, "Actual coverage startup failed: " + " | ".join(message)
    assert (
        json.loads(stdout)["version"] == measurement["version"]
    ), "Coverage version differs from lock"
    return {"COVERAGE_PROCESS_START": measurement["config"], "COVERAGE_FILE": data_base}


def controls(real_popen, argv, cwd, env, installed, record):
    """Actual helper failures after its real successful control; no fake returns."""
    code, _, message = run(real_popen, [*argv[:-1], "999.0.0"], cwd, env)
    record(
        {
            "event": "helper-control",
            "control": "wrong-version",
            "returncode": code,
            "diagnostic": message,
            "python": argv[0],
            "helper": argv[1],
        }
    )
    assert code != 0 and message, "Actual helper accepted a wrong expected version"
    # Append a controlled installed-state defect without reading owned source.
    with Path(installed["origin"]).open("a", encoding="utf-8") as stream:
        stream.write("\n__version__ = '999.0.0'\n")
    code, _, message = run(real_popen, argv, cwd, env)
    record(
        {
            "event": "helper-control",
            "control": "bad-installed-state",
            "returncode": code,
            "diagnostic": message,
            "python": argv[0],
            "helper": argv[1],
        }
    )
    assert code != 0 and message, "Actual helper accepted a bad installed state"
