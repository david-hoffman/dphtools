"""S3/S4/S5 actual installed-helper invocation and honest child measurement."""

from pathlib import Path
import sys
import re

import coverage
import pytest

from .release_probe_measurement_support import executable_identity
from .test_release_smoke import PROCESS_DRIVER, Smoke, real_bundle
from .test_release_support import snapshot
from .test_release_version import ROOT, diagnostic, worker


def require(condition, message):
    if not condition:
        print(message, file=sys.stderr)
    assert condition, message


def test_declared_actual_probe_file_is_present_in_opaque_worker(worker):
    helper = worker[0].parent / "release_probe.py"
    require(helper.is_file(), "Missing declared public installed probe: tools/release_probe.py")


def assert_actual_measurement(smoke, before):
    records = smoke.calls()
    installed = [r for r in records if r["event"] == "installed-check"]
    require(len(installed) == 2, "Both retained artifacts must install for measurement")
    require(
        {r["artifact"] for r in installed} == set(smoke.config["artifacts"]),
        "Measure both real retained artifacts",
    )
    require(
        len({r["environment"] for r in installed}) == 2, "Use two real disposable interpreters"
    )
    probes = [r for r in records if r["event"] == "actual-probe"]
    for item in installed:
        prefix = Path(item["environment"]).resolve()
        require(item["returncode"] == 0, "Independent real installed check failed")
        result = item["result"]
        require(
            result["metadata_version"] == result["package_version"] == "1.0.0",
            "Installed versions must match",
        )
        require(
            result["sum"] == 6 and result["backend"].lower() == "agg",
            "Approved public sum/backend invariant",
        )
        for key in ("origin", "utils_origin", "metadata_file"):
            require(
                Path(result[key]).resolve().is_relative_to(prefix),
                "Installed origin escaped actual prefix",
            )
        require(
            not prefix.is_relative_to(ROOT.resolve()),
            "Disposable environment must be outside checkout",
        )
        setup = [
            r
            for r in records
            if r["event"] == "coverage-startup"
            and Path(r["python"]).absolute().parent.parent.resolve() == prefix
        ]
        require(
            len(setup) == 1 and setup[0]["returncode"] == 0,
            "Coverage must really start in each clean interpreter",
        )
        require(
            Path(setup[0]["result"]["coverage_origin"]).resolve().is_relative_to(prefix),
            "Coverage must not import parent or checkout",
        )
        require(not prefix.exists(), "Actual disposable environment must be cleaned")
    smoke.assert_source_install_without_git()
    require(snapshot(smoke.bundle.dist) == before, "Retained artifact bytes must stay unchanged")
    require(
        len(probes) == 2,
        "Missing actual file-based installed-helper invocations (real installs/startup/cleanup/retained bytes passed)",
    )
    for item in installed:
        prefix = Path(item["environment"]).resolve()
        invocation = [p for p in probes if Path(p["environment"]).resolve() == prefix]
        require(len(invocation) == 1, "One actual owned helper call per clean environment")
        invocation = invocation[0]
        python = Path(invocation["python"]).absolute()
        require(
            python.parent.parent.resolve() == prefix,
            "Helper must use actual disposable interpreter",
        )
        require(
            Path(invocation["helper"]) == smoke.worker[0].parent / "release_probe.py",
            "Invoke actual opaque copied helper",
        )
        require(
            invocation["version"] == "1.0.0", "Actual helper version argument must match manifest"
        )
        require(
            not Path(invocation["cwd"])
            .resolve()
            .is_relative_to(smoke.worker[0].parent.parent.resolve()),
            "Actual helper runs outside opaque checkout",
        )
        finishes = [
            r
            for r in records
            if r["event"] == "finish"
            and r.get("actual_probe")
            and r["environment"] == item["environment"]
        ]
        require(
            len(finishes) == 1 and finishes[0]["returncode"] == 0,
            "Actual helper status must succeed",
        )
    helper = (smoke.worker[0].parent / "release_probe.py").resolve()
    for invocation in probes:
        started = [
            r
            for r in records
            if r["event"] == "actual-probe-started"
            and r["environment"] == invocation["environment"]
        ]
        require(len(started) == 1, "Record actual owned helper process identity")
        setup = next(
            r
            for r in records
            if r["event"] == "coverage-startup" and r["python"] == invocation["python"]
        )
        identities = [
            r
            for r in records
            if r["event"] == "actual-probe-identity"
            and r["environment"] == invocation["environment"]
        ]
        require(len(identities) == 1, "One interpreter identity for successful helper")
        identity = identities[0]
        require(identity["launcher_pid"] == started[0]["pid"], "Associate real launcher status")
        require(identity["pid"] > 0, "Record actual interpreter PID")
        require(
            identity["nonce"] == invocation["identity_nonce"], "Associate unique helper invocation"
        )
        require(
            identity["argv"] == [invocation["helper"], invocation["version"]],
            "Associate helper arguments",
        )
        require(
            Path(identity["prefix"]).resolve() == Path(invocation["environment"]).resolve(),
            "Associate interpreter prefix",
        )
        require(
            executable_identity(identity["python"]) == executable_identity(invocation["python"]),
            "Associate interpreter executable",
        )
        filename = Path(identity["data_file"])
        base = Path(setup["data_base"])
        require(
            filename.parent == base.parent and Path(identity["data_base"]) == base,
            "Keep helper data in ordinary family",
        )
        files = [filename] if filename.is_file() else []
        require(bool(files), "Actual helper process coverage must survive cleanup")
        measured = False
        for filename in files:
            data = coverage.CoverageData(basename=str(filename))
            data.read()
            # Filename/nonempty raw execution only; never render lines or arcs.
            for name in data.measured_files():
                path = Path(name)
                if path.resolve() == helper or (
                    not path.is_absolute() and path.parts[-2:] == ("tools", "release_probe.py")
                ):
                    measured = measured or bool(data.lines(name))
        require(measured, "No ordinary execution record for actual owned helper process")


def test_actual_probe_measurement_and_real_failure_controls(worker, tmp_path, real_package):
    smoke = Smoke(worker, real_bundle(tmp_path / "measured bundle", real_package))
    smoke.measure()
    smoke.config["probe_controls"] = True
    keys = [
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "TWINE_USERNAME",
        "TWINE_PASSWORD",
        "PYPI_TOKEN",
        "TESTPYPI_TOKEN",
    ]
    smoke.config["credential_keys"] = keys
    # Dummy values enter after ambient removal. Observe before real child launch.
    injection = (
        "import os\nos.environ.update(" + repr(dict.fromkeys(keys, "a5-dummy-sentinel")) + ")\n"
    )
    smoke.driver.write_text(injection + PROCESS_DRIVER, encoding="utf-8")
    before = snapshot(smoke.bundle.dist)
    result = smoke.run()
    code = result.returncode
    require(code == 0, diagnostic(result))
    assert_actual_measurement(smoke, before)
    # Reject mismatched observed association after genuine success. These are
    # fixture validation controls, never replacement process/coverage results.
    import copy

    records = smoke.calls()
    for field, wrong in (
        ("nonce", "wrong-invocation"),
        ("argv", ["wrong-helper", "999.0.0"]),
        ("prefix", str(tmp_path / "wrong-prefix")),
        ("python", str(tmp_path / "wrong-python")),
        ("launcher_pid", -1),
        ("data_file", str(tmp_path / "unassociated-data")),
    ):
        altered = copy.deepcopy(records)
        next(r for r in altered if r["event"] == "actual-probe-identity")[field] = wrong
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(smoke, "calls", lambda: altered)
            with pytest.raises(AssertionError):
                assert_actual_measurement(smoke, before)
    starts = [r for r in smoke.calls() if r["event"] == "start"]
    for stage in ("install", "check", "probe"):
        observed = [r for r in starts if r["stage"] == stage]
        require(
            bool(observed) and all(not r["credential_names"] for r in observed),
            "Inherited dummy credentials reached real child",
        )
    controls = [r for r in smoke.calls() if r["event"] == "helper-control"]
    require(len(controls) == 4, "Each real helper needs wrong-version and bad-state controls")
    for python in {r["python"] for r in controls}:
        cases = [r for r in controls if r["python"] == python]
        require(
            {r["control"] for r in cases} == {"wrong-version", "bad-installed-state"},
            "Both public rejection controls are required",
        )
        require(
            all(r["returncode"] != 0 and r["diagnostic"] for r in cases),
            "Real helper rejection must retain status and useful diagnostic",
        )
    require(
        not any(r.get("injected") for r in smoke.calls()),
        "Measurement must not substitute process results",
    )


def test_measurement_settings_select_existing_config_and_locked_tool(tmp_path):
    """Fixture self-check: configuration identity/tool input, no product oracle."""
    from .release_probe_measurement_support import settings

    selected = settings(tmp_path, ROOT)
    require(Path(selected["config"]).is_file(), "Existing ordinary coverage config is required")
    requirement = Path(selected["requirement"]).read_text()
    require(
        bool(re.match(r"coverage(?:\[[^\]]+\])?==", requirement))
        and "--hash=sha256:" in requirement,
        "Select only locked coverage provisioning",
    )
    require(
        selected["version"] == coverage.__version__,
        "Provision existing locked coverage tool version",
    )


def test_measurement_setup_keeps_real_provisioning_and_startup_failures_visible(tmp_path):
    """Fixture control only: ordinary tool/venv failures, no owned helper substitute."""
    import json
    import os
    import shutil
    import subprocess

    from .release_probe_measurement_support import provision, run, settings
    from .test_release_smoke import clean_env

    selected = settings(tmp_path, ROOT)
    root = tmp_path / "tool-only clean environment"
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = clean_env()
    records = []
    try:
        code, _, message = run(
            subprocess.Popen,
            [sys.executable, "-m", "venv", str(root)],
            tmp_path,
            env,
        )
        require(code == 0, "Real tool-control environment failed: " + " | ".join(message))
        values = provision(subprocess.Popen, python, env, tmp_path, selected, records.append)
        require(
            all(r["returncode"] == 0 for r in records),
            "Locked coverage must provision/start before negative fixture controls",
        )
        if os.environ.get("COVERAGE_PROCESS_CONFIG"):
            # A child with no explicit fixture config must still inherit ordinary
            # parent/subprocess measurement; the fixture must not turn it off.
            inherited = dict(env)
            inherited.pop("COVERAGE_PROCESS_START", None)
            code, _, message = run(
                subprocess.Popen,
                [
                    str(python),
                    "-c",
                    "import coverage; assert coverage.Coverage.current() is not None",
                ],
                tmp_path,
                inherited,
            )
            require(
                code == 0,
                "Ordinary inherited parent measurement failed: " + " | ".join(message),
            )
        # A real forwarding launcher exercises a different actual interpreter
        # PID on every host, without inventing data or substituting results.
        receipt = tmp_path / "forwarded-identity.json"
        child = tmp_path / "external-identity-control.py"
        child.write_text("import os; print(os.getpid())\n", encoding="utf-8")
        forwarded_env = dict(
            env,
            **values,
            RELEASE_FIXTURE_IDENTITY=str(receipt),
            RELEASE_FIXTURE_NONCE="forwarding-control"
        )
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))",
                str(python),
                str(child),
            ],
            cwd=tmp_path,
            env=forwarded_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = launcher.communicate(timeout=120)
        require(launcher.returncode == 0, "Real forwarding identity control failed")
        identity = json.loads(receipt.read_text(encoding="utf-8"))
        require(
            identity["pid"] == int(stdout) and identity["pid"] != launcher.pid,
            "Observe actual interpreter rather than forwarding launcher",
        )
        require(
            identity["argv"] == [str(child)] and identity["nonce"] == "forwarding-control",
            "Bind forwarded interpreter to exact control invocation",
        )
        require(Path(identity["data_file"]).is_file(), "Retain real forwarded ordinary data")
        missing = tmp_path / "absent-coverage-config"
        code, _, message = run(
            subprocess.Popen,
            [str(python), "-c", "pass"],
            tmp_path,
            dict(env, **{**values, "COVERAGE_PROCESS_START": str(missing)}),
        )
        require(code != 0 and bool(message), "Failed real startup must not continue unmeasured")
        records.clear()
        bad = {**selected, "requirement": str(tmp_path / "absent-tool-requirement")}
        try:
            provision(subprocess.Popen, python, env, tmp_path, bad, records.append)
        except AssertionError as error:
            require(bool(str(error)), "Provisioning failure must retain a diagnostic")
        else:
            require(False, "Real pip provisioning failure was hidden")
        require(
            len(records) == 1 and records[0]["returncode"] != 0 and records[0]["diagnostic"],
            "Retain actual failed tool status and diagnostics",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    require(not root.exists(), "Tool-control environment must be removed")


def test_executable_identity_keeps_directory_and_basename_with_real_controls(tmp_path):
    """Real interpreters sharing a binary must retain distinct selected identities."""
    import json
    import os
    import subprocess

    from .release_probe_measurement_support import run
    from .test_release_smoke import clean_env

    env = clean_env()
    # Tool-only identity proof must not inherit macOS program-name overrides.
    env.pop("PYTHONEXECUTABLE", None)
    env.pop("__PYVENV_LAUNCHER__", None)
    roots = [tmp_path / "selected", tmp_path / "other"]
    relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    for root in roots:
        code, _, message = run(
            subprocess.Popen,
            [sys.executable, "-m", "venv", "--without-pip", str(root)],
            tmp_path,
            env,
        )
        require(code == 0, "Real identity environment failed: " + " | ".join(message))
    selected = roots[0] / relative
    if os.name == "nt":
        alias = selected.parent / ".." / selected.parent.name / selected.name
    else:
        directory = tmp_path / "directory-alias"
        directory.symlink_to(roots[0], target_is_directory=True)
        alias = directory / relative
    probe = (
        "import json, sys, warnings; "
        "warnings.formatwarning=lambda message, category, filename, lineno, line=None: "
        "str(category.__name__)+': '+str(message)+'\\n'; "
        "sys.excepthook=lambda kind,value,tb: print(kind.__name__+': '+str(value),file=sys.stderr); "
        "print(json.dumps({'python':sys.executable,'prefix':sys.prefix}))"
    )
    observations = []
    for python in (selected, alias, roots[1] / relative, Path(sys.executable)):
        code, stdout, message = run(subprocess.Popen, [str(python), "-c", probe], tmp_path, env)
        require(code == 0, "Real identity control failed: " + " | ".join(message))
        observations.append(json.loads(stdout))
    for actual in observations[:2]:
        require(
            Path(actual["prefix"]).resolve() == roots[0].resolve(), "Alias lost selected prefix"
        )
        require(
            executable_identity(actual["python"]) == executable_identity(alias),
            "Directory alias must preserve selected interpreter identity",
        )
    for actual in observations[2:]:
        require(
            executable_identity(actual["python"]) != executable_identity(selected),
            "Other environment or parent interpreter must not match selected executable",
        )
    # A real alternative basename shares the binary but is not the executable
    # selected by this invocation. Windows hard links need no symlink privilege.
    renamed = selected.with_name(
        "python-different-name.exe" if os.name == "nt" else "python-different-name"
    )
    if os.name == "nt":
        os.link(selected, renamed)
    else:
        renamed.symlink_to(selected)
    require(renamed.samefile(selected), "Control must share the binary target")
    code, stdout, message = run(subprocess.Popen, [str(renamed), "-c", probe], tmp_path, env)
    require(code == 0, "Real basename control failed: " + " | ".join(message))
    actual = json.loads(stdout)
    require(
        Path(actual["prefix"]).resolve() == roots[0].resolve(),
        "Alternate basename lost selected prefix",
    )
    require(
        executable_identity(actual["python"]) == executable_identity(renamed),
        "Observe the actually launched alternate basename",
    )
    require(
        executable_identity(actual["python"]) != executable_identity(selected),
        "Same binary with different selected basename must not match",
    )
