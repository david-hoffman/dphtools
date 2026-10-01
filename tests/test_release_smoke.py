"""S3/S4/S5: real installed wheel/source behavior and subprocess-failure handling.

The package build uses opaque production sources. The subprocess observer never
substitutes successful install/check/probe results. It performs an additional real
installed-package check before temporary environments can be removed.
"""

from email.parser import BytesParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from .test_release_support import Bundle, DEPENDENCIES, rejected, snapshot, write_json
from . import test_release_version as version_cli
from .test_release_version import ROOT, diagnostic, invoke

worker = version_cli.worker

INSTALLED_PROBE = r"""
from importlib import metadata
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas
import scipy
import skimage
import dphtools
from dphtools import utils

expected = sys.argv[1]
assert metadata.version("dphtools") == expected, "installed metadata version disagrees"
assert dphtools.__version__ == expected, "package version disagrees"
origin = Path(dphtools.__file__).resolve()
prefix = Path(sys.prefix).resolve()
assert origin.is_relative_to(prefix), "import escaped the clean environment"
assert all(Path(module.__file__).resolve().is_relative_to(prefix)
           for module in (np, pandas, scipy, matplotlib, skimage)), "global dependency import"
assert utils.bin_ndarray(np.arange(4).reshape(2, 2), new_shape=(1, 1), operation="sum").item() == 6
print(json.dumps({"metadata_version": metadata.version("dphtools"),
                  "package_version": dphtools.__version__, "origin": str(origin),
                  "utils_origin": str(Path(utils.__file__).resolve()),
                  "metadata_file": str(metadata.distribution("dphtools").locate_file(next(
                      path for path in metadata.distribution("dphtools").files
                      if str(path).endswith(".dist-info/METADATA")))),
                  "prefix": str(prefix), "sum": 6, "backend": matplotlib.get_backend()}))
"""

GIT_PROBE = "import json, shutil; print(json.dumps({'git': shutil.which('git')}))"

PROCESS_DRIVER = r"""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import warnings
import uuid

entry, config_file, log_file, *args = sys.argv[1:]
config = json.loads(Path(config_file).read_text(encoding="utf-8"))
real_popen = subprocess.Popen
counts = {}
measurement_envs = {}
installed_results = {}
measurement_support = (runpy.run_path(config["measurement_support"])
                       if config.get("measurement_support") else None)
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{category.__name__}: {message}\n")
sys.excepthook = lambda kind, value, traceback: print(f"{kind.__name__}: {value}", file=sys.stderr)

def record(item):
    with Path(log_file).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item) + "\n")

class ObservedPopen(real_popen):
    def __init__(self, argv, *args, **kwargs):
        assert isinstance(argv, (list, tuple)), "Smoke subprocess must use literal argument lists"
        assert not kwargs.get("shell", False), "Smoke subprocess must not expand artifact paths in a shell"
        argv = [str(arg) for arg in argv]
        publisher = Path(argv[0]).stem.lower()
        if publisher in ("gh", "twine", "anaconda", "conda") or "twine" in argv:
            assert not any(word in argv for word in ("upload", "publish", "approve", "POST", "PUT", "PATCH", "DELETE")), (
                "Diagnostic workers cannot approve or publish")
            assert not (publisher == "gh" and "release" in argv and "create" in argv), (
                "Diagnostic worker cannot finalize a GitHub Release")
        stage = "other"
        if "pip" in argv or Path(argv[0]).name.lower().startswith("pip"):
            if "install" in argv:
                stage = "install"
            elif "check" in argv:
                stage = "check"
        elif "venv" in argv or "ensurepip" in argv:
            stage = "environment"
        elif Path(argv[0]).name.lower().startswith("python"):
            stage = "probe"
        # Resolve the ENVIRONMENT PATH, not a POSIX interpreter symlink's target.
        parent = Path(argv[0]).absolute().parent
        self.environment = str(parent.parent) if parent.name.lower() in ("bin", "scripts") else None
        self.stage = stage
        self.actual_probe = (len(argv) == 3 and Path(argv[1]).name == "release_probe.py")
        self.actual_argv = argv if self.actual_probe else None
        if self.environment in measurement_envs:
            kwargs["env"] = dict(os.environ if kwargs.get("env") is None else kwargs["env"],
                                 **measurement_envs[self.environment])
        self.identity_file = None
        if self.actual_probe and self.environment in measurement_envs:
            self.identity_nonce = uuid.uuid4().hex
            self.identity_file = Path(log_file).parent / ("helper-identity-" + self.identity_nonce + ".json")
            kwargs["env"].update(RELEASE_FIXTURE_IDENTITY=str(self.identity_file),
                                 RELEASE_FIXTURE_NONCE=self.identity_nonce)
        self.env_values = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        if self.actual_probe:
            assert Path(argv[1]).is_absolute(), "Actual helper invocation requires an absolute path"
            record({"event": "actual-probe", "python": argv[0], "helper": argv[1],
                    "version": argv[2], "environment": self.environment,
                    "cwd": str(kwargs.get("cwd") or Path.cwd()),
                    "identity_nonce": getattr(self, "identity_nonce", None)})
        self.cwd_value = str(kwargs.get("cwd") or Path.cwd())
        self.package = next((arg for arg in argv if arg in config["artifacts"]), None)
        counts[stage] = counts.get(stage, 0) + 1
        self.injected = stage == config.get("fail_stage") and counts[stage] == config.get("fail_index", 1)
        record({"event": "start", "stage": stage, "environment": self.environment,
                "cwd": self.cwd_value, "artifact": self.package, "injected": self.injected,
                "credential_names": [key for key in config.get("credential_keys", [])
                                     if key in self.env_values]})
        self.observed = False
        if stage == "install" and self.package and self.package.endswith(".tar.gz"):
            # Observe lookup in the installer's actual environment/cwd. Do not
            # remove Git, alter the installer, or substitute an install result.
            lookup = real_popen([sys.executable, "-c", config["git_probe"]],
                                cwd=self.cwd_value, env=self.env_values,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = lookup.communicate(timeout=30)
            assert lookup.returncode == 0, "Git availability observer failed"
            record({"event": "source-install-git", "artifact": self.package,
                    "git": json.loads(stdout.decode("utf-8"))["git"]})
        if self.injected:
            argv = [sys.executable, "-c", "import sys; print('RELEASE-FIXTURE-PROCESS-FAILURE', file=sys.stderr); sys.exit(23)"]
        super().__init__(argv, *args, **kwargs)
        if self.actual_probe:
            record({"event": "actual-probe-started", "pid": self.pid,
                    "environment": self.environment})

    def communicate(self, *args, **kwargs):
        result = super().communicate(*args, **kwargs)
        self.after_completion()
        return result

    def wait(self, *args, **kwargs):
        result = super().wait(*args, **kwargs)
        self.after_completion()
        return result

    def after_completion(self):
        if self.returncode is None or self.observed:
            return
        self.observed = True
        record({"event": "finish", "stage": self.stage, "returncode": self.returncode,
                "injected": self.injected, "actual_probe": self.actual_probe,
                "environment": self.environment})
        if self.identity_file is not None and self.returncode == 0:
            identity = json.loads(self.identity_file.read_text(encoding="utf-8"))
            assert identity["nonce"] == self.identity_nonce, "Wrong actual helper identity nonce"
            assert identity["argv"] == self.actual_argv[1:], "Wrong actual helper invocation"
            assert Path(identity["prefix"]).resolve() == Path(self.environment).resolve(), "Wrong actual interpreter prefix"
            assert Path(identity["python"]).absolute() == Path(self.actual_argv[0]).absolute(), "Wrong actual interpreter executable"
            record({"event": "actual-probe-identity", "launcher_pid": self.pid,
                    "environment": self.environment, **identity})
        if self.actual_probe and self.returncode == 0 and config.get("probe_controls"):
            measurement_support["controls"](real_popen, self.actual_argv, self.cwd_value,
                                            self.env_values, installed_results[self.environment], record)
        if self.stage != "install" or self.returncode or self.package is None:
            return
        root = Path(self.environment)
        cfg = root / "pyvenv.cfg"
        if cfg.is_file():
            settings = dict((key.strip(), value.strip()) for key, value in
                            (line.lower().split("=", 1) for line in cfg.read_text(encoding="utf-8").splitlines() if "=" in line))
            assert settings["include-system-site-packages"] == "false", "Environment inherited global packages"
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = dict(self.env_values, MPLBACKEND="Agg")
        probe = real_popen([str(python), "-c", config["probe"], config["version"]],
                           cwd=config["outside"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = probe.communicate(timeout=120)
        item = {"event": "installed-check", "artifact": self.package,
                "environment": str(root), "returncode": probe.returncode}
        if probe.returncode == 0:
            item["result"] = json.loads(stdout.decode("utf-8").splitlines()[-1])
        else:
            # Retain the exception message; never emit installed-package source.
            item["diagnostic"] = stderr.decode("utf-8", errors="replace").splitlines()[-1:]
        record(item)
        if item["returncode"] == 0:
            installed_results[str(root)] = item["result"]
        if config.get("measurement"):
            measurement_envs[str(root)] = measurement_support["provision"](
                real_popen, python, self.env_values, config["outside"], config["measurement"], record)
        if item["returncode"] == 0 and config.get("installed_fault"):
            # Inject ONLY an external installed-package defect after a real,
            # independently verified successful installation. Owned release code
            # and retained artifact bytes remain unchanged. Positive smoke paths
            # always exercise unmodified production package behavior.
            fault = config["installed_fault"]
            if fault == "version":
                target = Path(item["result"]["origin"])
                suffix = "\n__version__ = '999.0.0'\n"
            elif fault == "sum":
                target = Path(item["result"]["utils_origin"])
                suffix = "\ndef bin_ndarray(*args, **kwargs):\n    return __import__('numpy').array([[7]])\n"
            elif fault == "dependency":
                target = Path(item["result"]["metadata_file"])
                content = target.read_bytes()
                delimiter = b"\r\n\r\n" if b"\r\n\r\n" in content else b"\n\n"
                header, separator, body = content.partition(delimiter)
                assert separator, "Installed metadata fixture needs a header/body separator"
                newline = delimiter[:len(delimiter) // 2]
                target.write_bytes(header + newline + b"Requires-Dist: dphtools-release-fixture-absent" + delimiter + body)
                suffix = ""
            else:
                raise AssertionError("Unknown installed-package fixture fault")
            with target.open("a", encoding="utf-8") as stream:
                stream.write(suffix)
            record({"event": "installed-fault", "fault": fault, "environment": str(root)})

subprocess.Popen = ObservedPopen
sys.argv = [entry, *args]
sys.path[0] = str(Path(entry).resolve().parent)
runpy.run_path(entry, run_name="__main__")
"""


def clean_env():
    env = dict(os.environ)
    for key in list(env):
        if any(
            word in key.upper() for word in ("TOKEN", "PASSWORD", "CREDENTIAL", "SECRET")
        ) or key.upper().startswith(("TWINE_", "PYPI_", "TESTPYPI_", "GH_", "GITHUB_", "AWS_")):
            env.pop(key)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    env.pop("PIP_EXTRA_INDEX_URL", None)
    return env


@pytest.fixture(scope="module")
def real_package(tmp_path_factory):
    """Build opaque actual sources with isolated fixture Git data, never owner history."""
    root = tmp_path_factory.mktemp("real release package")
    source = root / "source copy"
    source.mkdir()
    shutil.copytree(
        ROOT / "dphtools", source / "dphtools", ignore=shutil.ignore_patterns("__pycache__")
    )
    # Standard packaging inputs are opaque copies. No task, tool, workflow,
    # history, or credential directory is needed in this isolated build fixture.
    for pattern in (
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "MANIFEST.in",
        "versioneer.py",
        "README*",
        "LICENSE*",
        "requirements*.txt",
    ):
        for path in ROOT.glob(pattern):
            if path.is_file():
                shutil.copy2(path, source / path.name)
    dist = root / "built distributions"
    env = clean_env()
    env.update(
        SETUPTOOLS_SCM_PRETEND_VERSION="1.0.0", SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DPHTOOLS="1.0.0"
    )
    git = shutil.which("git")
    assert git is not None, "Opaque real-package fixture requires installed Git"
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    template = root / "empty git template"
    template.mkdir()
    for command in (
        [git, "init", "--quiet", "--template", str(template)],
        [git, "add", "--all"],
        [
            git,
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Release Fixture",
            "-c",
            "user.email=release-fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Opaque fixture",
        ],
        [git, "tag", "1.0.0"],
        [git, "tag", "v1.0.0"],
    ):
        result = subprocess.run(
            command,
            cwd=source,
            env=env,
            input="",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, "Fixture Git setup failed: " + diagnostic(result)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist),
        ],
        cwd=source,
        env=env,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, "Opaque real-package build fixture failed: " + diagnostic(
        result
    )
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Build fixture must contain one wheel and one sdist"
    return wheels[0], sdists[0]


def real_bundle(root, real_package):
    bundle = Bundle(root)
    bundle.wheel.unlink()
    bundle.sdist.unlink()
    wheel, sdist = real_package
    bundle.wheel = bundle.dist / wheel.name
    bundle.sdist = bundle.dist / sdist.name
    shutil.copy2(wheel, bundle.wheel)
    shutil.copy2(sdist, bundle.sdist)
    bundle.refresh_manifest()
    return bundle


class Smoke:
    def __init__(self, worker, bundle):
        self.worker = worker
        self.bundle = bundle
        self.driver = bundle.root / "process-boundary.py"
        self.driver.write_text(PROCESS_DRIVER, encoding="utf-8")
        self.log = bundle.root / "process.jsonl"
        self.config_file = bundle.root / "process-config.json"
        self.config = {
            "artifacts": [str(bundle.wheel), str(bundle.sdist)],
            "version": bundle.version,
            "outside": str(worker[1]),
            "probe": INSTALLED_PROBE,
            "git_probe": GIT_PROBE,
        }
        # Canonical instrumented runs must measure actual disposable children too.
        # Uninstrumented fixture self-checks still delegate their original processes.
        import coverage

        if coverage.Coverage.current() is not None or os.environ.get("COVERAGE_PROCESS_START"):
            self.measure()

    def measure(self):
        from .release_probe_measurement_support import settings

        support = self.bundle.root / "probe-measurement-support.py"
        shutil.copy2(Path(__file__).with_name("release_probe_measurement_support.py"), support)
        self.config.update(
            measurement=settings(self.bundle.root, ROOT), measurement_support=str(support)
        )

    def run(self):
        write_json(self.config_file, self.config)
        entry, outside = self.worker
        return invoke(
            self.driver,
            entry,
            self.config_file,
            self.log,
            "smoke",
            "--manifest",
            self.bundle.manifest,
            "--dist",
            self.bundle.dist,
            cwd=outside,
            env=clean_env(),
            timeout=600,
        )

    def assert_source_install_without_git(self):
        observations = [item for item in self.calls() if item["event"] == "source-install-git"]
        assert len(observations) == 1, "Observe the real source-distribution install"
        assert observations[0]["artifact"] == str(self.bundle.sdist)
        assert (
            observations[0]["git"] is None
        ), "Git executable was available to the source installer"

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def smoke(worker, tmp_path, real_package):
    return Smoke(worker, real_bundle(tmp_path / "bundle", real_package))


def test_real_package_fixture_has_declared_metadata_and_no_git_dependency(real_package):
    """Fixture self-check: metadata only; no runtime source is inspected."""
    wheel, sdist = real_package
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(names) == 1
        wheel_metadata = BytesParser().parsebytes(archive.read(names[0]))
    with tarfile.open(sdist) as archive:
        names = [
            name
            for name in archive.getnames()
            if name.count("/") == 1 and name.endswith("/PKG-INFO")
        ]
        assert len(names) == 1
        sdist_metadata = BytesParser().parsebytes(archive.extractfile(names[0]).read())
        assert not any(".git" in Path(name).parts for name in archive.getnames())
    for package_metadata in (wheel_metadata, sdist_metadata):
        assert package_metadata["Name"] == "dphtools"
        assert package_metadata["Version"] == "1.0.0"
        assert package_metadata["Requires-Python"] == ">=3.8"
        assert sorted(package_metadata.get_all("Requires-Dist")) == sorted(DEPENDENCIES)


def test_smoke_installs_both_real_artifacts_in_separate_clean_environments(smoke):
    before = snapshot(smoke.bundle.dist)
    result = smoke.run()
    assert result.returncode == 0, diagnostic(result)
    smoke.assert_source_install_without_git()
    records = smoke.calls()
    installed = [item for item in records if item["event"] == "installed-check"]
    assert {item["artifact"] for item in installed} == set(smoke.config["artifacts"])
    assert len(installed) == 2
    assert len({item["environment"] for item in installed}) == 2
    checkout = smoke.worker[0].parent.parent.resolve()
    for item in installed:
        assert (
            item["returncode"] == 0
        ), "Installed-package invariant failed (source-free diagnostic recorded)"
        observed = item["result"]
        assert observed["metadata_version"] == observed["package_version"] == "1.0.0"
        assert observed["sum"] == 6
        assert observed["backend"].lower() == "agg"
        assert not Path(item["environment"]).resolve().is_relative_to(checkout)
        assert not Path(item["environment"]).exists(), "Disposable environment was not cleaned up"
    assert snapshot(smoke.bundle.dist) == before


@pytest.mark.parametrize("stage", ["install", "check", "probe"])
def test_smoke_external_process_or_assertion_failure_blocks_readiness(smoke, stage):
    smoke.config.update(fail_stage=stage, fail_index=1)
    before = snapshot(smoke.bundle.dist)
    result = smoke.run()
    rejected(result)
    assert any(
        call.get("injected") for call in smoke.calls()
    ), "Failure injection must reach its intended boundary"
    assert "RELEASE-FIXTURE-PROCESS-FAILURE" in result.stdout + result.stderr, diagnostic(result)
    assert snapshot(smoke.bundle.dist) == before


@pytest.mark.parametrize("fault", ["version", "sum", "dependency"])
def test_smoke_detects_actual_installed_version_behavior_and_dependency_defects(smoke, fault):
    smoke.config["installed_fault"] = fault
    before = snapshot(smoke.bundle.dist)
    result = smoke.run()
    rejected(result)
    assert any(
        call.get("event") == "installed-fault" for call in smoke.calls()
    ), "The intended installed-package defect must be reached before accepting failure"
    assert not any(call.get("injected") for call in smoke.calls()), "No fake process failures"
    assert snapshot(smoke.bundle.dist) == before


def test_smoke_validates_retained_identity_before_starting_installers(smoke):
    # Establish this public command before the early-failure variant.
    good = smoke.run()
    assert good.returncode == 0, diagnostic(good)
    smoke.log.unlink()
    smoke.bundle.wheel.write_bytes(b"changed after manifest")
    rejected(smoke.run())
    assert smoke.calls() == []


def test_partial_publication_uses_original_bytes_then_checks_downloaded_installs(
    worker, tmp_path, real_package
):
    """S5 command chain; hosted approval and GitHub finalization remain independent."""
    from .test_release_reconcile import Registry

    bundle = real_bundle(tmp_path / "recovery bundle", real_package)
    registry = Registry(worker, bundle)
    before = snapshot(bundle.dist)
    registry.publish([bundle.wheel.name])
    result = registry.run()
    assert result.returncode == 0, diagnostic(result)
    assert snapshot(registry.output) == {bundle.sdist.name: before[bundle.sdist.name]}
    # This updates only the controlled external registry fixture, never a service.
    registry.publish([bundle.wheel.name, bundle.sdist.name])
    shutil.rmtree(registry.output)
    registry.output.mkdir()
    result = registry.run("--require-complete")
    assert result.returncode == 0, diagnostic(result)
    assert snapshot(registry.output) == {}
    downloaded = bundle.root / "downloaded registry files"
    downloaded.mkdir()
    downloader = bundle.root / "external-download.py"
    downloader.write_text(
        "import json, sys, urllib.request\n"
        "from pathlib import Path\n"
        "for filename, url in json.loads(sys.argv[1]).items():\n"
        "    with urllib.request.urlopen(url) as response:\n"
        "        (Path(sys.argv[2]) / filename).write_bytes(response.read())\n",
        encoding="utf-8",
    )
    write_json(registry.config_file, registry.config)
    download = invoke(
        registry.driver,
        downloader,
        registry.config_file,
        registry.log,
        json.dumps(registry.urls),
        downloaded,
        cwd=worker[1],
        env=clean_env(),
    )
    assert download.returncode == 0, diagnostic(download)
    assert snapshot(downloaded) == before
    bundle.dist = downloaded
    bundle.wheel = downloaded / bundle.wheel.name
    bundle.sdist = downloaded / bundle.sdist.name
    # Keep the ORIGINAL manifest unchanged; downloaded bytes must satisfy it.
    smoke = Smoke(worker, bundle)
    result = smoke.run()
    assert result.returncode == 0, diagnostic(result)
    smoke.assert_source_install_without_git()
    checks = [item for item in smoke.calls() if item["event"] == "installed-check"]
    assert len(checks) == 2 and all(item["returncode"] == 0 for item in checks)
    assert snapshot(downloaded) == before


def test_process_fixture_retains_real_exit_status_and_does_not_fake_success(tmp_path):
    """Self-check only: standard subprocess observer and controlled failure injection."""
    driver = tmp_path / "driver.py"
    driver.write_text(PROCESS_DRIVER, encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import subprocess, sys\n"
        "result = subprocess.run([sys.executable, '-c', 'raise SystemExit(0)'], capture_output=True)\n"
        "assert result.returncode == 23\n"
        "assert b'RELEASE-FIXTURE-PROCESS-FAILURE' in result.stderr\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    write_json(config, {"artifacts": [], "fail_stage": "probe"})
    log = tmp_path / "calls.jsonl"
    result = invoke(driver, probe, config, log, cwd=tmp_path)
    assert result.returncode == 0, diagnostic(result)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(item.get("injected") for item in records)
    # Self-check Git lookup without changing any product installer environment.
    git = shutil.which("git")
    assert git is not None, "Fixture self-check requires installed Git"
    lookup = tmp_path / "git-lookup.py"
    lookup.write_text(GIT_PROBE, encoding="utf-8")
    for path, available in (("", False), (str(Path(git).parent), True)):
        result = invoke(lookup, cwd=tmp_path, env=dict(clean_env(), PATH=path))
        assert result.returncode == 0, diagnostic(result)
        assert (json.loads(result.stdout)["git"] is not None) is available


def test_installer_observer_uses_a_real_clean_environment_and_package_metadata(
    tmp_path, real_package
):
    """Self-check only: observer install branch; no fake successful release behavior."""
    wheel, _ = real_package
    driver = tmp_path / "driver.py"
    driver.write_text(PROCESS_DRIVER, encoding="utf-8")
    env_dir = tmp_path / "clean fixture environment"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = tmp_path / "external-installer.py"
    probe.write_text(
        "import shutil, subprocess, sys, venv\n"
        "try:\n"
        "    venv.EnvBuilder(with_pip=True).create(sys.argv[1])\n"
        "    subprocess.run([sys.argv[2], '-m', 'pip', 'install', '--no-index', '--no-deps', sys.argv[3]], check=True, capture_output=True)\n"
        "finally:\n"
        "    shutil.rmtree(sys.argv[1], ignore_errors=True)\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    # This fixture check measures only metadata transport. The product smoke
    # tests above always use INSTALLED_PROBE and real unmodified package imports.
    write_json(
        config,
        {
            "artifacts": [str(wheel)],
            "version": "1.0.0",
            "outside": str(tmp_path),
            "probe": "import json; from importlib import metadata; print(json.dumps({'version': metadata.version('dphtools')}))",
        },
    )
    log = tmp_path / "calls.jsonl"
    result = invoke(
        driver, probe, config, log, env_dir, python, wheel, cwd=tmp_path, env=clean_env()
    )
    assert result.returncode == 0, diagnostic(result)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    checks = [record for record in records if record["event"] == "installed-check"]
    assert len(checks) == 1
    assert checks[0]["returncode"] == 0
    assert checks[0]["result"] == {"version": "1.0.0"}
    assert not env_dir.exists()
