"""Fresh exact-artifact installation and complete scoped command contracts."""

from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from . import test_release_smoke as release_smoke
from .test_release_support import write_json
from .test_release_version import diagnostic, invoke

ROOT = Path(__file__).resolve().parents[1]
worker = release_smoke.worker


def installation(worker, tmp_path, real_package, mutation=False):
    bundle = release_smoke.real_bundle(tmp_path / "bundle", real_package)
    observed = release_smoke.Smoke(worker, bundle)
    if mutation:
        # Observe both real successful probes, then change retained bytes before
        # the worker's final identity check. No process result is substituted.
        original = observed.driver.read_text()
        original = original.replace(
            '        if self.stage != "install" or self.returncode or self.package is None:',
            '        if self.actual_probe and self.returncode == 0 and counts["probe"] == 2:\n'
            '            artifact = Path(config["artifacts"][0])\n'
            '            artifact.write_bytes(artifact.read_bytes() + b"changed-after-install")\n'
            '        if self.stage != "install" or self.returncode or self.package is None:',
        )
        observed.driver.write_text(original, encoding="utf-8")
    write_json(observed.config_file, observed.config)
    constraints = bundle.root / "constraints.txt"
    constraints.write_text(
        "".join(
            f"{d.metadata['Name']}=={d.version}\n"
            for d in metadata.distributions()
            if d.metadata["Name"].lower() != "dphtools"
        ),
        encoding="utf-8",
    )
    entry, outside = worker
    result = invoke(
        observed.driver,
        entry.with_name("verification_install.py"),
        observed.config_file,
        observed.log,
        bundle.dist,
        constraints,
        cwd=outside,
        env=release_smoke.clean_env(),
        timeout=600,
    )
    return observed, result


@pytest.mark.parametrize("mutation", [False, True])
def test_exact_artifact_worker_checks_real_clean_installs_and_retained_identity(
    worker, tmp_path, real_package, mutation
):
    observed, result = installation(worker, tmp_path, real_package, mutation)
    records = observed.calls()
    checks = [item for item in records if item["event"] == "installed-check"]
    assert len(checks) == 2 and all(item["returncode"] == 0 for item in checks)
    assert len({item["environment"] for item in checks}) == 2
    observed.assert_source_install_without_git()
    assert all(not Path(item["environment"]).exists() for item in checks)
    if mutation:
        assert result.returncode != 0 and "Artifacts changed during installation" in diagnostic(
            result
        )
    else:
        assert result.returncode == 0, diagnostic(result)
        assert "Exact current wheel and source clean installations passed" in result.stdout


@pytest.mark.parametrize("defect", ["missing-wheel", "ambiguous-metadata"])
def test_invalid_build_artifacts_fail_before_installation(worker, tmp_path, real_package, defect):
    bundle = release_smoke.real_bundle(tmp_path / "bundle", real_package)
    if defect == "missing-wheel":
        bundle.wheel.unlink()
    else:
        with zipfile.ZipFile(bundle.wheel, "a") as archive:
            archive.writestr("other.dist-info/METADATA", "Name: other\nVersion: 1.0\n")
    constraints = bundle.root / "constraints.txt"
    constraints.write_text("", encoding="utf-8")
    result = invoke(
        worker[0].with_name("verification_install.py"),
        bundle.dist,
        constraints,
        cwd=worker[1],
        env=release_smoke.clean_env(),
    )
    assert result.returncode != 0
    assert (
        "Expected one current wheel" if defect == "missing-wheel" else "Ambiguous wheel metadata"
    ) in diagnostic(result)


def test_real_library_command_declares_complete_domain_and_unvalidated_complement(tmp_path):
    report = tmp_path / "library-command"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/verification.py"),
            "library",
            "--report-dir",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checks = json.loads((report / "checks.json").read_text())
    assert checks["scope"]["name"] == "library"
    assert checks["scope"]["coverage_claim"] == "scoped" and checks["coverage_complete"] is True
    assert checks["scope"]["domains"] == ["library"]
    assert "tools/verification.py" in checks["scope"]["unvalidated_sources"]
    assert "tests/test_verification_install.py" in checks["scope"]["unvalidated_test_paths"]
    assert all(step["state"] == "passed" for step in checks["steps"])
    install = next(step for step in checks["steps"] if step["name"] == "clean-install")
    assert install["returncode"] == 0 and install["duration_seconds"] > 0
    assert len(install["input_identity"]["artifacts"]) == 2
    coverage = json.loads((report / "coverage.json").read_text())
    assert set(coverage["files"]) == set(checks["scope"]["measured_sources"])
    assert coverage["totals"]["missing_lines"] == coverage["totals"]["missing_branches"] == 0


def test_importing_install_worker_does_not_create_an_environment_or_install(worker):
    helper = worker[0].with_name("verification_install.py")
    before = set(worker[1].iterdir())
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; sys.path.insert(0,sys.argv[1]); runpy.run_path(sys.argv[2],run_name='import_probe'); print('imported without installation')",
            str(helper.parent),
            str(helper),
        ],
        cwd=worker[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "imported without installation"
    assert set(worker[1].iterdir()) == before
