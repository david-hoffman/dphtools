"""Exercise optional reuse through the actual verifier and installed check tools."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import venv

import pytest

ROOT = Path(__file__).resolve().parents[1]


def seal(record):
    """Reseal deliberately valid metadata to test its declared semantics."""
    record.pop("receipt_digest", None)
    record["receipt_digest"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")
    ).hexdigest()


@pytest.fixture
def reuse_command(tmp_path):
    root = tmp_path / "repository"
    shutil.copytree(ROOT / "tools", root / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "setup.cfg", "requirements-dev.lock"):
        shutil.copy2(ROOT / name, root / name)
    (root / "dphtools").mkdir()
    (root / "dphtools/__init__.py").write_text('"""An isolated package."""\n\nVALUE = 1\n')
    (root / "tests").mkdir()
    (root / "notebooks").mkdir()
    for name in ("setup.py", "versioneer.py"):
        (root / name).write_text('"""An inert fixture."""\n')
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    def run(receipt=None):
        args = [sys.executable, str(root / "tools/verification.py"), "fast"]
        if receipt is not None:
            args.extend(["--reuse", str(receipt)])
        result = subprocess.run(
            args, cwd=root, env=env, text=True, capture_output=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
        paths = list((root / "reports/verification").glob("*/checks.json"))
        path = next(path for path in paths if str(path.parent) in result.stdout)
        return path, json.loads(path.read_text())

    return root, env, run


def test_unchanged_check_reuses_original_command_and_retained_log(reuse_command):
    _, _, run = reuse_command
    original, source = run()
    reused_path, reused = run(original)
    step = next(step for step in reused["steps"] if step["name"] == "docstrings")
    assert step["state"] == "reused"
    assert step["provenance"]["receipt"] == str(original)
    assert step["provenance"]["command"] == source["steps"][-1]["command"]
    assert step["provenance"]["receipt_sha256"]
    assert step["returncode"] is None
    assert reused["complete"] is True and reused["outcome"] == "passed"
    assert [step["state"] for step in reused["steps"][:2]] == ["passed", "passed"]
    # A reused receipt is not presented as a new execution or recursively trusted.
    _, next_run = run(reused_path)
    assert next_run["steps"][-1]["state"] == "passed"


@pytest.mark.parametrize(
    "fault", ["source", "log", "command", "partial", "failed", "identity", "seal", "version"]
)
def test_changed_or_unverifiable_evidence_runs_check_again(reuse_command, fault):
    root, _, run = reuse_command
    original, record = run()
    if fault == "source":
        (root / "dphtools/__init__.py").write_text('"""A changed package."""\n\nVALUE = 2\n')
    elif fault == "log":
        (original.parent / record["steps"][-1]["log"]["path"]).write_text("tampered")
    elif fault == "command":
        record["steps"][-1]["command"].append("--help")
    elif fault == "partial":
        record["complete"] = False
    elif fault == "failed":
        record["steps"][0]["state"] = "failed"
        record["outcome"] = "failed"
    elif fault == "identity":
        record["identity_digest"] = "foreign"
    elif fault == "version":
        record["document_version"] = "foreign"
    else:
        record["duration_seconds"] = -1
    if fault not in ("source", "log"):
        if fault != "seal":
            seal(record)
        original.write_text(json.dumps(record))
    _, fresh = run(original)
    assert fresh["steps"][-1]["state"] == "passed"
    assert fresh["steps"][-1]["reuse_rejection"]


def test_reuse_is_never_a_full_gate_or_an_uncontrolled_import_proof(reuse_command):
    root, env, run = reuse_command
    original, _ = run()
    env["PYTHONPATH"] = str(root / "tools")
    _, fresh = run(original)
    assert fresh["steps"][-1]["state"] == "passed"
    result = subprocess.run(
        [sys.executable, str(root / "tools/verification.py"), "full", "--reuse", str(original)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "fast" in result.stderr


@pytest.mark.parametrize("unknown", ["import-path", "root-module", "root-package", "user-site"])
def test_unknown_import_inputs_decline_even_identical_receipts(reuse_command, unknown):
    root, env, run = reuse_command
    if unknown == "import-path":
        env["PYTHONPATH"] = str(root / "tools")
    elif unknown == "root-module":
        (root / "local.py").write_text('"""An untracked import provider."""\n')
    elif unknown == "root-package":
        package = root / "pydocstyle"
        package.mkdir()
        (package / "__main__.py").write_text("raise SystemExit(0)\n")
    else:
        env["PYTHONUSERBASE"] = str(root / "private-user-base")
        result = subprocess.run(
            [sys.executable, "-c", "import site; print(site.getusersitepackages())"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        Path(result.stdout.strip()).mkdir(parents=True)
    original, source = run()
    assert source["steps"][-1]["input_identity"]["runtime"] is None
    _, fresh = run(original)
    assert fresh["steps"][-1]["state"] == "passed"
    assert "cannot be completely identified" in fresh["steps"][-1]["reuse_rejection"]


@pytest.mark.parametrize(
    "unknown", ["external-symlink", "customization-package", "customization-file"]
)
def test_runtime_identity_rejects_unbound_prefix_imports(reuse_command, tmp_path, unknown):
    root, env, _ = reuse_command
    prefix = tmp_path / "private-prefix"
    prefix.mkdir()
    if unknown == "external-symlink":
        (prefix / "external").symlink_to(root, target_is_directory=True)
    elif unknown == "customization-package":
        (prefix / "sitecustomize").mkdir()
    else:
        (prefix / "sitecustomize.py").write_text("# A dormant customization.\n")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from verification_reuse import runtime_identity; "
            "sys.prefix = sys.argv[2]; "
            "assert runtime_identity(Path(sys.argv[3])) is None",
            str(root / "tools"),
            str(prefix),
            str(root),
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "hook", ["external-module", "unknown-hook", "site-customization", "customization-package"]
)
def test_external_site_hook_cannot_reuse_changed_check_provider(reuse_command, tmp_path, hook):
    """A real external module supplied through .pth remains an unknown input."""
    root, env, _ = reuse_command
    prefix = tmp_path / "hooked-environment"
    venv.EnvBuilder(system_site_packages=True).create(prefix)
    alias = prefix / "lib64"
    if alias.is_symlink():
        alias.unlink()
    executable = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = subprocess.run(
        [str(executable), "-c", "import site; print(site.getsitepackages()[0])"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    external = tmp_path / "external-imports"
    external.mkdir()
    provider = external / "pydocstyle.py"
    provider.write_text("raise SystemExit(0)\n")
    site_directory = Path(site.stdout.strip())
    if hook == "external-module":
        (site_directory / "external.pth").write_text(
            "import sys; sys.path.insert(0, " + repr(str(external)) + ")\n"
        )
    elif hook == "unknown-hook":
        (site_directory / "unrecognized.pth").write_text("# An unaudited site hook.\n")
    elif hook == "site-customization":
        (site_directory / "sitecustomize.py").write_text("# An unaudited customization.\n")
    else:
        customization = site_directory / "sitecustomize"
        customization.mkdir()
        (customization / "__init__.py").write_text("# An unaudited startup package.\n")
    args = [str(executable), str(root / "tools/verification.py"), "fast"]
    first = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    receipt = next((root / "reports/verification").glob("*/checks.json"))
    record = json.loads(receipt.read_text())
    assert record["steps"][-1]["input_identity"]["runtime"] is None
    provider.write_text("raise SystemExit(7)\n")
    result = subprocess.run(
        [*args, "--reuse", str(receipt)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == (1 if hook == "external-module" else 0), (
        result.stdout + result.stderr
    )
    path = next(
        path
        for path in (root / "reports/verification").glob("*/checks.json")
        if str(path.parent) in result.stdout
    )
    step = json.loads(path.read_text())["steps"][-1]
    assert step["state"] == ("failed" if hook == "external-module" else "passed")
    assert step["returncode"] == (7 if hook == "external-module" else 0)
