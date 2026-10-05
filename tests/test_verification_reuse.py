"""Exercise optional reuse through the actual verifier and installed check tools."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

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


@pytest.mark.parametrize("unknown", ["import-path", "root-module", "user-site"])
def test_unknown_import_inputs_decline_even_identical_receipts(reuse_command, unknown):
    root, env, run = reuse_command
    if unknown == "import-path":
        env["PYTHONPATH"] = str(root / "tools")
    elif unknown == "root-module":
        (root / "local.py").write_text('"""An untracked import provider."""\n')
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


def test_runtime_identity_rejects_external_directory_symlinks(reuse_command, tmp_path):
    root, env, _ = reuse_command
    prefix = tmp_path / "private-prefix"
    prefix.mkdir()
    (prefix / "external").symlink_to(root, target_is_directory=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from verification_reuse import runtime_identity; "
            "sys.prefix = sys.base_prefix = sys.argv[2]; "
            "assert runtime_identity(Path(sys.argv[3])) is None",
            str(root / "tools"),
            str(prefix),
            str(root),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
