"""Exercise optional reuse through the actual verifier and installed check tools."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import venv
from numpy._core import _multiarray_umath as native_module
from importlib.machinery import EXTENSION_SUFFIXES

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
    # Keep a cold check from changing the exact runtime bytes it fingerprints.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A preinstalled user site is an uncontrolled import provider on some hosts.
    env["PYTHONUSERBASE"] = str(root / "private-user-base")

    def run(receipt=None):
        previous = set((root / "reports/verification").glob("*/checks.json"))
        args = [sys.executable, str(root / "tools/verification.py"), "fast"]
        if receipt is not None:
            args.extend(["--reuse", str(receipt)])
        result = subprocess.run(
            args, cwd=root, env=env, text=True, capture_output=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
        paths = set((root / "reports/verification").glob("*/checks.json")) - previous
        assert len(paths) == 1, result.stdout + result.stderr
        path = paths.pop()
        return path, json.loads(path.read_text())

    return root, env, run


def test_unchanged_check_reuses_original_command_and_retained_log(reuse_command):
    root, env, run = reuse_command
    original, source = run()
    reused_path, reused = run(original)
    step = next(step for step in reused["steps"] if step["name"] == "docstrings")
    assert reused_path != original
    assert step["state"] == "reused", (
        step.get("reuse_rejection"),
        runtime_diagnostics(root, env),
    )
    assert step["provenance"]["receipt"] == str(original)
    assert step["provenance"]["command"] == source["steps"][-1]["command"]
    assert step["provenance"]["receipt_sha256"]
    assert step["returncode"] is None
    assert reused["complete"] is True and reused["outcome"] == "passed"
    assert [step["state"] for step in reused["steps"][:2]] == ["passed", "passed"]
    # A reused receipt is not presented as a new execution or recursively trusted.
    _, next_run = run(reused_path)
    assert next_run["steps"][-1]["state"] == "passed"


def runtime_diagnostics(root, env):
    """Show import and alias facts if a host cannot satisfy the reuse contract."""
    script = r"""
import hashlib, json, site, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
prefixes = sorted({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()})
paths = [path for prefix in prefixes for path in prefix.rglob('*')]
user_site = Path(site.getusersitepackages())
print(json.dumps({
    'prefixes': [str(path) for path in prefixes],
    'sys_path': sys.path,
    'user_site': {'path': str(user_site), 'exists': user_site.exists()},
    'customizations': [name for name in ('sitecustomize', 'usercustomize') if name in sys.modules],
    'directory_aliases': [(str(path), str(path.resolve())) for path in paths
                          if path.is_symlink() and path.is_dir()],
    'startup_hooks': [(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
                      for path in paths if path.is_file() and path.suffix == '.pth'],
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root / "tools")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr


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


@pytest.mark.parametrize(
    "unknown",
    ["import-path", "root-module", "root-package", "native-module", "native-package", "user-site"],
)
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
    elif unknown == "native-module":
        shutil.copy2(native_module.__file__, root / ("_multiarray_umath" + EXTENSION_SUFFIXES[0]))
    elif unknown == "native-package":
        package = root / "_multiarray_umath"
        package.mkdir()
        shutil.copy2(native_module.__file__, package / ("__init__" + EXTENSION_SUFFIXES[0]))
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


@pytest.mark.parametrize("target_prefix", ["environment", "base"])
def test_runtime_identity_binds_internal_directory_aliases(reuse_command, tmp_path, target_prefix):
    """The alias and its fully hashed target both determine runtime identity."""
    root, env, _ = reuse_command
    prefix = tmp_path / "environment"
    base = tmp_path / "base"
    prefix.mkdir()
    base.mkdir()
    target_root = prefix if target_prefix == "environment" else base
    first = target_root / "first"
    second = target_root / "second"
    first.mkdir()
    second.mkdir()
    (first / "input").write_text("original bytes")
    (second / "input").write_text("original bytes")
    alias = prefix / "Headers"
    alias.symlink_to(first, target_is_directory=True)
    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from verification_reuse import runtime_identity
root, prefix, base, target, replacement = map(Path, sys.argv[2:])
sys.prefix, sys.base_prefix = str(prefix), str(base)
sys.path = [str(root), str(root / 'tools')]
original = runtime_identity(root)
assert original is not None, 'A fully hashed internal alias must be eligible'
alias = prefix / 'Headers'
renamed = prefix / 'AlternateHeaders'
alias.rename(renamed)
assert runtime_identity(root) != original, 'The alias path must bind the identity'
renamed.rename(alias)
assert runtime_identity(root) == original
(target / 'input').write_text('changed bytes')
assert runtime_identity(root) != original, 'Target bytes must bind the identity'
(target / 'input').write_text('original bytes')
assert runtime_identity(root) == original
alias.unlink()
alias.symlink_to(replacement, target_is_directory=True)
assert runtime_identity(root) != original, 'Same-content retargeting must bind the identity'
alias.unlink()
alias.symlink_to(root, target_is_directory=True)
assert runtime_identity(root) is None, 'An external alias must remain ineligible'
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root / "tools"),
            str(root),
            str(prefix),
            str(base),
            str(first),
            str(second),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_venv_directory_aliases_allow_unchanged_command_reuse(reuse_command, tmp_path):
    """A genuine venv keeps its stock lib64 alias and reuses the real check."""
    root, env, _ = reuse_command
    prefix = tmp_path / "alias-environment"
    venv.EnvBuilder(system_site_packages=True).create(prefix)
    lib64 = prefix / "lib64"
    original_lib64 = lib64.readlink() if lib64.is_symlink() else None
    target = prefix / "owned-headers"
    target.mkdir()
    (target / "input").write_text("Bound runtime bytes.\n")
    (prefix / "Headers").symlink_to(target, target_is_directory=True)
    executable = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    args = [str(executable), str(root / "tools/verification.py"), "fast"]
    first = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    original = next((root / "reports/verification").glob("*/checks.json"))
    source = json.loads(original.read_text())
    assert source["steps"][-1]["input_identity"]["runtime"] is not None
    result = subprocess.run(
        [*args, "--reuse", str(original)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    paths = set((root / "reports/verification").glob("*/checks.json")) - {original}
    assert len(paths) == 1, result.stdout + result.stderr
    step = json.loads(paths.pop().read_text())["steps"][-1]
    assert step["state"] == "reused", step.get("reuse_rejection")
    assert step["provenance"]["receipt"] == str(original)
    if original_lib64 is not None:
        assert lib64.readlink() == original_lib64
