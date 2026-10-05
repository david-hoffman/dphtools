"""Exercise optional reuse through the actual verifier and installed check tools."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import tempfile
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
    if step["state"] != "reused":
        fail_with_runtime_diagnostics(
            root, env, f"Docstrings state: {step['state']}; {step.get('reuse_rejection')}"
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


def runtime_diagnostics(root, env, executable=None):
    """Show import and alias facts if a host cannot satisfy the reuse contract."""
    script = r"""
import hashlib, json, site, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
prefixes = sorted({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()})
paths = [path for prefix in prefixes for path in prefix.rglob('*')]
user_site = Path(site.getusersitepackages())
print(json.dumps({
    'executable': sys.executable,
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
        [str(executable or sys.executable), "-c", script, str(root / "tools")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr


def fail_with_runtime_diagnostics(root, env, reason, executable=None):
    """Retain full facts in the failure XML and print them only on failure."""
    diagnostics = runtime_diagnostics(root, env, executable)
    print(diagnostics, flush=True)
    pytest.fail(f"{reason}\n{diagnostics}", pytrace=False)


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
    if source["steps"][-1]["input_identity"]["runtime"] is None:
        fail_with_runtime_diagnostics(root, env, "Venv runtime is unidentified", executable)
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
    if step["state"] != "reused":
        fail_with_runtime_diagnostics(root, env, step.get("reuse_rejection"), executable)
    assert step["provenance"]["receipt"] == str(original)
    if original_lib64 is not None:
        assert lib64.readlink() == original_lib64


def changed_config_runs_failing_check(root, env, original, code):
    """Assert the public command executes the changed check instead of reusing it."""
    previous = set((root / "reports/verification").glob("*/checks.json"))
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/verification.py"),
            "fast",
            "--reuse",
            str(original),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    paths = set((root / "reports/verification").glob("*/checks.json")) - previous
    assert len(paths) == 1, result.stdout + result.stderr
    current = paths.pop()
    record = json.loads(current.read_text())
    step = record["steps"][-1]
    assert step["state"] == "failed" and step["returncode"] == 1
    assert step["command"] == json.loads(original.read_text())["steps"][-1]["command"]
    assert "inputs changed" in step["reuse_rejection"]
    assert code in (current.parent / step["log"]["path"]).read_text()
    return record


@pytest.mark.parametrize(
    ("filename", "change"),
    [
        (".pydocstylerc", "modify"),
        (".pydocstylerc.ini", "modify"),
        (".pep257", "modify"),
        (".pydocstylerc", "create"),
        (".pydocstylerc", "delete"),
    ],
)
def test_root_docstring_config_changes_execute_new_failures(reuse_command, filename, change):
    """Implicit pydocstyle discovery must retain its actual configuration inputs."""
    import configparser

    root, env, run = reuse_command
    parser = configparser.ConfigParser()
    parser.read(root / "setup.cfg")
    parser.remove_section("pydocstyle")
    with (root / "setup.cfg").open("w") as stream:
        parser.write(stream)
    if change == "create":
        source = '"""An isolated package."""\n\n\nclass Example:\n    """A documented class."""\n'
        expected_code = "D203"
    else:
        source = '"""An isolated package."""\n\n\ndef example():\n    return 1\n'
        expected_code = "D103"
    (root / "dphtools/__init__.py").write_text(source)
    configuration = root / filename
    if change != "create":
        configuration.write_text("[pydocstyle]\ninherit = false\nselect = D104\n")
    original, before = run()
    assert before["steps"][-1]["state"] == "passed"
    if change == "delete":
        configuration.unlink()
    else:
        configuration.write_text("[pydocstyle]\ninherit = false\nselect = " + expected_code + "\n")
    after = changed_config_runs_failing_check(root, env, original, expected_code)
    assert after["identity"]["inputs"] != before["identity"]["inputs"]
    if change == "delete":
        assert filename in before["identity"]["inputs"]
        assert filename not in after["identity"]["inputs"]
    else:
        assert filename in after["identity"]["inputs"]


def test_ancestor_config_change_executes_failure_with_unchanged_root_config(reuse_command):
    """Root convention selection still inherits ancestor decorator exclusions."""
    root, env, run = reuse_command
    root_configuration = (root / "setup.cfg").read_bytes()
    (root / "dphtools/__init__.py").write_text(
        '"""An isolated package."""\n\n\n'
        "def _decorator(function):\n    return function\n\n\n"
        "@_decorator\ndef example():\n    return 1\n"
    )
    configuration = root.parent / ".pydocstyle"
    configuration.write_text("[pydocstyle]\ninherit = false\nignore-decorators = _decorator\n")
    original, before = run()
    configuration.write_text("[pydocstyle]\ninherit = false\nignore-decorators = never_match\n")
    after = changed_config_runs_failing_check(root, env, original, "D103")
    assert (root / "setup.cfg").read_bytes() == root_configuration
    assert (
        before["identity"]["inputs"]["../.pydocstyle"]
        != after["identity"]["inputs"]["../.pydocstyle"]
    )


@pytest.mark.parametrize(
    "filename",
    [
        "setup.cfg",
        "tox.ini",
        ".pydocstyle",
        ".pydocstyle.ini",
        ".pydocstylerc",
        ".pydocstylerc.ini",
        "pyproject.toml",
        ".pep257",
    ],
)
def test_input_identity_binds_config_lifecycle_through_ancestors(
    reuse_command, tmp_path, filename
):
    """Every pinned discovery name binds creation, modification, and deletion."""
    _, env, _ = reuse_command
    root = tmp_path / "outer/middle/repository"
    root.mkdir(parents=True)
    script = r"""
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from verification_inputs import input_identity, file_hash
root = Path(sys.argv[2])
for directory in (root, root.parent, root.parent.parent):
    path = directory / sys.argv[3]
    key = Path(os.path.relpath(path, root)).as_posix()
    before = input_identity(root, [])['inputs']
    assert key not in before
    path.write_text('first configuration bytes')
    created = input_identity(root, [])['inputs']
    assert created.get(key) == file_hash(path), 'Created config must bind relative ancestor key'
    path.write_text('changed configuration bytes')
    changed = input_identity(root, [])['inputs']
    assert changed[key] == file_hash(path) and changed != created
    path.unlink()
    assert input_identity(root, [])['inputs'] == before, 'Deletion must remove the bound input'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "tools"), str(root), filename],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("extension", [".PY", ".PYC", ".PYD"])
@pytest.mark.parametrize("provider", ["module", "package"])
def test_uppercase_import_suffixes_are_uncontrolled_on_every_host(
    reuse_command, tmp_path, extension, provider
):
    """Windows-recognized providers remain unknown even on case-sensitive hosts."""
    import py_compile

    root, env, _ = reuse_command
    if provider == "module":
        path = root / ("pydocstyle" + extension)
    else:
        package = root / "pydocstyle"
        package.mkdir()
        path = package / ("__init__" + extension)
    if extension == ".PY":
        path.write_text("raise SystemExit(7)\n")
    elif extension == ".PYC":
        source = tmp_path / "private-source.py"
        source.write_text("raise SystemExit(7)\n")
        py_compile.compile(str(source), cfile=str(path), doraise=True)
    else:
        shutil.copy2(native_module.__file__, path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from verification_reuse import runtime_identity; "
            "assert runtime_identity(Path(sys.argv[2])) is None",
            str(root / "tools"),
            str(root),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_uppercase_windows_check_provider_runs_again_after_change(reuse_command):
    """The real Windows entrypoint must observe an uppercase provider's new exit."""
    root, env, run = reuse_command
    provider = root / "pydocstyle.PY"
    provider.write_text("raise SystemExit(0)\n")
    original, before = run()
    assert before["steps"][-1]["input_identity"]["runtime"] is None
    provider.write_text("raise SystemExit(7)\n")
    previous = set((root / "reports/verification").glob("*/checks.json"))
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/verification.py"),
            "fast",
            "--reuse",
            str(original),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == (1 if os.name == "nt" else 0), result.stdout + result.stderr
    paths = set((root / "reports/verification").glob("*/checks.json")) - previous
    assert len(paths) == 1, result.stdout + result.stderr
    step = json.loads(paths.pop().read_text())["steps"][-1]
    assert step["state"] == ("failed" if os.name == "nt" else "passed")
    assert step["returncode"] == (7 if os.name == "nt" else 0)
    assert "cannot be completely identified" in step["reuse_rejection"]


def locked_coverage_hook_variants():
    """Return the verified platform bytes without changing the installed hook."""
    from importlib.metadata import distribution

    installed = Path(distribution("coverage").locate_file("a1_coverage.pth"))
    lf = installed.read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(lf).hexdigest() == (
        "ef2ed06d19867ec669c09a804060666a9cd5e383af0a9d11aa2de79b77d448e8"
    )
    assert hashlib.sha256(crlf).hexdigest() == (
        "f1498191b7f52180654ccdb6195233612805e26344100c093058343ea04afd36"
    )
    return installed, lf, crlf


def test_runtime_identity_accepts_exact_locked_coverage_platform_hooks(reuse_command, tmp_path):
    """Both audited hooks bind distinct identities; any unknown bytes decline."""
    root, env, _ = reuse_command
    _, lf, crlf = locked_coverage_hook_variants()
    prefix = tmp_path / "private-hook-prefix"
    prefix.mkdir()
    hook = prefix / "a1_coverage.pth"
    hook.write_bytes(lf)
    crlf_file = tmp_path / "windows-hook"
    crlf_file.write_bytes(crlf)
    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from verification_reuse import runtime_identity
root, prefix, windows = map(Path, sys.argv[2:])
sys.prefix = sys.base_prefix = str(prefix)
sys.path = [str(root), str(root / 'tools'), str(prefix)]
hook = prefix / 'a1_coverage.pth'
original = hook.read_bytes()
lf = runtime_identity(root)
assert lf is not None
hook.write_bytes(windows.read_bytes())
crlf = runtime_identity(root)
assert crlf is not None, 'The exact locked Windows coverage hook must be eligible'
assert crlf != lf, 'Known byte variants must retain distinct fingerprints'
hook.write_bytes(original + b'# Unknown startup bytes\n')
assert runtime_identity(root) is None
hook.write_bytes(original)
assert runtime_identity(root) == lf
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root / "tools"),
            str(root),
            str(prefix),
            str(crlf_file),
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_venv_reuses_exact_windows_hook_and_rechecks_changed_bytes(reuse_command, tmp_path):
    """The real command accepts CRLF but executes fresh checks on byte changes."""
    root, env, _ = reuse_command
    installed, lf, crlf = locked_coverage_hook_variants()
    original_installed = installed.read_bytes()
    prefix = tmp_path / "windows-hook-environment"
    venv.EnvBuilder(system_site_packages=True).create(prefix)
    executable = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = subprocess.run(
        [str(executable), "-c", "import site; print(site.getsitepackages()[0])"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    site_directory = Path(site.stdout.strip())
    site_directory.mkdir(parents=True, exist_ok=True)
    hook = site_directory / "a1_coverage.pth"
    hook.write_bytes(crlf)

    def run(receipt=None):
        previous = set((root / "reports/verification").glob("*/checks.json"))
        args = [str(executable), str(root / "tools/verification.py"), "fast"]
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

    original, source = run()
    if source["steps"][-1]["input_identity"]["runtime"] is None:
        fail_with_runtime_diagnostics(
            root, env, "Windows hook runtime is unidentified", executable
        )
    _, reused = run(original)
    assert reused["steps"][-1]["state"] == "reused"
    assert reused["steps"][-1]["provenance"]["receipt"] == str(original)
    hook.write_bytes(lf)
    _, changed = run(original)
    assert changed["steps"][-1]["state"] == "passed"
    assert "inputs changed" in changed["steps"][-1]["reuse_rejection"]
    assert changed["steps"][-1]["input_identity"]["runtime"] is not None
    assert changed["steps"][-1]["input_identity"] != source["steps"][-1]["input_identity"]
    hook.write_bytes(lf + b"# Unknown startup bytes\n")
    _, unknown = run(original)
    assert unknown["steps"][-1]["state"] == "passed"
    assert unknown["steps"][-1]["input_identity"]["runtime"] is None
    assert unknown["steps"][-1]["reuse_rejection"]
    assert installed.read_bytes() == original_installed


@pytest.fixture
def startup_command(reuse_command, tmp_path):
    """Run unchanged owned startup code from a genuinely private interpreter."""
    root, env, _ = reuse_command
    prefix = tmp_path / "startup-environment"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(prefix)
    executable = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = subprocess.run(
        [str(executable), "-c", "import site; print(site.getsitepackages()[0])"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    (root / "tools/__init__.py").write_text('"""Owned startup instrumentation."""\n')
    (root / "tools/audit_probe.py").write_text(
        '"""An immutable plugin controlled by the active coverage configuration."""\n\n'
        "import sys\n\n\n"
        "def coverage_init(reg, options):\n"
        '    """Apply configuration only to the actual docstring check child."""\n'
        '    if "pydocstyle" in sys.orig_argv and options.get("fail_doc") == "yes":\n'
        "        raise SystemExit(7)\n"
    )
    shutil.copytree(root / "tools", Path(site.stdout.strip()) / "tools")
    env.pop("COVERAGE_PROCESS_CONFIG", None)

    def run(receipt=None, expected=0):
        previous = set((root / "reports/verification").glob("*/checks.json"))
        args = [str(executable), str(root / "tools/verification.py"), "fast"]
        if receipt is not None:
            args.extend(["--reuse", str(receipt)])
        # The verifier's checks use root even when its caller uses another directory.
        result = subprocess.run(
            args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60
        )
        assert result.returncode == expected, result.stdout + result.stderr
        paths = set((root / "reports/verification").glob("*/checks.json")) - previous
        assert len(paths) == 1, result.stdout + result.stderr
        path = paths.pop()
        return path, json.loads(path.read_text())

    return root, env, run


@pytest.mark.parametrize("location", ["absolute", "relative", "create"])
def test_active_coverage_configuration_change_executes_new_failure(startup_command, location):
    """External startup options must invalidate an otherwise identical passing check."""
    root, env, run = startup_command
    with tempfile.TemporaryDirectory(prefix="reuse-startup-config-") as directory:
        config = (
            root.parent / "startup.coveragerc"
            if location == "relative"
            else Path(directory) / "startup.coveragerc"
        )
        env["COVERAGE_PROCESS_START"] = (
            "../startup.coveragerc" if location == "relative" else str(config)
        )
        options = (
            "[run]\nbranch = True\nparallel = True\ninclude = */tools/*.py\n"
            "plugins = tools.audit_probe\n\n[tools.audit_probe]\nfail_doc = "
        )
        if location != "create":
            config.write_text(options + "no\n")
        original, before = run()
        if location != "create":
            _, reused = run(original)
            assert reused["steps"][-1]["state"] == "reused"
        config.write_text(options + "yes\n")
        current, after = run(original, expected=1)
        step = after["steps"][-1]
        assert step["state"] == "failed" and step["returncode"] == 1
        assert step["command"] == before["steps"][-1]["command"]
        assert "inputs changed" in step["reuse_rejection"]
        assert "SystemExit: 7" in (current.parent / step["log"]["path"]).read_text()
        assert after["identity"]["coverage_startup"] == {
            "path": str(config.resolve()),
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        }
        if location == "create":
            assert before["steps"][-1]["input_identity"]["runtime"] is None


def test_startup_configuration_identity_handles_presence_paths_and_unknown_bytes(
    reuse_command, tmp_path
):
    """Use the same public identity functions as the verifier without runtime seams."""
    root, env, _ = reuse_command
    configuration = root.parent / "external.coveragerc"
    caller = tmp_path / "different-caller"
    caller.mkdir()
    script = r"""
import hashlib, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from verification_inputs import input_identity
from verification_reuse import runtime_identity
root, config = map(Path, sys.argv[2:])
def identity():
    return input_identity(root, [])['coverage_startup']
os.environ.pop('COVERAGE_PROCESS_START', None)
os.environ.pop('COVERAGE_PROCESS_CONFIG', None)
assert identity() is None
os.environ['COVERAGE_PROCESS_START'] = ''
assert identity() is None
os.environ['COVERAGE_PROCESS_START'] = '../external.coveragerc'
assert identity() == {'path': str(config.resolve()), 'sha256': None}
assert runtime_identity(root) is None, 'Missing configuration cannot provide reusable evidence'
config.write_text('[run]\nbranch = True\n')
first = identity()
assert first == {'path': str(config.resolve()), 'sha256': hashlib.sha256(config.read_bytes()).hexdigest()}
config.write_text('[run]\nbranch = False\n')
assert identity() != first
os.environ['COVERAGE_PROCESS_START'] = str(config)
assert identity()['path'] == str(config.resolve())
config.unlink()
assert identity()['sha256'] is None
assert runtime_identity(root) is None
config.mkdir()
assert identity()['sha256'] is None, 'An unreadable configuration is unidentified'
assert runtime_identity(root) is None
for inline in ('', 'inline takes priority even when the file is unreadable'):
    os.environ['COVERAGE_PROCESS_CONFIG'] = inline
    assert identity() is None, 'Inline presence, not truthiness, selects the active input'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root / "tools"), str(root), str(configuration)],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inline_coverage_configuration_preserves_reuse_with_inactive_file(reuse_command):
    """The pinned hook prioritizes serialized options over the external file."""
    from coverage.config import CoverageConfig

    root, env, run = reuse_command
    configuration = root.parent / "inactive.coveragerc"
    env["COVERAGE_PROCESS_START"] = str(configuration)
    options = CoverageConfig()
    options.branch = options.parallel = True
    options.include = ["*/tools/*.py"]
    options.data_file = env.get("COVERAGE_FILE", str(root.parent / "inline.coverage"))
    env["COVERAGE_PROCESS_CONFIG"] = options.serialize()
    original, before = run()
    assert before["identity"]["coverage_startup"] is None
    configuration.write_text("An unreadable coverage configuration.\n")
    _, after = run(original)
    assert after["steps"][-1]["state"] == "reused"
    assert after["identity"]["coverage_startup"] is None
