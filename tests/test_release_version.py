"""S1/S3: canonical release versions through the production worker CLI.

Expectations come only from roles/public-contract.md's explicit version grammar.
No runtime modules are imported. Missing public commands are failures, not skips.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def copy_project_context(repo):
    """Keep ordinary package/build context available without inspecting source bytes."""
    shutil.copytree(
        ROOT / "dphtools", repo / "dphtools", ignore=shutil.ignore_patterns("__pycache__")
    )
    for pattern in (
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "MANIFEST.in",
        "versioneer.py",
        "README*",
        "LICENSE*",
    ):
        for path in ROOT.glob(pattern):
            if path.is_file():
                shutil.copy2(path, repo / path.name)


def diagnostic(result):
    """Report status and messages without exposing child traceback source lines."""
    summaries = []
    for label, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
        lines = stream.splitlines()
        if any("Traceback (most recent call last)" in line for line in lines):
            # Unhandled child exceptions retain their final category/message only.
            lines = lines[-1:]
        else:
            # Warning source snippets are indented; retain category/message lines.
            lines = [line for line in lines if line and not line[0].isspace()]
        summaries.append(f"{label}=" + " | ".join(lines))
    return f"exit={result.returncode}; " + "; ".join(summaries)


def invoke(entry, *args, cwd, env=None, timeout=120):
    """Run the real Python entry point without shell expansion or raw tracebacks."""
    child_env = dict(os.environ if env is None else env)
    for key in list(child_env):
        if any(
            word in key.upper() for word in ("TOKEN", "PASSWORD", "CREDENTIAL", "SECRET")
        ) or key.upper().startswith(("TWINE_", "PYPI_", "TESTPYPI_", "GH_", "GITHUB_", "AWS_")):
            child_env.pop(key)
    child_env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(entry), *map(str, args)],
        cwd=cwd,
        env=child_env,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def worker(tmp_path):
    """Copy opaque runtime artifacts; do not inspect implementation bytes."""
    assert (
        ROOT / "tools/release.py"
    ).is_file(), "Missing public functionality: tools/release.py is not present"
    repo = tmp_path / "checkout with spaces & literal $value"
    shutil.copytree(ROOT / "tools", repo / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    copy_project_context(repo)
    outside = tmp_path / "outside checkout"
    outside.mkdir()
    return repo / "tools/release.py", outside


@pytest.mark.parametrize(
    ("version", "channel"),
    [
        ("0.0.0", "pypi"),
        ("1.0.0", "pypi"),
        ("21.34.55", "pypi"),
        ("1.0.0a0", "testpypi"),
        ("1.0.0b2", "testpypi"),
        ("1.0.0rc1", "testpypi"),
    ],
)
def test_version_reports_canonical_value_and_channel(worker, version, channel):
    entry, outside = worker
    result = invoke(entry, "version", version, cwd=outside)
    assert result.returncode == 0, diagnostic(result)
    payload = json.loads(result.stdout)
    assert payload["version"] == version
    assert payload["channel"] == channel


@pytest.mark.parametrize(
    "version",
    [
        "",
        "1",
        "1.0",
        "1.0.0.0",
        "v1.0.0",
        "V1.0.0",
        "01.0.0",
        "1.00.0",
        "1.0.00",
        "1.0.0RC1",
        "1.0.0rc01",
        "1.0.0alpha1",
        "1.0.0-rc1",
        "1.0.0rc",
        "1.0.0.dev1",
        "1.0.0.post1",
        "1.0.0+local",
        "1!1.0.0",
        " 1.0.0",
        "1.0.0\n",
        "1.0.0;echo unsafe",
    ],
)
def test_version_rejects_noncanonical_spellings(worker, version):
    entry, outside = worker
    # A valid request first distinguishes version rejection from an absent CLI.
    valid = invoke(entry, "version", "1.0.0", cwd=outside)
    assert valid.returncode == 0, diagnostic(valid)
    assert json.loads(valid.stdout)["version"] == "1.0.0"
    result = invoke(entry, "version", version, cwd=outside)
    assert result.returncode != 0, diagnostic(result)
    assert (result.stdout + result.stderr).strip(), "Rejection needs a diagnostic"
