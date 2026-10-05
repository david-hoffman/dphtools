"""Identify immutable verifier inputs and Versioneer's actual Git context."""

import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

CHECK_VERSION = "2.0"


def digest(value):
    """Hash an ordinary JSON identity using a stable representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def file_hash(path):
    """Hash exact artifact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_identity(root, sources):
    """Bind source, tests, configuration, tools, lock, and installed environment."""
    paths = {root / source for source in sources}
    for directory, pattern in (
        ("dphtools", "*"),
        ("tools", "*"),
        ("tests", "*"),
        ("notebooks", "*.ipynb"),
        (".github", "*"),
        (".githooks", "*"),
    ):
        paths.update(
            path
            for path in (root / directory).rglob(pattern)
            if path.is_file() and "__pycache__" not in path.parts
        )
    paths.update(
        root / name
        for name in (
            "setup.py",
            "versioneer.py",
            "README.md",
            "LICENSE.md",
            "MANIFEST.in",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            "requirements.txt",
            "requirements-dev.in",
            "requirements-dev.lock",
            ".gitignore",
            ".flake8",
            "mypy.ini",
            ".mypy.ini",
            ".pydocstyle",
            ".pydocstyle.ini",
        )
        if (root / name).is_file()
    )
    # Pydocstyle 6.3 inherits these configurations through every ancestor.
    paths.update(
        directory / name
        for directory in (root, *root.parents)
        for name in (
            "setup.cfg",
            "tox.ini",
            ".pydocstyle",
            ".pydocstyle.ini",
            ".pydocstylerc",
            ".pydocstylerc.ini",
            "pyproject.toml",
            ".pep257",
        )
        if (directory / name).is_file()
    )
    paths.update((root / "notebooks").rglob(".gitignore"))
    inputs = {
        Path(os.path.relpath(path, root)).as_posix(): file_hash(path) for path in sorted(paths)
    }
    dependencies = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata["Name"].lower() != "dphtools"
    }
    environment = {
        "ambient_environment_digest": digest(dict(os.environ)),
        "executable": sys.executable,
        "executable_sha256": file_hash(sys.executable),
        "version": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "dependencies": dict(sorted(dependencies.items())),
    }
    return {
        "check_version": CHECK_VERSION,
        "inputs": inputs,
        "environment": environment,
        "coverage_settings": inputs.get("setup.cfg"),
    }


def git_identity(root):
    """Capture revision, tags, reachable history, shallow/dirty state, and version."""
    commands = {
        "revision": ["git", "rev-parse", "HEAD"],
        "tags": ["git", "show-ref", "--tags", "--dereference"],
        "history": ["git", "rev-list", "HEAD"],
        "shallow": ["git", "rev-parse", "--is-shallow-repository"],
        "dirty": ["git", "status", "--porcelain", "--untracked-files=normal"],
        "describe": ["git", "describe", "--tags", "--always", "--long", "--dirty"],
        "version": [
            sys.executable,
            "-c",
            "import json, runpy; "
            "print(json.dumps(runpy.run_path('dphtools/_version.py')['get_versions']()))",
        ],
    }
    result = {}
    for name, command in commands.items():
        try:
            probe = subprocess.run(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="backslashreplace",
                check=False,
            )
            result[name] = {"returncode": probe.returncode, "output": probe.stdout.strip()}
        except OSError as error:
            result[name] = {"returncode": 1, "output": str(error)}
    return result
