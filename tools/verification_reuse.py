"""Reuse only closed, unchanged deterministic docstring evidence with provenance."""

import json
from importlib.machinery import all_suffixes
import os
from pathlib import Path
import site
import sys

from verification_inputs import CHECK_VERSION, digest, file_hash

# Audited startup hooks from the unchanged setuptools/coverage verification lock.
KNOWN_STARTUP_HOOKS = {
    "distutils-precedence.pth": (
        "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224",
    ),
    "a1_coverage.pth": (
        "ef2ed06d19867ec669c09a804060666a9cd5e383af0a9d11aa2de79b77d448e8",
        "f1498191b7f52180654ccdb6195233612805e26344100c093058343ea04afd36",
    ),
}


def runtime_identity(root):
    """Recheck exact interpreter/dependency bytes; decline uncontrolled imports."""
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        return None
    if Path(site.getusersitepackages()).exists():
        return None
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        return None
    suffixes = (*[suffix.casefold() for suffix in all_suffixes()], ".pyd")
    if any(
        path.name.casefold().endswith(suffixes) and path.name not in ("setup.py", "versioneer.py")
        for path in root.iterdir()
    ):
        return None
    if any(
        path.is_dir()
        and path.name not in ("dphtools", "tools", "tests")
        and any(child.name.casefold().endswith(suffixes) for child in path.iterdir())
        for path in root.iterdir()
    ):
        return None
    prefixes = sorted({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()})
    if any(
        not (
            path in (root.resolve(), (root / "tools").resolve())
            or any(path.is_relative_to(prefix) for prefix in prefixes)
        )
        for path in (Path(entry or root).resolve() for entry in sys.path)
    ):
        return None
    entries = []
    aliases = []
    for prefix in prefixes:
        for path in sorted(prefix.rglob("*")):
            if path.name in ("sitecustomize", "usercustomize"):
                return None
            if path.is_symlink() and path.is_dir():
                target = path.resolve()
                if not any(target.is_relative_to(canonical) for canonical in prefixes):
                    return None
                # The canonical prefix traversal hashes the target's actual bytes.
                aliases.append((str(path), str(target)))
            if path.is_file():
                hashed = file_hash(path)
                if (
                    path.suffix == ".pth" and hashed not in KNOWN_STARTUP_HOOKS.get(path.name, ())
                ) or path.name.startswith(("sitecustomize.", "usercustomize.")):
                    return None
                entries.append((str(path), hashed))
    return {
        "runtime_digest": digest({"files": entries, "directory_aliases": aliases}),
        "runtime_files": len(entries),
        "root": str(root),
        "root_entries": sorted(
            path.name
            for path in root.iterdir()
            if path.name
            not in (
                "reports",
                ".git",
                ".python",
                ".venv",
                ".venv-delivery",
                ".pytest_cache",
                "__pycache__",
            )
        ),
    }


def reusable(receipt, identity, command):
    """Validate the original invocation, inputs, command and actual diagnostic bytes."""
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        seal = record.pop("receipt_digest")
        if not (
            seal == digest(record)
            and record["document_version"] == CHECK_VERSION
            and record["mode"] == "fast"
            and record["complete"] is True
            and record["outcome"] == "passed"
            and record["failed"] == []
            and record["identity_digest"] == digest(record["identity"])
            and [step["name"] for step in record["steps"]] == ["format", "lint", "docstrings"]
            and all(
                step["state"] == "passed"
                and step["returncode"] == 0
                and not step["blocking_reasons"]
                for step in record["steps"]
            )
        ):
            raise ValueError("Original receipt is incomplete, failed or unverifiable")
        for step in record["steps"]:
            path = (receipt.parent / step["log"]["path"]).resolve()
            if not (
                step["input_digest"] == digest(step["input_identity"])
                and path.is_relative_to(receipt.parent.resolve())
                and file_hash(path) == step["log"]["sha256"]
            ):
                raise ValueError("Original step inputs or log bytes differ")
        step = record["steps"][-1]
        if step["command"] != command or step["input_identity"] != identity:
            raise ValueError("Command or declared inputs changed")
        if identity["runtime"] is None:
            raise ValueError("Runtime imports cannot be completely identified")
        return {
            "receipt": str(receipt.resolve()),
            "receipt_sha256": file_hash(receipt),
            "command": step["command"],
            "input_digest": step["input_digest"],
            "log": str(path),
            "log_sha256": step["log"]["sha256"],
            "duration_seconds": step["duration_seconds"],
        }, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, str(error)
