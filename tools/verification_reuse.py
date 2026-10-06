"""Reuse only closed, unchanged deterministic docstring evidence with provenance."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib.machinery import all_suffixes
import os
from pathlib import Path
import site
import sys

from verification_inputs import CHECK_VERSION, coverage_startup_identity, digest, file_hash

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


def startup_configuration_is_closed(root):
    """Decline custom startup plugins and configurations that cannot be identified."""
    inline = os.environ.get("COVERAGE_PROCESS_CONFIG")
    filename = os.environ.get("COVERAGE_PROCESS_START")
    if inline is None and not filename:
        return True
    try:
        from coverage.config import CoverageConfig, read_coverage_config

        if inline is not None:
            configuration = CoverageConfig.deserialize(inline)
        else:
            if filename == ".coveragerc" or os.environ.get("COVERAGE_FORCE_CONFIG"):
                return False
            configuration = read_coverage_config(str(root / filename), warn=lambda message: None)
        return configuration.plugins == []
    except Exception:
        # Any import/read/parse failure leaves the startup configuration unidentified.
        return False


def _runtime_file_hash(path):
    """Hash fresh runtime bytes without rebuilding a Path for each entry."""
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def runtime_identity(root):
    """Recheck exact interpreter/dependency bytes; decline uncontrolled imports."""
    startup = coverage_startup_identity(root)
    if startup is not None and startup["sha256"] is None:
        return None
    if not startup_configuration_is_closed(root):
        return None
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
    trees = []
    for prefix in prefixes:
        pending = [prefix] if prefix.is_dir() else []
        paths = []
        while pending:
            descendants = []
            try:
                directory = os.scandir(pending.pop())
            except OSError:
                continue
            with directory:
                for entry in directory:
                    paths.append(entry)
                    with suppress(OSError):
                        if entry.is_dir(follow_symlinks=False):
                            descendants.append(entry.path)
            pending.extend(reversed(descendants))
        # Keep Path component ordering and use each entry's cached file type.
        paths.sort(key=lambda entry: os.path.normcase(entry.path).split(os.sep))
        files = {
            entry.path: entry
            for entry in paths
            if (Path(entry.path).is_file() if entry.is_symlink() else entry.is_file())
        }
        trees.append((paths, files))
        for entry in paths:
            if entry.name in ("sitecustomize", "usercustomize"):
                return None
            if entry.is_symlink() and Path(entry.path).is_dir():
                target = Path(entry.path).resolve()
                if not any(target.is_relative_to(canonical) for canonical in prefixes):
                    return None
                # The canonical prefix traversal hashes the target's actual bytes.
                aliases.append((entry.path, str(target)))
            if entry.path in files and entry.name.startswith(("sitecustomize.", "usercustomize.")):
                return None
    # Reject unbound prefixes before reading otherwise reusable runtime bytes.
    for paths, files in trees:
        # Outer test workers already provide concurrency; bound nested readers.
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Parallelize substantial reads; tiny files avoid Future overhead.
            hashes = {
                path: pool.submit(_runtime_file_hash, path)
                for path, entry in files.items()
                if entry.stat().st_size >= 65536
            }
            for entry in paths:
                if entry.path in files:
                    future = hashes.get(entry.path)
                    hashed = future.result() if future else _runtime_file_hash(entry.path)
                    if (
                        entry.name.endswith(".pth")
                        and entry.name != ".pth"
                        and hashed not in KNOWN_STARTUP_HOOKS.get(entry.name, ())
                    ):
                        pool.shutdown(cancel_futures=True)
                        return None
                    entries.append((entry.path, hashed))
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
