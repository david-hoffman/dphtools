"""Reuse only closed, unchanged deterministic docstring evidence with provenance."""

import json
import os
from pathlib import Path
import site
import sys

from verification_inputs import CHECK_VERSION, digest, file_hash


def runtime_identity(root):
    """Recheck exact interpreter/dependency bytes; decline uncontrolled imports."""
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        return None
    if Path(site.getusersitepackages()).exists():
        return None
    if any(path.name not in ("setup.py", "versioneer.py") for path in root.glob("*.py*")):
        return None
    entries = []
    for prefix in sorted({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}):
        for path in sorted(prefix.rglob("*")):
            if path.is_symlink() and path.is_dir():
                return None
            if path.is_file():
                entries.append((str(path), file_hash(path)))
    return {
        "runtime_digest": digest(entries),
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
