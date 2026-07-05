"""Audit test fixtures for size and provenance hints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIMIT_BYTES = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-bytes", type=int, default=DEFAULT_LIMIT_BYTES)
    parser.add_argument("paths", nargs="*", default=["tests"])
    args = parser.parse_args()

    errors: list[str] = []
    for raw_path in args.paths:
        path = ROOT / raw_path
        if not path.exists():
            continue
        files = (
            [path]
            if path.is_file()
            else [candidate for candidate in path.rglob("*") if candidate.is_file()]
        )
        for file_path in files:
            if file_path.name.startswith("."):
                continue
            size = file_path.stat().st_size
            if size > args.limit_bytes:
                metadata = file_path.with_suffix(file_path.suffix + ".yml")
                if not metadata.is_file():
                    relpath = file_path.relative_to(ROOT)
                    errors.append(f"{relpath} is {size} bytes and lacks fixture metadata")

    if errors:
        print("Fixture audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Fixture audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
