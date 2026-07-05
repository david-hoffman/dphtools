"""Diff coverage gate placeholder for later harness phases."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enabled", action="store_true")
    parser.parse_args()

    print("Diff coverage enforcement is not enabled in Phase 0.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
