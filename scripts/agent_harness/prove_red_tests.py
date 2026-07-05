"""Prove red tests against a base commit.

This script is intentionally conservative in Phase 0. Full base-checkout
or worktree orchestration is enabled in Phase 2.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tests", nargs="+", required=True)
    parser.add_argument("--phase2-enabled", action="store_true")
    args = parser.parse_args()

    if not args.phase2_enabled:
        print(
            "Red-test proof is documented but not automated in Phase 0. "
            "Run this script again after Phase 2 enables safe base-worktree orchestration."
        )
        return 2

    print("Phase 2 red-test proof is not implemented yet.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
