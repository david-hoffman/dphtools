"""Downstream smoke-test placeholder."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "No downstream consumers are configured. "
        "Add downstream smoke targets after a downstream inventory issue identifies them."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
