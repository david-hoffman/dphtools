#!/usr/bin/env python3
"""Install this invocation's exact wheel and source distribution in clean environments."""

from email.parser import BytesParser
from pathlib import Path
import os
import sys
import zipfile

from release import distributions, smoke


def main(directory, constraints):
    """Check build metadata, smoke both retained artifacts, and recheck their hashes."""
    directory = Path(directory)
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one current wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Ambiguous wheel metadata")
        version = BytesParser().parsebytes(archive.read(names[0]))["Version"]
    files = distributions(directory, version)
    os.environ["PIP_CONSTRAINT"] = str(Path(constraints).resolve())
    smoke({"version": version, "files": files}, directory)
    if distributions(directory, version) != files:
        raise ValueError("Artifacts changed during installation")
    print("Exact current wheel and source clean installations passed", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
