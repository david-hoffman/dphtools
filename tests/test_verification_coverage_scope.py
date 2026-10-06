"""Coverage measures owned code without tracing a repository-local environment."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import coverage

ROOT = Path(__file__).resolve().parents[1]


def test_repo_named_dphtools_does_not_instrument_environment_dependencies(tmp_path):
    repository = tmp_path / "dphtools"
    repository.mkdir()
    shutil.copy2(ROOT / "setup.cfg", repository / "setup.cfg")
    owned = repository / "dphtools/subpkg/deep.py"
    owned.parent.mkdir(parents=True)
    owned.write_text("VALUE = 42\n", encoding="utf-8")
    foreign = repository / ".venv/lib/site-packages/foreign.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("VALUE = 17\n", encoding="utf-8")
    installed = foreign.parent / "dphtools/utils/installed.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("VALUE = 23\n", encoding="utf-8")
    copied = tmp_path / "private-cli/tools/subpkg/probe.py"
    copied.parent.mkdir(parents=True)
    copied.write_text("VALUE = 11\n", encoding="utf-8")
    driver = repository / "driver.py"
    driver.write_text(
        "import runpy, sys\nfor path in sys.argv[1:]:\n    runpy.run_path(path)\n",
        encoding="utf-8",
    )
    data_file = tmp_path / ".coverage"
    env = dict(
        os.environ,
        COVERAGE_RCFILE=str(repository / "setup.cfg"),
        COVERAGE_FILE=str(data_file),
        DPHTOOLS_COVERAGE_ROOT=str(repository),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            str(driver),
            str(owned),
            str(foreign),
            str(installed),
            str(copied),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = list(tmp_path.glob(".coverage.*"))
    assert files
    measured = set()
    for path in files:
        data = coverage.CoverageData(basename=str(path))
        data.read()
        measured.update(Path(name).resolve() for name in data.measured_files())
    assert owned in measured
    assert installed in measured
    assert copied in measured
    assert foreign not in measured
