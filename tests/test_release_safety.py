"""S4: useful local worker safety observations, not hosted authorization proof."""

import os

import pytest

from .test_release_reconcile import HTTP_DRIVER
from .test_release_support import Bundle, install_executable, snapshot, write_json
from . import test_release_version as version_cli
from .test_release_version import diagnostic, invoke

worker = version_cli.worker

PUBLISHER_TRAP = r"""
import os
from pathlib import Path
import sys
with Path(os.environ["RELEASE_TEST_PUBLISHER_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(Path(sys.argv[0]).name + "\n")
print("Diagnostic worker invoked a publication/approval tool", file=sys.stderr)
raise SystemExit(91)
"""


@pytest.mark.parametrize("command", ["version", "manifest", "check"])
def test_local_bundle_workers_make_no_registry_or_publication_calls(worker, tmp_path, command):
    bundle = Bundle(tmp_path / "bundle")
    driver = bundle.root / "network-boundary.py"
    driver.write_text(HTTP_DRIVER, encoding="utf-8")
    config = bundle.root / "http-config.json"
    write_json(config, {})
    http_log = bundle.root / "http.jsonl"
    publisher_log = bundle.root / "publishers.txt"
    bin_dir = bundle.root / "external publisher traps"
    for name in ("gh", "twine", "conda", "anaconda"):
        install_executable(bin_dir, name, PUBLISHER_TRAP)
    env = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        RELEASE_TEST_PUBLISHER_LOG=str(publisher_log),
    )
    for key in list(env):
        if any(word in key.upper() for word in ("TOKEN", "PASSWORD", "CREDENTIAL")):
            env.pop(key)
    args = {
        "version": ["version", bundle.version],
        "manifest": bundle.manifest_args(),
        "check": ["check", "--manifest", bundle.manifest, "--dist", bundle.dist],
    }[command]
    before = snapshot(bundle.dist)
    entry, outside = worker
    result = invoke(driver, entry, config, http_log, *args, cwd=outside, env=env)
    assert result.returncode == 0, diagnostic(result)
    assert not http_log.exists(), "This diagnostic command contacted an external registry"
    assert (
        not publisher_log.exists()
    ), "This diagnostic command called a publisher or approval tool"
    assert snapshot(bundle.dist) == before
