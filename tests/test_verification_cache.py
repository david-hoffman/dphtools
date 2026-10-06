"""Concurrent public verifier invocations keep real pytest caches private."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from .test_verification_parallel import OVERLAPPING_TESTS, ParallelCommand


@pytest.mark.parametrize("mode", ["local", "distributed"])
def test_concurrent_pytest_caches_belong_to_each_invocation(tmp_path, mode):
    """Actual overlapping runs write distinct caches and retain exact execution."""
    command = ParallelCommand(tmp_path)
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    command.env["PARALLEL_TEST_BARRIER"] = str(barrier)
    source = OVERLAPPING_TESTS.replace(
        "def test_workers(number, tmp_path):",
        "def test_workers(number, tmp_path, pytestconfig):",
    ).replace(
        "'coverage': os.environ['COVERAGE_FILE']",
        "'coverage': os.environ['COVERAGE_FILE'], "
        "'cache': str(pytestconfig.getini('cache_dir'))",
    )
    (command.root / "tests/test_real.py").write_text(source)
    if mode == "local":
        result = command.parallel()
        assert result.returncode == 0, result.stdout + result.stderr
        directories = sorted(command.report.glob("worker-*"))
        reports = [command.report]
    else:
        result = command.run("collect", command.collect)
        assert result.returncode == 0, result.stdout + result.stderr
        collection = json.loads((command.collect / "checks.json").read_text())
        arguments = next(
            step["command"] for step in collection["steps"] if step["name"] == "collection"
        )
        assert "cache_dir=" + str(command.collect.resolve() / "pytest-cache") in arguments
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(command.shard, (0, 1)))
        for _, result in outcomes:
            assert result.returncode == 0, result.stdout + result.stderr
        directories = [directory for directory, _ in outcomes]
        result = command.aggregate(command.report, directories)
        assert result.returncode == 0, result.stdout + result.stderr
        reports = directories
    observations = [json.loads(path.read_text()) for path in sorted(barrier.glob("result-*"))]
    assert len(observations) == 2
    assert max(item["start"] for item in observations) < min(item["end"] for item in observations)
    caches = [Path(item["cache"]) for item in observations]
    assert len(set(caches)) == 2
    assert all(cache.is_absolute() and cache.is_dir() for cache in caches)
    assert set(caches) == {directory.resolve() / "pytest-cache" for directory in directories}
    assert not (command.root / ".pytest_cache").exists()
    execution = [
        entry
        for directory in reports
        for entry in json.loads(
            (directory / ("tests.json" if mode == "local" else "execution.json")).read_text()
        )["executions"]
    ]
    assert len(execution) == 9
    assert all(entry["outcome"] == "passed" for entry in execution)
    assert len({entry["node"] for entry in execution}) == 3
    coverage = json.loads((command.report / "coverage.json").read_text())
    assert coverage["totals"]["missing_lines"] == 0
    assert coverage["totals"]["missing_branches"] == 0


@pytest.mark.parametrize("count", [5, 7, 10])
def test_real_new_machine_counts_execute_and_aggregate_exactly(tmp_path, count):
    """The public machine counts retain every node and strict sealed aggregation."""
    command = ParallelCommand(tmp_path)
    (command.root / "tests/test_real.py").write_text(
        "import subprocess, sys, pytest, dphtools\n"
        f"@pytest.mark.parametrize('number', range({count}))\n"
        "def test_real(number):\n"
        "    assert dphtools.VALUE == 42\n"
        "    subprocess.run([sys.executable, '-c', "
        "'import dphtools.child; assert dphtools.child.VALUE == 17'], check=True)\n"
    )
    result = command.run("collect", command.collect, "--shard-count", str(count))
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(command.manifest.read_text())
    assert manifest["count"] == count
    assert len(manifest["assignments"]) == count
    assert all(manifest["assignments"])
    directories = [command.root / f"reports/shard-{index}" for index in range(count)]

    def launch(index):
        return command.run(
            "shard",
            directories[index],
            "--manifest",
            str(command.manifest),
            "--shard-count",
            str(count),
            "--shard-index",
            str(index),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(launch, range(count)))
    for result in outcomes:
        assert result.returncode == 0, result.stdout + result.stderr
    result = command.run(
        "aggregate",
        command.report,
        "--manifest",
        str(command.manifest),
        "--shard-count",
        str(count),
        "--shards",
        *map(str, directories),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    executed = [
        entry
        for directory in directories
        for entry in json.loads((directory / "execution.json").read_text())["executions"]
    ]
    assert len(executed) == 3 * (count + 1)
    assert sorted({entry["node"] for entry in executed}) == manifest["nodes"]
    assert all(entry["outcome"] == "passed" for entry in executed)
    coverage = json.loads((command.report / "coverage.json").read_text())
    assert coverage["totals"]["missing_lines"] == 0
    assert coverage["totals"]["missing_branches"] == 0
