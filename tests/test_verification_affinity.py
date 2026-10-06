"""Advisory fixture affinity preserves real setup and exact execution ownership."""

import json
from pathlib import Path

import pytest

from .test_verification_parallel import ParallelCommand
from .test_verification_shards import shards_runtime

FIXTURE_TESTS = """
import os, subprocess, sys
from pathlib import Path
import pytest
import dphtools

@pytest.fixture(scope='module')
def shared():
    (Path(os.environ['AFFINITY_SETUP_RECORDS']) / str(os.getpid())).write_text('setup')
    return dphtools.VALUE

def test_parent(shared):
    assert shared == 42

def test_child(shared):
    assert shared == 42
    subprocess.run([sys.executable, '-c',
                    'import dphtools.child; assert dphtools.child.VALUE == 17'], check=True)
"""


@pytest.mark.parametrize("mode", ["local", "distributed"])
def test_real_fixture_affinity_keeps_one_setup_and_all_execution(tmp_path, mode):
    """Both public scheduling paths keep shared setup without omitting any test."""
    command = ParallelCommand(tmp_path)
    records = tmp_path / "setups"
    records.mkdir()
    command.env["AFFINITY_SETUP_RECORDS"] = str(records)
    (command.root / "tests/test_real.py").write_text(FIXTURE_TESTS)
    group = ["tests/test_real.py::test_child", "tests/test_real.py::test_parent"]
    seed = command.root / "tools/verification-durations.json"
    shards_runtime.write_json(
        seed,
        shards_runtime.sealed({"durations": {}, "groups": [group, ["unknown"], [group[0]]]}),
    )
    if mode == "local":
        result = command.parallel()
        assert result.returncode == 0, result.stdout + result.stderr
        execution = json.loads((command.report / "tests.json").read_text())
        assignments = [
            sorted({entry["node"] for entry in json.loads(path.read_text())["executions"]})
            for path in command.report.glob("worker-*/tests.json")
        ]
    else:
        result = command.run("collect", command.collect, "--durations", str(seed))
        assert result.returncode == 0, result.stdout + result.stderr
        manifest = json.loads(command.manifest.read_text())
        assignments = manifest["assignments"]
        directories = []
        for index in (0, 1):
            directory, result = command.shard(index)
            assert result.returncode == 0, result.stdout + result.stderr
            directories.append(directory)
        result = command.aggregate(command.report, directories)
        assert result.returncode == 0, result.stdout + result.stderr
        execution = {
            "nodes": manifest["nodes"],
            "executions": [
                entry
                for directory in directories
                for entry in json.loads((directory / "execution.json").read_text())["executions"]
            ],
        }
    assert len(list(records.iterdir())) == 1
    assert len(execution["nodes"]) == 3
    assert len(execution["executions"]) == 9
    assert len({entry["node"] for entry in execution["executions"]}) == 3
    assert any(set(group) <= set(assignment) for assignment in assignments)
    coverage = json.loads((command.report / "coverage.json").read_text())
    assert coverage["totals"]["missing_lines"] == 0
    assert coverage["totals"]["missing_branches"] == 0


@pytest.mark.parametrize("selected,count", [(["a", "b"], 1), (["a", "b", "c"], 3)])
def test_advisory_affinity_handles_partial_groups_and_required_worker_count(selected, count):
    """Stale or overly cohesive hints cannot omit nodes or empty a required worker."""
    assignments = shards_runtime.partition(selected, {}, count, [["a", "b", "outside"], ["a"], []])
    assert len(assignments) == count
    assert all(assignments)
    assert sorted(node for assignment in assignments for node in assignment) == selected
