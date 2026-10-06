"""Configurable physical shards retain exact real execution and report gates."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from .test_verification_shards import (
    ROOT,
    ShardedCommand,
    _reseal,
    completed_shards,
    shards_runtime,
    verification,
)


def distributed_command(command, count, mode, directory, *arguments):
    """Invoke the public CLI with an explicitly selected physical shard count."""
    return command.run(mode, directory, "--shard-count", str(count), *arguments)


@pytest.fixture(params=(4, 8))
def distributed_shards(tmp_path_factory, request):
    """Build each real four/eight-shard proof only for its positive scenario."""
    count = request.param
    command = ShardedCommand(tmp_path_factory.mktemp(f"distributed-{count}"))
    test = command.root / "tests/test_real.py"
    with test.open("a") as stream:
        stream.write(
            "\nimport pytest\n"
            "@pytest.mark.parametrize('number', range(16))\n"
            "def test_assigned_number(number):\n    assert number >= 0\n"
        )
    result = distributed_command(command, count, "collect", command.collect)
    assert result.returncode == 0, result.stdout + result.stderr
    directories = []
    for index in range(count):
        directory = command.root / f"reports/shard-{index}"
        result = distributed_command(
            command,
            count,
            "shard",
            directory,
            "--manifest",
            str(command.manifest),
            "--shard-index",
            str(index),
            "--workers",
            "4",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        directories.append(directory)
    return command, count, directories


def test_real_distributed_roundtrip_preserves_complete_nodes_and_child_coverage(
    distributed_shards, tmp_path
):
    command, count, directories = distributed_shards
    manifest = json.loads(command.manifest.read_text())
    assert manifest["count"] == count
    assert len(manifest["assignments"]) == count
    assert len(manifest["nodes"]) == 19
    assigned = [node for group in manifest["assignments"] for node in group]
    assert sorted(assigned) == manifest["nodes"] and len(assigned) == len(set(assigned))
    executed = []
    for index, directory in enumerate(directories):
        receipt = json.loads((directory / "shard.json").read_text())
        assert receipt["count"] == count and receipt["index"] == index
        assert receipt["manifest"] == manifest["digest"]
        actual = json.loads((directory / "execution.json").read_text())
        assert actual["nodes"] == manifest["nodes"]
        assert actual["exitstatus"] == 0
        assert len(list(directory.glob("worker-*/checks.json"))) == min(4, len(receipt["nodes"]))
        for node in receipt["nodes"]:
            phases = [report for report in actual["executions"] if report["node"] == node]
            assert sorted(report["phase"] for report in phases) == ["call", "setup", "teardown"]
            assert all(report["outcome"] == "passed" for report in phases)
        executed.extend(receipt["nodes"])
    assert sorted(executed) == manifest["nodes"] and len(executed) == len(set(executed))
    report = tmp_path / "aggregate"
    result = distributed_command(
        command,
        count,
        "aggregate",
        report,
        "--manifest",
        str(command.manifest),
        "--shards",
        *map(str, directories),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(ElementTree.parse(report / "pytest.xml").iter("testcase"))) == 19
    coverage = json.loads((report / "coverage.json").read_text())
    assert coverage["files"]["dphtools/child.py"]["executed_lines"] == [2]
    assert coverage["totals"]["missing_lines"] == 0
    assert coverage["totals"]["missing_branches"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "count-missing",
        "count-bool",
        "count-float",
        "count-string",
        "count-unsupported",
        "count-assignments",
        "assignment-empty",
        "assignment-missing",
        "assignment-duplicate",
        "requested-count",
    ],
)
def test_resealed_distributed_manifest_faults_block_collection_and_install(
    completed_shards, tmp_path, monkeypatch, fault
):
    command, _ = completed_shards
    count = 2
    collection = tmp_path / "collection"
    shutil.copytree(command.collect, collection)
    manifest = collection / "manifest.json"

    def mutate(value):
        if fault == "count-missing":
            value.pop("count")
        elif fault.startswith("count-"):
            value["count"] = {
                "count-bool": True,
                "count-float": float(count),
                "count-string": str(count),
                "count-unsupported": 3,
                "count-assignments": 8 if count == 4 else 4,
            }[fault]
        elif fault == "assignment-empty":
            value["assignments"][1].extend(value["assignments"][0])
            value["assignments"][0].clear()
        elif fault == "assignment-missing":
            value["assignments"][0].pop()
        elif fault == "assignment-duplicate":
            value["assignments"][0].append(value["assignments"][1][0])

    _reseal(manifest, mutate)
    selected_count = (
        8
        if fault == "requested-count"
        else (8 if count == 4 else 4) if fault == "count-assignments" else count
    )
    report = tmp_path / "blocked-shard"
    result = distributed_command(
        command,
        selected_count,
        "shard",
        report,
        "--manifest",
        str(manifest),
        "--shard-index",
        "0",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads((report / "checks.json").read_text())
    assert (
        next(step for step in receipt["steps"] if step["name"] == "shard-inputs")["state"]
        == "failed"
    )
    assert (
        next(step for step in receipt["steps"] if step["name"] == "collection")["state"]
        == "blocked"
    )
    assert (
        next(step for step in receipt["steps"] if step["name"] == "install")["state"] == "blocked"
    )
    assert not (report / "shard.json").exists()
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = verification.VerificationRun(command.root, "shard", tmp_path / "api-shard")
    shards_runtime.run_sharded(
        run,
        SimpleNamespace(manifest=manifest, shard_count=selected_count, shard_index=0),
    )
    assert run.finish() == 1
    assert next(step for step in run.steps if step["name"] == "shard-inputs")["state"] == "failed"
    assert next(step for step in run.steps if step["name"] == "install")["state"] == "blocked"


@pytest.mark.parametrize("index", [False, 8])
def test_distributed_interface_rejects_index_before_launch(
    completed_shards, tmp_path, monkeypatch, index
):
    command, _ = completed_shards
    count = 2
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    report = tmp_path / "invalid-index"
    run = verification.VerificationRun(command.root, "shard", report)
    shards_runtime.run_sharded(
        run,
        SimpleNamespace(manifest=command.manifest, shard_count=count, shard_index=index),
    )
    assert run.finish() == 1
    assert next(step for step in run.steps if step["name"] == "shard-inputs")["state"] == "failed"
    assert next(step for step in run.steps if step["name"] == "collection")["state"] == "blocked"
    assert next(step for step in run.steps if step["name"] == "install")["state"] == "blocked"
    assert not (report / "execution.json").exists()


@pytest.mark.parametrize("count", [True, 3])
def test_collection_interface_rejects_unsupported_count(tmp_path, monkeypatch, count):
    command = ShardedCommand(tmp_path)
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = verification.VerificationRun(command.root, "collect", command.collect)
    shards_runtime.run_sharded(
        run, SimpleNamespace(manifest=None, durations=[], shard_count=count)
    )
    assert run.finish() == 1
    assert next(step for step in run.steps if step["name"] == "manifest")["state"] == "failed"
    assert not command.manifest.exists()


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "count", "count-float", "index", "nodes", "requested-count"]
)
def test_distributed_aggregation_rejects_incomplete_or_resealed_foreign_ownership(
    completed_shards, tmp_path, monkeypatch, fault
):
    command, originals = completed_shards
    count = 2
    directories = [tmp_path / f"shard-{index}" for index in range(count)]
    for original, directory in zip(originals, directories):
        shutil.copytree(original, directory)
    if fault == "missing":
        directories.pop()
    elif fault == "duplicate":
        directories[1] = directories[0]
    elif fault != "requested-count":

        def mutate(value):
            if fault == "count":
                value["count"] = 8 if count == 4 else 4
            elif fault == "count-float":
                value["count"] = float(count)
            elif fault == "index":
                value["index"] = count
            else:
                value["nodes"].append("tests/test_real.py::foreign")

        _reseal(directories[0] / "shard.json", mutate)
    report = tmp_path / "aggregate"
    result = distributed_command(
        command,
        4 if fault == "requested-count" else count,
        "aggregate",
        report,
        "--manifest",
        str(command.manifest),
        "--shards",
        *map(str, directories),
    )
    assert result.returncode == 1, result.stdout + result.stderr
    checks = json.loads((report / "checks.json").read_text())
    assert checks["complete"] is True and checks["outcome"] == "failed"
    assert not (report / "coverage.json").exists()
    for key, value in command.env.items():
        monkeypatch.setenv(key, value)
    run = verification.VerificationRun(command.root, "aggregate", tmp_path / "api-aggregate")
    shards_runtime.run_sharded(
        run,
        SimpleNamespace(
            manifest=command.manifest,
            shard_count=4 if fault == "requested-count" else count,
            shards=directories,
        ),
    )
    assert run.finish() == 1
    assert not (run.directory / "coverage.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("collect", "--shard-count", "3"),
        ("collect", "--shard-count", "1"),
        ("shard", "--manifest", "missing", "--shard-count", "4", "--shard-index", "4"),
        ("shard", "--manifest", "missing", "--shard-count", "8", "--shard-index", "8"),
        ("shard", "--manifest", "missing", "--shard-count", "8", "--shard-index", "-1"),
    ],
)
def test_configurable_shard_cli_rejects_invalid_count_or_index_before_report(tmp_path, arguments):
    report = tmp_path / "unused"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/verification.py"),
            *arguments,
            "--report-dir",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2 and "error:" in result.stderr
    assert not report.exists()
