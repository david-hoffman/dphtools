"""Freeze and verify a bounded partition of real pytest/doctest execution."""

import json
import math
import os
from pathlib import Path
import shutil
from xml.etree import ElementTree

from verification_inputs import CHECK_VERSION, digest, file_hash, git_identity

SHARD_COUNT = 2
QUALITY_STEPS = {
    "preflight",
    "format",
    "lint",
    "docstrings",
    "types",
    "audit",
    "build",
    "wheel-artifacts",
}


def require(condition, message):
    """Reject unverifiable execution with a concrete diagnostic."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    """Read JSON artifacts using one explicit encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write a complete ordinary JSON artifact atomically."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sealed(value):
    """Bind every field of an ordinary receipt to its digest."""
    return dict(value, digest=digest(value))


def check_seal(value, field="digest"):
    """Reject stale or changed manifest/receipt metadata."""
    require(
        value.get(field) == digest({k: v for k, v in value.items() if k != field}),
        "Artifact digest mismatch",
    )


def portable_inputs(value):
    """Remove only machine-local executable, host, and ambient identity fields."""
    identity = dict(value)
    identity["environment"] = {
        key: value
        for key, value in identity["environment"].items()
        if key not in ("executable", "executable_sha256", "platform", "ambient_environment_digest")
    }
    return identity


def portable_identity(run):
    """Bind portable exact inputs, Versioneer context, and the current CI attempt."""
    identity = portable_inputs(run.identity)
    identity["git"] = git_identity(run.root)
    identity["run"] = {
        "id": os.environ.get("GITHUB_RUN_ID", "local"),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
    }
    return identity


def duration_weights(record, system):
    """Select measured host costs, retaining universal duration records as fallback."""
    return record.get("platform_durations", {}).get(system, record["durations"])


def partition(nodes, durations, count=SHARD_COUNT, groups=()):
    """Assign every real node once to deterministic duration-balanced workers."""
    require(bool(nodes) and nodes == sorted(set(nodes)), "Collection is empty or duplicate")
    require(type(count) is int and 1 <= count <= len(nodes), "Invalid worker count")
    require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in durations.values()
        ),
        "Invalid recorded durations",
    )
    pending = set(nodes)
    batches = []
    for group in groups:
        batch = sorted(pending.intersection(group))
        if batch:
            batches.append(batch)
            pending.difference_update(batch)
    batches.extend([node] for node in sorted(pending))
    # Advisory affinity cannot prevent the required number of nonempty workers.
    if len(batches) < count:
        batches = [[node] for node in nodes]
    costs = {tuple(batch): sum(durations.get(node, 1.0) for node in batch) for batch in batches}
    assignments = [[] for _ in range(count)]
    totals = [0.0 for _ in range(count)]
    for batch in sorted(batches, key=lambda batch: (-costs[tuple(batch)], batch)):
        index = min(
            range(count), key=lambda index: (totals[index], len(assignments[index]), index)
        )
        assignments[index].extend(batch)
        totals[index] += costs[tuple(batch)]
    require(all(assignments), "Not enough collected nodes for nonempty workers")
    return [sorted(assignment) for assignment in assignments]


def artifact(directory, record):
    """Resolve only retained private artifact paths and verify their actual bytes."""
    directory = Path(directory).resolve()
    path = (directory / record["path"]).resolve()
    require(path.is_relative_to(directory), "Artifact escapes its report directory")
    require(
        path.is_file() and file_hash(path) == record["sha256"],
        f"Changed or absent artifact: {record['path']}",
    )
    return path


def file_record(directory, path):
    """Record a private artifact's path and exact hash."""
    return {"path": path.relative_to(directory).as_posix(), "sha256": file_hash(path)}


def completed_checks(directory, mode, identity, steps, final):
    """Require the invocation's final successful receipt, including its last log."""
    checks = read_json(directory / "checks.json")
    check_seal(checks, "receipt_digest")
    require(checks["document_version"] == CHECK_VERSION, "Unsupported invocation version")
    require(
        checks["mode"] == mode and checks["complete"] is True and checks["outcome"] == "passed",
        "Invocation is missing, failed, or incomplete",
    )
    require(
        portable_inputs(checks["identity"])
        == {key: value for key, value in identity.items() if key not in ("git", "run")},
        "Invocation inputs differ",
    )
    require(
        checks["identity_digest"] == digest(checks["identity"]),
        "Invocation identity digest differs",
    )
    require(
        checks["steps"][:-1] == steps and checks["steps"][-1]["name"] == final,
        "Invocation steps differ or are incomplete",
    )
    require(
        all(step["state"] == "passed" and step["returncode"] == 0 for step in checks["steps"]),
        "Invocation steps failed",
    )
    for step in checks["steps"]:
        require(
            step["input_digest"] == digest(step["input_identity"]), "Step identity digest differs"
        )
        artifact(directory, step["log"])


class NodeRecorder:
    """Record actual collection and every setup/call/teardown report from pytest."""

    def __init__(self, settings):
        self.settings = settings
        self.nodes = []
        self.executions = []

    def pytest_collection_modifyitems(self, session, config, items):
        """Verify the full collection before selecting assigned execution nodes."""
        import pytest

        self.nodes = sorted(item.nodeid for item in items)
        if not self.nodes or len(self.nodes) != len(set(self.nodes)):
            raise pytest.UsageError("Empty or duplicate full collection")
        if "expected" in self.settings:
            if self.nodes != self.settings["expected"]:
                raise pytest.UsageError("Current collection differs from manifest")
        if "assigned" in self.settings:
            assigned = set(self.settings["assigned"])
            config.hook.pytest_deselected(
                items=[item for item in items if item.nodeid not in assigned]
            )
            items[:] = [item for item in items if item.nodeid in assigned]

    def pytest_runtest_logreport(self, report):
        """Retain every actual lifecycle phase, outcome, and observed duration."""
        self.executions.append(
            {
                "node": report.nodeid,
                "phase": report.when,
                "outcome": report.outcome,
                "duration": report.duration,
            }
        )

    def pytest_sessionfinish(self, session, exitstatus):
        """Retain completed execution data even when pytest reports failure."""
        durations = {}
        for report in self.executions:
            durations[report["node"]] = durations.get(report["node"], 0.0) + report["duration"]
        write_json(
            self.settings["output"],
            sealed(
                {
                    "nodes": self.nodes,
                    "executions": self.executions,
                    "exitstatus": int(exitstatus),
                    "durations": durations,
                }
            ),
        )


def pytest_configure(config):
    """Install the recorder only for explicitly configured verifier subprocesses."""
    settings = os.environ.get("VERIFICATION_SHARD_CONFIG")
    if settings:
        config.pluginmanager.register(
            NodeRecorder(read_json(settings)), "verification-node-recorder"
        )


def test_step(run, name, settings, dependencies=(), temporary=None):
    """Run real pytest collection/execution with a private recorder configuration."""
    settings = dict(settings, output=str(run.directory / (name + ".json")))
    configuration = run.directory / (name + "-config.json")
    write_json(configuration, settings)
    env = dict(run.env, VERIFICATION_SHARD_CONFIG=str(configuration))
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    arguments = [
        "pytest",
        "-p",
        "verification_shards",
        f"--rootdir={run.root}",
        "--doctest-modules",
        "dphtools",
        "tests",
        "-ra",
        "-o",
        f"cache_dir={run.directory.resolve() / 'pytest-cache'}",
    ]
    if name == "collection":
        arguments.append("--collect-only")
    else:
        arguments.append(f"--junitxml={run.directory / 'pytest.xml'}")
        temporary = run.directory / "temporary" if temporary is None else temporary
        arguments.append(f"--basetemp={temporary}")
        arguments = ["coverage", "run", "-m", *arguments]
    run.module(name, arguments, dependencies=dependencies, env=env)


def validate_execution(record, assigned):
    """Require one successful setup/call/teardown for every assigned real node."""
    require(record["exitstatus"] == 0, "pytest execution failed")
    phases = {node: [] for node in assigned}
    durations = {node: 0.0 for node in assigned}
    for report in record["executions"]:
        require(
            report["node"] in phases and report["outcome"] == "passed",
            "Foreign, failed, or skipped execution",
        )
        require(
            type(report["duration"]) in (int, float)
            and math.isfinite(report["duration"])
            and report["duration"] >= 0,
            "Invalid execution duration",
        )
        phases[report["node"]].append(report["phase"])
        durations[report["node"]] += report["duration"]
    require(
        all(value == ["setup", "call", "teardown"] for value in phases.values()),
        "Missing or duplicate node execution",
    )
    return durations


def load_manifest(run, path, count=SHARD_COUNT):
    """Validate retained collection gates, partition, wheel, and current identity."""
    manifest = read_json(path)
    check_seal(manifest)
    require(
        manifest["identity"] == portable_identity(run),
        "Manifest inputs/platform/current run differ",
    )
    actual_count = manifest.get("count")
    require(type(actual_count) is int and actual_count in (2, 4, 5, 7, 8), "Invalid shard count")
    require(actual_count == count, "Manifest shard count differs from requested count")
    assignments = manifest["assignments"]
    require(len(assignments) == actual_count and all(assignments), "Invalid shard count")
    require(manifest["nodes"] == sorted(set(manifest["nodes"])), "Duplicate collection nodes")
    combined = [node for assignment in assignments for node in assignment]
    require(
        sorted(combined) == manifest["nodes"] and len(combined) == len(set(combined)),
        "Missing or duplicate assigned nodes",
    )
    quality = read_json(artifact(path.parent, manifest["quality"]))
    check_seal(quality)
    require(quality["identity"] == manifest["identity"], "Quality inputs differ")
    require(
        QUALITY_STEPS <= {step["name"] for step in quality["steps"]},
        "Missing collection quality gate",
    )
    require(
        all(step["state"] == "passed" for step in quality["steps"]),
        "Collection quality gate failed",
    )
    completed_checks(path.parent, "collect", manifest["identity"], quality["steps"], "manifest")
    for step in quality["steps"]:
        artifact(path.parent, step["log"])
    artifact(path.parent, manifest["wheel"])
    return manifest


def collect(run, arguments):
    """Run quality/build once and freeze actual pytest/doctest collection."""
    from verification import fast_checks, preflight, prepare_checks

    preflight(run)
    fast_checks(run)
    wheels, _ = prepare_checks(run)
    test_step(run, "collection", {}, dependencies=("wheel-artifacts",))

    def freeze():
        record = read_json(run.directory / "collection.json")
        require(record["exitstatus"] == 0, "Collection failed")
        count = getattr(arguments, "shard_count", SHARD_COUNT)
        require(type(count) is int and count in (2, 4, 5, 7, 8), "Invalid shard count")
        durations = {}
        groups = []
        for path in arguments.durations:
            previous = read_json(path)
            check_seal(previous)
            durations.update(duration_weights(previous, run.identity["environment"]["system"]))
            groups.extend(previous.get("groups", []))
        identity = portable_identity(run)
        quality = sealed({"identity": identity, "steps": run.steps})
        write_json(run.directory / "quality.json", quality)
        path = arguments.manifest or run.directory / "manifest.json"
        require(
            path.parent.resolve() == run.directory.resolve(),
            "Manifest must remain with its collection artifacts",
        )
        write_json(
            path,
            sealed(
                {
                    "identity": identity,
                    "count": count,
                    "nodes": record["nodes"],
                    "assignments": partition(record["nodes"], durations, count, groups),
                    "quality": file_record(run.directory, run.directory / "quality.json"),
                    "wheel": file_record(run.directory, wheels[0]),
                }
            ),
        )
        return 0

    run.run_step("manifest", action=freeze, dependencies=("collection",))


def shard(run, arguments):
    """Verify full collection and execute only assigned nodes in private paths."""
    from verification_parallel import parallel_test_step

    manifest = {}

    def inputs():
        loaded = load_manifest(
            run, arguments.manifest, getattr(arguments, "shard_count", SHARD_COUNT)
        )
        require(
            type(arguments.shard_index) is int and arguments.shard_index in range(loaded["count"]),
            "Invalid shard index",
        )
        manifest.update(loaded)
        return 0

    run.run_step("shard-inputs", action=inputs)
    assigned = manifest["assignments"][arguments.shard_index] if manifest else []
    test_step(
        run, "collection", {"expected": manifest.get("nodes", [])}, dependencies=("shard-inputs",)
    )
    wheels = [arguments.manifest.parent / manifest["wheel"]["path"]] if manifest else []
    run.module(
        "install",
        [
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(run.directory / "install"),
            *map(str, wheels),
        ],
        dependencies=("collection",),
    )
    parallel_test_step(
        run,
        getattr(arguments, "workers", 1),
        name="execution",
        settings={"expected": manifest.get("nodes", []), "assigned": assigned},
        dependencies=("install",),
    )

    def receipt():
        record = read_json(run.directory / "execution.json")
        require(record["nodes"] == manifest["nodes"], "Executed collection differs")
        durations = validate_execution(record, assigned)
        raw = sorted(path for path in run.directory.glob(".coverage*") if path.is_file())
        require(bool(raw), "No actual parent/child coverage data")
        files = [run.directory / "execution.json", run.directory / "pytest.xml", *raw]
        files.extend(run.directory / step["log"]["path"] for step in run.steps)
        write_json(
            run.directory / "shard.json",
            sealed(
                {
                    "identity": manifest["identity"],
                    "manifest": manifest["digest"],
                    "index": arguments.shard_index,
                    "count": manifest["count"],
                    "nodes": assigned,
                    "durations": durations,
                    "steps": run.steps,
                    "artifacts": [file_record(run.directory, path) for path in files],
                }
            ),
        )
        return 0

    run.run_step("shard-receipt", action=receipt, dependencies=("execution",))


def aggregate(run, arguments):
    """Require matching complete shards per OS, then combine real reports/data."""
    from verification import process_coverage

    manifests = {}

    def inputs():
        manifests.update(
            load_manifest(run, arguments.manifest, getattr(arguments, "shard_count", SHARD_COUNT))
        )
        return 0

    run.run_step("aggregate-inputs", action=inputs)
    test_step(
        run,
        "collection",
        {"expected": manifests.get("nodes", [])},
        dependencies=("aggregate-inputs",),
    )

    def merge():
        count = manifests["count"]
        require(len(arguments.shards) == count, "Missing or duplicate shards")
        indices = set()
        executed = []
        suites = ElementTree.Element("testsuites")
        for directory in arguments.shards:
            receipt = read_json(directory / "shard.json")
            check_seal(receipt)
            require(
                receipt["identity"] == manifests["identity"]
                and receipt["manifest"] == manifests["digest"],
                "Shard inputs/platform/manifest/current run differ",
            )
            index = receipt["index"]
            require(
                type(index) is int
                and index in range(count)
                and index not in indices
                and type(receipt["count"]) is int
                and receipt["count"] == count,
                "Duplicate or invalid shard index",
            )
            indices.add(index)
            require(
                receipt["nodes"] == manifests["assignments"][index],
                "Shard nodes differ from assignment",
            )
            require(
                {"shard-inputs", "collection", "install", "execution"}
                <= {step["name"] for step in receipt["steps"]}
                and all(step["state"] == "passed" for step in receipt["steps"]),
                "Failed or incomplete shard steps",
            )
            completed_checks(
                directory, "shard", receipt["identity"], receipt["steps"], "shard-receipt"
            )
            paths = [artifact(directory, record) for record in receipt["artifacts"]]
            names = [path.name for path in paths]
            require(
                len(names) == len(set(names)) and {"execution.json", "pytest.xml"} <= set(names),
                "Missing or duplicate shard reports",
            )
            actual = read_json(directory / "execution.json")
            require(actual["nodes"] == manifests["nodes"], "Shard collection differs")
            require(
                validate_execution(actual, receipt["nodes"]) == receipt["durations"],
                "Recorded durations differ",
            )
            executed.extend(receipt["nodes"])
            raw = [path for path in paths if path.name.startswith(".coverage")]
            require(
                bool(raw)
                and {path.name for path in raw}
                == {path.name for path in directory.glob(".coverage*") if path.is_file()},
                "Missing actual raw coverage",
            )
            for offset, path in enumerate(raw):
                shutil.copyfile(path, run.directory / f".coverage.shard-{index}-{offset}")
            junit = ElementTree.parse(directory / "pytest.xml")
            cases = list(junit.iter("testcase"))
            require(len(cases) == len(receipt["nodes"]), "Shard JUnit node count differs")
            for suite in junit.iter("testsuite"):
                suites.append(suite)
        require(
            sorted(executed) == manifests["nodes"] and len(executed) == len(set(executed)),
            "Missing or duplicate executed nodes",
        )
        ElementTree.ElementTree(suites).write(
            run.directory / "pytest.xml", encoding="utf-8", xml_declaration=True
        )
        return 0

    run.run_step("coverage-data", action=merge, dependencies=("collection",))
    process_coverage(run)


def run_sharded(run, arguments):
    """Dispatch three narrow modes while retaining normal receipt/failure handling."""
    try:
        {"collect": collect, "shard": shard, "aggregate": aggregate}[run.mode](run, arguments)
    except (KeyError, TypeError, AttributeError, IndexError, ElementTree.ParseError) as error:
        run.run_step("shard-format", action=lambda: 1, dependencies=())
        print(f"Invalid shard artifact: {error}", flush=True)
