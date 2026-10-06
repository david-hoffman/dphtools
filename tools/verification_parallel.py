"""Run bounded local pytest workers and combine their fresh execution evidence."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
from xml.etree import ElementTree

from verification_shards import (
    check_seal,
    completed_checks,
    duration_weights,
    partition,
    portable_inputs,
    read_json,
    require,
    sealed,
    test_step,
    validate_execution,
    write_json,
)


def merge_workers(directory, workers, assignments, nodes, identity, name="tests", selected=None):
    """Preserve fresh diagnostic reports and require exact successful node ownership."""
    directory = Path(directory)
    suites = ElementTree.Element("testsuites")
    junit_counts = []
    for index, worker in enumerate(workers):
        for offset, path in enumerate(sorted(worker.glob(".coverage*"))):
            if path.is_file():
                shutil.copyfile(path, directory / f".coverage.worker-{index}-{offset}")
    for worker in workers:
        junit = ElementTree.parse(worker / "pytest.xml")
        junit_counts.append(len(list(junit.iter("testcase"))))
        suites.extend(junit.iter("testsuite"))
    ElementTree.ElementTree(suites).write(
        directory / "pytest.xml", encoding="utf-8", xml_declaration=True
    )
    require(
        bool(workers) and len(workers) == len(assignments), "Worker count differs from assignment"
    )
    assigned = [node for assignment in assignments for node in assignment]
    require(
        sorted(assigned) == (nodes if selected is None else selected)
        and len(assigned) == len(set(assigned)),
        "Missing or duplicate assigned nodes",
    )
    reports = []
    durations = {}
    for worker, assignment, count in zip(workers, assignments, junit_counts):
        completed_checks(worker, "test-worker", portable_inputs(identity), [], name)
        require(bool(assignment) and count == len(assignment), "Worker JUnit node count differs")
        require(any(path.is_file() for path in worker.glob(".coverage*")), "No worker coverage")
        record = read_json(worker / (name + ".json"))
        check_seal(record)
        require(record["nodes"] == nodes, "Worker full collection differs")
        observed = validate_execution(record, assignment)
        require(record["durations"] == observed, "Worker recorded durations differ")
        reports.extend(record["executions"])
        durations.update(observed)
    write_json(
        directory / (name + ".json"),
        sealed({"nodes": nodes, "executions": reports, "exitstatus": 0, "durations": durations}),
    )


def parallel_test_step(
    run, workers, name="tests", settings=None, dependencies=("coverage-erase",)
):
    """Run serially or split complete real nodes among private subprocess reports."""
    require(type(workers) is int and 1 <= workers <= 8, "Workers must be between 1 and 8")
    settings = dict(settings or {})
    if workers == 1:
        test_step(run, name, settings, dependencies=dependencies)

        def validate_serial():
            try:
                record = read_json(run.directory / (name + ".json"))
                check_seal(record)
                nodes = record["nodes"]
                require(bool(nodes) and nodes == sorted(set(nodes)), "Invalid serial collection")
                require(nodes == settings.get("expected", nodes), "Serial collection differs")
                selected = settings.get("assigned", nodes)
                require(
                    bool(selected)
                    and len(selected) == len(set(selected))
                    and set(selected) <= set(nodes),
                    "Invalid serial node assignment",
                )
                observed = validate_execution(record, selected)
                require(record["durations"] == observed, "Serial recorded durations differ")
                junit = ElementTree.parse(run.directory / "pytest.xml")
                require(
                    len(list(junit.iter("testcase"))) == len(selected),
                    "Serial JUnit node count differs",
                )
                return 0
            except (KeyError, TypeError, IndexError, ElementTree.ParseError) as error:
                raise ValueError(f"Invalid serial artifact: {error}") from error

        if run.steps[-1]["state"] == "passed":
            run.run_step(name + "-validation", action=validate_serial, dependencies=(name,))
        return
    if "expected" not in settings:
        test_step(run, "collection", {}, dependencies=dependencies)
        dependencies = (*dependencies, "collection")

    def execute():
        from verification import VerificationRun

        try:
            if "expected" in settings:
                nodes = settings["expected"]
            else:
                collection = read_json(run.directory / "collection.json")
                check_seal(collection)
                require(collection["exitstatus"] == 0, "Collection failed")
                nodes = collection["nodes"]
            selected = settings.get("assigned", nodes)
            require(
                nodes == sorted(set(nodes)) and set(selected) <= set(nodes),
                "Invalid full or assigned collection",
            )
            previous = run.root / "tools/verification-durations.json"
            durations = {}
            if previous.is_file():
                record = read_json(previous)
                check_seal(record)
                durations = duration_weights(record, run.identity["environment"]["system"])
            assignments = partition(selected, durations, min(workers, len(selected)))
            with ExitStack() as stack:
                children = []
                temporaries = []
                for index in range(len(assignments)):
                    child = VerificationRun(
                        run.root, "test-worker", run.directory / f"worker-{index}"
                    )
                    temporary = Path(
                        stack.enter_context(tempfile.TemporaryDirectory(prefix="dphtools-worker-"))
                    ).resolve()
                    system_temporary = temporary / "os"
                    system_temporary.mkdir()
                    child.env = dict(
                        run.env,
                        COVERAGE_FILE=str(child.directory.resolve() / ".coverage"),
                        PYTHONDONTWRITEBYTECODE="1",
                        MPLCONFIGDIR=str(child.directory.resolve() / "matplotlib"),
                        TMPDIR=str(system_temporary),
                        TEMP=str(system_temporary),
                        TMP=str(system_temporary),
                    )
                    children.append(child)
                    temporaries.append(temporary / "pytest")

                def launch(index):
                    child = children[index]
                    test_step(
                        child,
                        name,
                        dict(settings, expected=nodes, assigned=assignments[index]),
                        temporary=temporaries[index],
                    )
                    return child.finish()

                with ThreadPoolExecutor(max_workers=len(children)) as pool:
                    list(pool.map(launch, range(len(children))))
                merge_workers(
                    run.directory,
                    [child.directory for child in children],
                    assignments,
                    nodes,
                    run.identity,
                    name=name,
                    selected=selected,
                )
            return 0
        except (KeyError, TypeError, IndexError, ElementTree.ParseError) as error:
            raise ValueError(f"Invalid worker artifact: {error}") from error

    run.run_step(name, action=execute, dependencies=dependencies)
