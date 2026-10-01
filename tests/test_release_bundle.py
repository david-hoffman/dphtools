"""S3/S4: independent manifest, archive, evidence and retained-byte checks."""

import json
import shutil
import tarfile

import pytest

from .test_release_support import (
    Bundle,
    DEPENDENCIES,
    PLATFORMS,
    STEPS,
    rejected,
    snapshot,
    write_json,
)
from . import test_release_version as version_cli
from .test_release_version import diagnostic, invoke

worker = version_cli.worker


@pytest.fixture
def bundle(worker, tmp_path):
    return Bundle(tmp_path / "bundle")


def generate(worker, bundle, args=None):
    entry, outside = worker
    return invoke(entry, *(bundle.manifest_args() if args is None else args), cwd=outside)


def ready_both(worker, bundle):
    """Require both intended interfaces before testing their rejection variants."""
    bundle.ready(worker)
    result = generate(worker, bundle)
    assert result.returncode == 0, "Valid manifest precondition: " + diagnostic(result)
    bundle.generated.unlink()


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0rc1"])
def test_manifest_binds_exact_artifacts_reports_notes_and_identity(worker, tmp_path, version):
    bundle = Bundle(tmp_path / "bundle", version)
    before = snapshot(bundle.dist)
    result = generate(worker, bundle)
    assert result.returncode == 0, diagnostic(result)
    payload = json.loads(bundle.generated.read_text(encoding="utf-8"))
    expected = bundle.expected()
    assert sorted(
        [
            {field: item[field] for field in ("filename", "size", "sha256")}
            for item in payload["files"]
        ],
        key=lambda item: item["filename"],
    ) == sorted(expected.pop("files"), key=lambda item: item["filename"])
    for field, value in expected.items():
        assert payload[field] == value, field
    assert not ({"token", "credentials", "manifest_sha256", "manifest_hash"} & payload.keys())
    assert snapshot(bundle.dist) == before
    bundle.manifest = bundle.generated
    bundle.ready(worker)
    assert snapshot(bundle.dist) == before


def test_check_accepts_independently_constructed_manifest_without_rebuilding(worker, bundle):
    before = snapshot(bundle.root)
    bundle.ready(worker)
    assert snapshot(bundle.root) == before


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("defect", ["missing", "malformed", "ambiguous"])
def test_manifest_rejects_missing_malformed_or_ambiguous_platform_evidence(
    worker, bundle, platform, defect
):
    good = generate(worker, bundle)
    assert good.returncode == 0, diagnostic(good)
    bundle.generated.unlink()
    report = bundle.reports / platform / "checks.json"
    if defect == "missing":
        report.unlink()
    elif defect == "malformed":
        report.write_text("{broken JSON", encoding="utf-8")
    else:
        duplicate = report.parent / "duplicate"
        duplicate.mkdir()
        shutil.copy2(report, duplicate / "checks.json")
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists(), "Invalid evidence cannot produce a ready manifest"


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("defect", ["missing", "nonzero", "duplicate"])
def test_manifest_requires_every_full_verification_step_once(worker, bundle, step, defect):
    good = generate(worker, bundle)
    assert good.returncode == 0, diagnostic(good)
    bundle.generated.unlink()
    path = bundle.reports / PLATFORMS[0] / "checks.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    target = next(item for item in report["steps"] if item["name"] == step)
    if defect == "missing":
        report["steps"].remove(target)
    elif defect == "nonzero":
        target["returncode"] = 19
    else:
        report["steps"].append(target.copy())
    write_json(path, report)
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_version", "0.9"),
        ("mode", "fast"),
        ("python", None),
        ("platform", None),
        ("steps", {}),
        ("failed", ["audit"]),
    ],
)
def test_manifest_rejects_invalid_report_schema_or_unsuccessful_summary(
    worker, bundle, field, value
):
    good = generate(worker, bundle)
    assert good.returncode == 0, diagnostic(good)
    bundle.generated.unlink()
    path = bundle.reports / PLATFORMS[1] / "checks.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    write_json(path, report)
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--source-sha", "abcd"),
        ("--source-sha", "A" * 40),
        ("--workflow-sha", "g" * 40),
        ("--workflow-sha", "0" * 39),
        ("--run-id", "0"),
        ("--run-id", "-1"),
        ("--run-id", "1.5"),
        ("--version", "v1.0.0"),
    ],
)
def test_manifest_rejects_invalid_identity_inputs(worker, bundle, option, value):
    good = generate(worker, bundle)
    assert good.returncode == 0, diagnostic(good)
    bundle.generated.unlink()
    args = bundle.manifest_args()
    args[args.index(option) + 1] = value
    rejected(generate(worker, bundle, args))
    assert not bundle.generated.exists()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "changes",
    [
        {"name": "unrelated-package"},
        {"version": "1.0.1"},
        {"python": None},
        {"python": ">=3.9"},
        {"dependencies": ()},
        {"dependencies": DEPENDENCIES[:-1]},
        {"dependencies": (*DEPENDENCIES, "unexpected")},
        {"dependencies": ("numpy>=2", *DEPENDENCIES[1:])},
    ],
)
def test_manifest_and_check_reject_metadata_defects_even_with_matching_hashes(
    worker, bundle, kind, changes
):
    ready_both(worker, bundle)
    # Version mutation changes metadata only; the filenames still declare 1.0.0.
    if "version" in changes:
        from .test_release_support import metadata, write_sdist, write_wheel

        writer = write_wheel if kind == "wheel" else write_sdist
        writer(getattr(bundle, kind), bundle.version, metadata(**changes))
    else:
        bundle.rewrite_archive(kind, **changes)
    bundle.refresh_manifest()
    before = snapshot(bundle.dist)
    rejected(bundle.run(worker, "check"))
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists()
    assert snapshot(bundle.dist) == before


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "member", ["../escaped.txt", "/absolute.txt", "C:/escape.txt", "..\\escape.txt"]
)
def test_unsafe_archive_paths_fail_before_any_extraction(worker, bundle, kind, member):
    ready_both(worker, bundle)
    bundle.rewrite_archive(kind, extras=[(member, b"unsafe marker")])
    bundle.refresh_manifest()
    before = snapshot(bundle.root)
    rejected(bundle.run(worker, "check"))
    rejected(generate(worker, bundle))
    assert snapshot(bundle.root) == before
    assert not (bundle.root.parent / "escaped.txt").exists()


def test_sdist_link_escaping_archive_is_rejected(worker, bundle):
    ready_both(worker, bundle)
    # Repack fixture members unchanged and add an escaping symlink.
    with tarfile.open(bundle.sdist) as archive:
        members = [(item, archive.extractfile(item).read()) for item in archive.getmembers()]
    from io import BytesIO

    with tarfile.open(bundle.sdist, "w:gz") as archive:
        for item, data in members:
            archive.addfile(item, BytesIO(data))
        link = tarfile.TarInfo("dphtools-1.0.0/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)
    bundle.refresh_manifest()
    rejected(bundle.run(worker, "check"))
    rejected(generate(worker, bundle))


@pytest.mark.parametrize(
    "field",
    [
        "document_version",
        "mode",
        "python",
        "platform",
        "steps",
        "failed",
        "measurement_limits",
    ],
)
def test_manifest_rejects_missing_canonical_report_fields(worker, bundle, field):
    ready_both(worker, bundle)
    path = bundle.reports / PLATFORMS[0] / "checks.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    del report[field]
    write_json(path, report)
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists()


@pytest.mark.parametrize(
    "field,value", [("command", 17), ("returncode", "0"), ("returncode", False)]
)
def test_manifest_rejects_malformed_step_records(worker, bundle, field, value):
    ready_both(worker, bundle)
    path = bundle.reports / PLATFORMS[0] / "checks.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["steps"][0][field] = value
    write_json(path, report)
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_manifest_and_check_reject_malformed_archives_with_matching_manifest_hashes(
    worker, bundle, kind
):
    ready_both(worker, bundle)
    getattr(bundle, kind).write_bytes(b"not a distribution archive")
    bundle.refresh_manifest()
    rejected(bundle.run(worker, "check"))
    rejected(generate(worker, bundle))


@pytest.mark.parametrize("defect", ["missing-wheel", "missing-sdist", "extra-wheel", "extra-file"])
def test_manifest_requires_exactly_the_two_distribution_files(worker, bundle, defect):
    ready_both(worker, bundle)
    if defect.startswith("missing-"):
        getattr(bundle, defect.removeprefix("missing-")).unlink()
    elif defect == "extra-wheel":
        shutil.copy2(bundle.wheel, bundle.dist / "dphtools-1.0.0-py2.py3-none-any.whl")
    else:
        (bundle.dist / "unexpected.txt").write_bytes(b"not in the release bundle")
    rejected(generate(worker, bundle))
    assert not bundle.generated.exists()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_multiple_archive_metadata_identities_are_ambiguous(worker, bundle, kind):
    from .test_release_support import metadata

    ready_both(worker, bundle)
    extra_name = (
        "another-1.0.0.dist-info/METADATA" if kind == "wheel" else "another-1.0.0/PKG-INFO"
    )
    bundle.rewrite_archive(kind, extras=[(extra_name, metadata())])
    bundle.refresh_manifest()
    rejected(bundle.run(worker, "check"))
    rejected(generate(worker, bundle))


@pytest.mark.parametrize(
    "defect",
    ["missing-wheel", "missing-sdist", "extra-wheel", "extra-file", "modified", "truncated"],
)
def test_check_rejects_missing_extra_or_changed_retained_files(worker, bundle, defect):
    bundle.ready(worker)
    if defect.startswith("missing-"):
        getattr(bundle, defect.removeprefix("missing-")).unlink()
    elif defect == "extra-wheel":
        shutil.copy2(bundle.wheel, bundle.dist / "dphtools-1.0.0-py2.py3-none-any.whl")
    elif defect == "extra-file":
        (bundle.dist / "unapproved.txt").write_bytes(b"not approved")
    elif defect == "modified":
        content = bytearray(bundle.wheel.read_bytes())
        content[len(content) // 2] ^= 1
        bundle.wheel.write_bytes(content)
    else:
        bundle.sdist.write_bytes(bundle.sdist.read_bytes()[:-5])
    before = snapshot(bundle.dist)
    rejected(bundle.run(worker, "check"))
    assert snapshot(bundle.dist) == before, "No silent rebuild or repair"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("repository", "other/repo"),
        ("version", "1.0.1"),
        ("version", "v1.0.0"),
        ("channel", "testpypi"),
        ("source_ref", "refs/heads/feature"),
        ("source_sha", "short"),
        ("source_sha", "A" * 40),
        ("workflow_sha", "bad"),
        ("run_id", 0),
        ("run_id", True),
        ("run_id", "731029"),
        ("files", []),
        ("verification", {}),
        ("notes", None),
    ],
)
def test_check_validates_manifest_fields(worker, bundle, field, value):
    bundle.ready(worker)
    payload = bundle.expected()
    payload[field] = value
    write_json(bundle.manifest, payload)
    rejected(bundle.run(worker, "check"))


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "repository",
        "version",
        "channel",
        "source_ref",
        "source_sha",
        "workflow_sha",
        "run_id",
        "files",
        "verification",
        "notes",
    ],
)
def test_check_rejects_missing_required_manifest_field(worker, bundle, field):
    bundle.ready(worker)
    payload = bundle.expected()
    del payload[field]
    write_json(bundle.manifest, payload)
    rejected(bundle.run(worker, "check"))


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "filename",
        "traversal",
        "size",
        "negative-size",
        "boolean-size",
        "digest",
        "uppercase-digest",
        "verification-digest",
        "malformed-json",
    ],
)
def test_check_rejects_file_record_and_digest_defects(worker, bundle, defect):
    bundle.ready(worker)
    payload = bundle.expected()
    record = payload["files"][0]
    if defect == "duplicate":
        payload["files"][1] = record.copy()
    elif defect in ("filename", "traversal"):
        record["filename"] = (
            "different.whl" if defect == "filename" else "../" + record["filename"]
        )
    elif defect in ("size", "negative-size", "boolean-size"):
        record["size"] = {"size": record["size"] + 1, "negative-size": -1, "boolean-size": True}[
            defect
        ]
    elif defect in ("digest", "uppercase-digest"):
        record["sha256"] = "0" * 64 if defect == "digest" else record["sha256"].upper()
    elif defect == "verification-digest":
        payload["verification"][PLATFORMS[0]] = "not-a-digest"
    write_json(bundle.manifest, payload)
    if defect == "malformed-json":
        bundle.manifest.write_text("{", encoding="utf-8")
    before = snapshot(bundle.dist)
    rejected(bundle.run(worker, "check"))
    assert snapshot(bundle.dist) == before
