"""Independent post-implementation S3/S4/S5 evidence from approved public policy.

Mocks represent external Git/GitHub/HTTPS only. Hosted approval and ordering need
separate D/setup evidence. No receipt in this suite authorizes publication.
"""

import base64
import json
from pathlib import Path
import shutil
import stat

import pytest

from .release_workflow_support import ARTIFACT_ID, PREFIX, Workflow, BOOTSTRAP
from .test_release_reconcile import Registry
from .test_release_smoke import (
    PROCESS_DRIVER,
    Smoke,
    clean_env,
    real_bundle,
)
from .test_release_support import (
    Bundle,
    PLATFORMS,
    RUN_ID,
    SOURCE_SHA,
    WORKFLOW_SHA,
    digest,
    rejected,
    snapshot,
    write_json,
)
from .test_release_version import diagnostic, invoke, worker


def require_success(result):
    """Keep raw child streams out of pytest's assertion operand rendering."""
    code = result.returncode
    message = diagnostic(result)
    assert code == 0, message


@pytest.fixture
def workflow(tmp_path):
    return Workflow(tmp_path)


def assert_read_only(workflow):
    assert all(call["method"] == "GET" for call in workflow.calls())


def assert_published_after_uploads(workflow):
    """Observe external Release state and actual publication after uploads."""
    release = workflow.config["routes"]["GET " + PREFIX + "releases/tags/1.0.0"]["json"]
    assert release["draft"] is False
    calls = workflow.calls()
    publications = [
        index
        for index, call in enumerate(calls)
        if call["method"] == "PATCH"
        and call["endpoint"] == PREFIX + "releases/491"
        and json.loads(base64.b64decode(call["body"])).get("draft") is False
    ]
    assert publications, "Draft creation and uploads alone do not publish a Release"
    uploads = [
        index
        for index, call in enumerate(calls)
        if call["method"] == "POST" and "/assets?name=" in call["endpoint"]
    ]
    assert uploads and max(uploads) < min(publications)


def test_normal_resolve_freezes_trusted_main_identity(workflow):
    result = workflow.run("resolve", "--version", "1.0.0")
    require_success(result)
    assert workflow.outputs() == {
        "version": "1.0.0",
        "channel": "pypi",
        "source_sha": SOURCE_SHA,
        "workflow_sha": SOURCE_SHA,
        "origin_run": str(RUN_ID),
        "resume": "false",
    }
    assert_read_only(workflow)


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_REPOSITORY", "other/repo"),
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("TRUSTED_WORKFLOW_SHA", "short"),
    ],
)
def test_resolve_rejects_untrusted_context_after_valid_control(workflow, key, value):
    workflow.ready("resolve", "--version", "1.0.0")
    workflow.env[key] = value
    rejected(workflow.run("resolve", "--version", "1.0.0"))
    assert_read_only(workflow)
    assert not workflow.output.exists()


@pytest.mark.parametrize("value", ["v1.0.0", "1.0.0.dev1", "1.0.0+local"])
def test_resolve_rejects_unapproved_versions(workflow, value):
    workflow.ready("resolve", "--version", "1.0.0")
    rejected(workflow.run("resolve", "--version", value))
    assert_read_only(workflow)
    assert not workflow.output.exists()


def recovery_context(workflow):
    workflow.env.update(TRUSTED_WORKFLOW_SHA=WORKFLOW_SHA, GITHUB_RUN_ID=str(RUN_ID + 1))
    workflow.config["git"]["head"] = WORKFLOW_SHA


def test_recovery_uses_exact_original_run_artifact_and_revision(workflow):
    recovery_context(workflow)
    result = workflow.run("resolve", "--version", "1.0.0", "--resume", str(RUN_ID))
    require_success(result)
    assert workflow.outputs() == {
        "version": "1.0.0",
        "channel": "pypi",
        "source_sha": SOURCE_SHA,
        "workflow_sha": SOURCE_SHA,
        "origin_run": str(RUN_ID),
        "resume": "true",
        "artifact_id": str(ARTIFACT_ID),
        "artifact_digest": workflow.artifact["digest"],
    }
    assert_read_only(workflow)
    assert {call["endpoint"] for call in workflow.calls()} <= {
        PREFIX + f"actions/runs/{RUN_ID}",
        PREFIX + f"actions/runs/{RUN_ID}/artifacts?per_page=100",
        PREFIX + f"actions/artifacts/{ARTIFACT_ID}",
        PREFIX + f"actions/artifacts/{ARTIFACT_ID}/zip",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "run-id",
        "repository",
        "workflow",
        "branch",
        "event",
        "expired",
        "missing",
        "artifact-run",
        "artifact-source",
        "changed-helper",
        "source-off-main",
    ],
)
def test_recovery_rejects_identity_and_retention_conflicts(workflow, fault):
    recovery_context(workflow)
    workflow.ready("resolve", "--version", "1.0.0", "--resume", str(RUN_ID))
    run = workflow.config["routes"]["GET " + PREFIX + f"actions/runs/{RUN_ID}"]["json"]
    fields = {
        "run-id": ("id", RUN_ID + 2),
        "repository": ("repository", {"full_name": "other/repo"}),
        "workflow": ("path", ".github/workflows/other.yml"),
        "branch": ("head_branch", "feature"),
        "event": ("event", "pull_request"),
    }
    if fault in fields:
        key, value = fields[fault]
        run[key] = value
    elif fault == "expired":
        workflow.artifact["expired"] = True
        workflow.reset_artifact()
    elif fault == "missing":
        workflow.route(f"actions/runs/{RUN_ID}/artifacts?per_page=100", {"artifacts": []})
    elif fault.startswith("artifact-"):
        workflow.artifact["workflow_run"]["id" if fault == "artifact-run" else "head_sha"] = (
            RUN_ID + 2 if fault == "artifact-run" else WORKFLOW_SHA
        )
        workflow.reset_artifact()
    elif fault == "changed-helper":
        workflow.config["git"]["objects"] = {WORKFLOW_SHA + ":tools/release.py": "8" * 40}
    else:
        workflow.config["git"]["ancestors"] = [WORKFLOW_SHA]
    rejected(workflow.run("resolve", "--version", "1.0.0", "--resume", str(RUN_ID)))
    assert_read_only(workflow)
    assert not workflow.output.exists()


def test_artifact_verifies_exact_original_archive_bytes(workflow):
    before = snapshot(workflow.bundle.root)
    result = workflow.run("artifact", *workflow.artifact_args())
    require_success(result)
    assert any(
        call["endpoint"] == PREFIX + f"actions/artifacts/{ARTIFACT_ID}/zip"
        for call in workflow.calls()
    )
    assert snapshot(workflow.bundle.root) == before
    assert_read_only(workflow)


@pytest.mark.parametrize(
    "fault",
    [
        "expired",
        "missing",
        "wrong-id",
        "wrong-run",
        "wrong-revision",
        "wrong-digest",
        "changed-archive",
    ],
)
def test_artifact_rejects_missing_expired_or_conflicting_original_bytes(workflow, fault):
    workflow.ready("artifact", *workflow.artifact_args())
    if fault == "missing":
        workflow.route(f"actions/artifacts/{ARTIFACT_ID}", status=404)
    elif fault == "changed-archive":
        workflow.route(f"actions/artifacts/{ARTIFACT_ID}/zip", raw=workflow.archive + b"altered")
    else:
        if fault == "expired":
            workflow.artifact["expired"] = True
        elif fault == "wrong-id":
            workflow.artifact["id"] += 1
        elif fault in ("wrong-run", "wrong-revision"):
            workflow.artifact["workflow_run"]["id" if fault == "wrong-run" else "head_sha"] = (
                RUN_ID + 1 if fault == "wrong-run" else WORKFLOW_SHA
            )
        else:
            workflow.artifact["digest"] = "sha256:" + "0" * 64
        workflow.reset_artifact()
    # Expected digest is frozen independently, not taken from mutated metadata.
    args = workflow.artifact_args()
    args[args.index("--artifact-digest") + 1] = "sha256:" + digest(workflow.archive)
    rejected(workflow.run("artifact", *args))
    assert_read_only(workflow)


def test_bind_and_summary_expose_complete_approval_identity(workflow):
    workflow.ready("bind", *workflow.binding())
    before = snapshot(workflow.bundle.dist)
    result = workflow.run(
        "summary",
        *workflow.binding(),
        "--artifact-id",
        str(ARTIFACT_ID),
        "--artifact-digest",
        workflow.artifact["digest"],
    )
    require_success(result)
    manifest_hash = digest(workflow.bundle.manifest.read_bytes())
    assert workflow.outputs()["manifest_digest"] == manifest_hash
    summary = workflow.summary.read_text(encoding="utf-8")
    for value in (
        SOURCE_SHA,
        str(RUN_ID),
        "1.0.0",
        "pypi",
        manifest_hash,
        str(ARTIFACT_ID),
        workflow.artifact["digest"],
        "david-hoffman/dphtools",
    ):
        assert value in summary, "Missing approval identity: " + value
    for file in workflow.payload["files"]:
        assert file["filename"] in summary and file["sha256"] in summary
    for platform, report_hash in workflow.payload["verification"].items():
        assert platform in summary and report_hash in summary
    assert snapshot(workflow.bundle.dist) == before
    assert_read_only(workflow)


@pytest.mark.parametrize(
    "fault",
    ["version", "source", "workflow", "run", "report-missing", "report-changed", "wheel-changed"],
)
def test_bind_requires_frozen_manifest_and_required_reports(workflow, fault):
    workflow.ready("bind", *workflow.binding())
    args = workflow.binding()
    if fault in ("version", "source", "workflow", "run"):
        option, value = {
            "version": ("--version", "1.0.1"),
            "source": ("--source-sha", WORKFLOW_SHA),
            "workflow": ("--workflow-sha", WORKFLOW_SHA),
            "run": ("--origin-run", str(RUN_ID + 1)),
        }[fault]
        args[args.index(option) + 1] = value
    elif fault == "wheel-changed":
        workflow.bundle.wheel.write_bytes(b"different original bytes")
    else:
        report = workflow.bundle.reports / PLATFORMS[0] / "checks.json"
        if fault == "report-missing":
            report.unlink()
        else:
            report.write_bytes(report.read_bytes() + b" ")
    rejected(workflow.run("bind", *args))
    assert_read_only(workflow)


def test_tag_pins_original_source_even_after_unrelated_main_progress(workflow):
    workflow.config["git"]["head"] = WORKFLOW_SHA
    result = workflow.run("tag", *workflow.tag_args())
    require_success(result)
    writes = [call for call in workflow.calls() if call["method"] != "GET"]
    assert len(writes) == 1
    assert writes[0]["endpoint"] == PREFIX + "git/refs"
    assert json.loads(base64.b64decode(writes[0]["body"])) == {
        "ref": "refs/tags/1.0.0",
        "sha": SOURCE_SHA,
    }


@pytest.mark.parametrize("fault", ["off-main", "manifest-hash", "wrong-tag", "ambiguous-tag"])
def test_tag_conflicts_stop_without_retargeting_or_writes(workflow, fault):
    workflow.ready("tag", *workflow.tag_args())
    args = workflow.tag_args()
    if fault == "off-main":
        workflow.config["git"]["ancestors"] = []
    elif fault == "manifest-hash":
        args[-1] = "0" * 64
    elif fault == "wrong-tag":
        workflow.route("git/ref/tags/1.0.0", {"object": {"type": "commit", "sha": WORKFLOW_SHA}})
    else:
        workflow.route("git/ref/tags/1.0.0", status=500)
    rejected(workflow.run("tag", *args))
    assert_read_only(workflow)


def test_finalize_uploads_retained_files_and_retries_without_conflicting_overwrite(workflow):
    workflow.route("git/ref/tags/1.0.0", {"object": {"type": "commit", "sha": SOURCE_SHA}})
    original = {
        path.name: path.read_bytes()
        for path in (
            workflow.bundle.wheel,
            workflow.bundle.sdist,
            workflow.bundle.manifest,
            workflow.receipt,
        )
    }
    result = workflow.run("finalize", *workflow.tag_args(), "--receipt", workflow.receipt)
    require_success(result)
    calls = workflow.calls()
    uploads = [
        call for call in calls if call["method"] == "POST" and "/assets?name=" in call["endpoint"]
    ]
    assert {
        call["endpoint"].split("?name=", 1)[1]: base64.b64decode(call["body"]) for call in uploads
    } == original
    creates = [
        call
        for call in calls
        if call["method"] == "POST" and call["endpoint"] == PREFIX + "releases"
    ]
    assert len(creates) == 1
    body = json.loads(base64.b64decode(creates[0]["body"]))
    assert body["tag_name"] == "1.0.0" and body["target_commitish"] == SOURCE_SHA
    assert body["draft"] is True and body["prerelease"] is False
    assert workflow.payload["notes"].strip() in body["body"]
    assert_published_after_uploads(workflow)
    assert {
        path.name: path.read_bytes()
        for path in (
            workflow.bundle.wheel,
            workflow.bundle.sdist,
            workflow.bundle.manifest,
            workflow.receipt,
        )
    } == original
    workflow.clear()
    result = workflow.run("finalize", *workflow.tag_args(), "--receipt", workflow.receipt)
    require_success(result)
    assert not any(
        call["method"] == "POST" for call in workflow.calls()
    ), "Retry must not create or upload duplicate retained files"
    assert (
        workflow.config["routes"]["GET " + PREFIX + "releases/tags/1.0.0"]["json"]["draft"]
        is False
    )
    workflow.config["routes"]["GET " + PREFIX + "releases/491/assets?per_page=100"]["json"][0][
        "digest"
    ] = ("sha256:" + "0" * 64)
    workflow.clear()
    rejected(workflow.run("finalize", *workflow.tag_args(), "--receipt", workflow.receipt))
    assert_read_only(workflow)


@pytest.mark.parametrize("fault", ["off-main", "manifest-hash", "changed-file", "wrong-tag"])
def test_finalize_frozen_identity_failure_prevents_release_writes(workflow, fault):
    workflow.route("git/ref/tags/1.0.0", {"object": {"type": "commit", "sha": SOURCE_SHA}})
    workflow.ready("finalize", *workflow.tag_args(), "--receipt", workflow.receipt)
    args = workflow.tag_args()
    if fault == "off-main":
        workflow.config["git"]["ancestors"] = []
    elif fault == "manifest-hash":
        args[-1] = "0" * 64
    elif fault == "changed-file":
        workflow.bundle.wheel.write_bytes(b"changed after approved identity")
    else:
        workflow.route("git/ref/tags/1.0.0", {"object": {"type": "commit", "sha": WORKFLOW_SHA}})
    rejected(workflow.run("finalize", *args, "--receipt", workflow.receipt))
    assert_read_only(workflow)


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0rc1"])
@pytest.mark.parametrize("published", ["none", "wheel", "both"])
def test_downloaded_surface_retains_exact_selected_registry_originals(
    worker, tmp_path, version, published
):
    registry = Registry(worker, Bundle(tmp_path / "bundle", version))
    registry.ready()
    names = {
        "none": [],
        "wheel": [registry.bundle.wheel.name],
        "both": [registry.bundle.wheel.name, registry.bundle.sdist.name],
    }[published]
    registry.publish(names)
    downloaded = tmp_path / "verified downloads"
    result = registry.run("--downloaded", downloaded)
    require_success(result)
    original = snapshot(registry.bundle.dist)
    assert snapshot(downloaded) == {name: original[name] for name in names}
    assert snapshot(registry.output) == {
        name: value for name, value in original.items() if name not in names
    }
    assert all(
        call["method"] == "GET" and call["url"] in {registry.api, *registry.urls.values()}
        for call in registry.calls()
    )


def test_downloaded_surface_never_exposes_corrupt_bytes_as_verified(worker, tmp_path):
    registry = Registry(worker, Bundle(tmp_path / "bundle"))
    registry.publish([registry.bundle.wheel.name])
    downloaded = tmp_path / "verified downloads"
    good = registry.run("--downloaded", downloaded)
    require_success(good)
    shutil.rmtree(downloaded)
    shutil.rmtree(registry.output)
    registry.output.mkdir()
    registry.log.unlink()
    registry.config[registry.urls[registry.bundle.wheel.name]]["body"] = base64.b64encode(
        b"corrupt external bytes"
    ).decode()
    rejected(registry.run("--downloaded", downloaded))
    assert any(
        call["url"] == registry.urls[registry.bundle.wheel.name] for call in registry.calls()
    )
    assert registry.bundle.wheel.name not in snapshot(downloaded)


def redirect_registry(workflow, worker):
    """Real urllib redirects, controlled HTTPS IO; no owned transport patched."""
    registry = Registry(worker, workflow.bundle)
    registry.driver.write_text(BOOTSTRAP, encoding="utf-8")
    registry.publish([registry.bundle.wheel.name])
    for url, item in registry.config.items():
        workflow.config["routes"]["GET " + url] = {"raw": item["body"]}
    return registry


@pytest.mark.parametrize(
    "unsafe",
    [
        "https://127.0.0.1/private",
        "https://example.invalid/file",
        "http://files.pythonhosted.org/file",
        "https://test-files.pythonhosted.org/file",
    ],
)
@pytest.mark.parametrize("redirect", [False, True])
def test_downloaded_surface_does_not_fetch_unsafe_or_redirected_hosts(
    workflow, worker, unsafe, redirect
):
    registry = redirect_registry(workflow, worker)
    downloaded = workflow.root / "verified downloads"

    def run():
        write_json(workflow.config_file, workflow.config)
        write_json(workflow.bootstrap_config, {"environment": workflow.env})
        return invoke(
            registry.driver,
            worker[0],
            workflow.bootstrap_config,
            "reconcile",
            "--manifest",
            registry.bundle.manifest,
            "--dist",
            registry.bundle.dist,
            "--output",
            registry.output,
            "--downloaded",
            downloaded,
            cwd=worker[1],
            env=clean_env(),
        )

    good = run()
    require_success(good)
    assert snapshot(downloaded) == {
        registry.bundle.wheel.name: digest(registry.bundle.wheel.read_bytes())
    }
    shutil.rmtree(downloaded)
    shutil.rmtree(registry.output)
    registry.output.mkdir()
    workflow.clear()
    if redirect:
        workflow.config["routes"]["GET " + registry.urls[registry.bundle.wheel.name]] = {
            "status": 302,
            "headers": {"Location": unsafe},
        }
    else:
        registry.payload["urls"][0]["url"] = unsafe
        workflow.config["routes"]["GET " + registry.api] = {
            "raw": base64.b64encode(json.dumps(registry.payload).encode()).decode()
        }
    # Supply external bytes at the forbidden endpoint so an attempted fetch cannot
    # masquerade as a successful safety rejection caused by a missing fixture.
    workflow.config["routes"]["GET " + unsafe] = {
        "raw": base64.b64encode(registry.bundle.wheel.read_bytes()).decode()
    }
    result = run()
    rejected(result)
    assert all(
        call["endpoint"] != unsafe for call in workflow.calls()
    ), "Forbidden external host was fetched"
    assert not snapshot(downloaded)


@pytest.mark.parametrize(
    "setting", ["PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"]
)
def test_smoke_ignores_inherited_import_and_install_retargeting(
    worker, tmp_path, real_package, setting
):
    smoke = Smoke(worker, real_bundle(tmp_path / "real bundle", real_package))
    # A real successful clean control is required before environmental injection.
    good = smoke.run()
    require_success(good)
    smoke.log.unlink()
    foreign = tmp_path / "foreign location"
    foreign.mkdir()
    (foreign / "dphtools.py").write_text(
        "raise RuntimeError('FOREIGN-IMPORT-REACHED')\n", encoding="utf-8"
    )
    value = "1" if setting == "PIP_USER" else str(foreign)
    # Inject after startup, preserving the externally supplied offline wheel source.
    prefix = "import os\nos.environ[" + repr(setting) + "] = " + repr(value) + "\n"
    smoke.driver.write_text(prefix + PROCESS_DRIVER, encoding="utf-8")
    before = snapshot(smoke.bundle.dist)
    result = smoke.run()
    require_success(result)
    records = [item for item in smoke.calls() if item["event"] == "installed-check"]
    assert len(records) == 2 and all(item["returncode"] == 0 for item in records)
    for item in records:
        origin = Path(item["result"]["origin"]).resolve()
        assert origin.is_relative_to(Path(item["environment"]).resolve())
        assert not origin.is_relative_to(foreign.resolve())
        assert not Path(item["environment"]).exists()
    assert sorted(path.name for path in foreign.iterdir()) == ["dphtools.py"]
    assert snapshot(smoke.bundle.dist) == before


def test_recovery_downloaded_installs_precede_retained_file_finalization(
    workflow, worker, real_package
):
    """S5 public caller chain; this sequence does not prove hosted ordering."""
    bundle = workflow.bundle
    original_wheel, original_sdist = real_package
    shutil.copy2(original_wheel, bundle.wheel)
    shutil.copy2(original_sdist, bundle.sdist)
    for path in (bundle.wheel, bundle.sdist):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    workflow.payload = bundle.expected()
    workflow.payload["workflow_sha"] = SOURCE_SHA
    write_json(bundle.manifest, workflow.payload)
    workflow.archive = workflow.archive_bytes()
    workflow.artifact["digest"] = "sha256:" + digest(workflow.archive)
    workflow.reset_artifact()
    recovery_context(workflow)
    for command, args in (
        ("resolve", ["--version", "1.0.0", "--resume", str(RUN_ID)]),
        ("artifact", workflow.artifact_args()),
        ("bind", workflow.binding()),
        (
            "summary",
            [
                *workflow.binding(),
                "--artifact-id",
                str(ARTIFACT_ID),
                "--artifact-digest",
                workflow.artifact["digest"],
            ],
        ),
    ):
        result = workflow.run(command, *args)
        require_success(result)
    assert_read_only(workflow)
    original = snapshot(bundle.dist)
    registry = Registry(worker, bundle)
    registry.publish([bundle.wheel.name])
    downloaded = workflow.root / "published originals"
    result = registry.run("--downloaded", downloaded)
    require_success(result)
    assert snapshot(downloaded) == {bundle.wheel.name: original[bundle.wheel.name]}
    assert snapshot(registry.output) == {bundle.sdist.name: original[bundle.sdist.name]}
    # External registry state only: publish the original staged file in the fake.
    registry.publish([bundle.wheel.name, bundle.sdist.name])
    shutil.rmtree(downloaded)
    shutil.rmtree(registry.output)
    registry.output.mkdir()
    result = registry.run("--downloaded", downloaded, "--require-complete")
    require_success(result)
    assert snapshot(downloaded) == original and snapshot(registry.output) == {}
    retained_dist, retained_wheel, retained_sdist = bundle.dist, bundle.wheel, bundle.sdist
    bundle.dist = downloaded
    bundle.wheel = downloaded / retained_wheel.name
    bundle.sdist = downloaded / retained_sdist.name
    smoke = Smoke(worker, bundle)
    result = smoke.run()
    require_success(result)
    smoke.assert_source_install_without_git()
    installed = [item for item in smoke.calls() if item["event"] == "installed-check"]
    assert len(installed) == 2 and all(item["returncode"] == 0 for item in installed)
    assert len({item["environment"] for item in installed}) == 2
    bundle.dist, bundle.wheel, bundle.sdist = retained_dist, retained_wheel, retained_sdist
    write_json(
        workflow.receipt,
        {**workflow.payload, "published": workflow.payload["files"], "missing": []},
    )
    workflow.route("git/ref/tags/1.0.0", {"object": {"type": "commit", "sha": SOURCE_SHA}})
    workflow.clear()
    retained_files = {
        path.name: path.read_bytes()
        for path in (bundle.wheel, bundle.sdist, bundle.manifest, workflow.receipt)
    }
    result = workflow.run("finalize", *workflow.tag_args(), "--receipt", workflow.receipt)
    require_success(result)
    uploads = [
        call
        for call in workflow.calls()
        if call["method"] == "POST" and "/assets?name=" in call["endpoint"]
    ]
    for path in (bundle.wheel, bundle.sdist, bundle.manifest, workflow.receipt):
        matches = [call for call in uploads if call["endpoint"].endswith("?name=" + path.name)]
        assert len(matches) == 1
        assert base64.b64decode(matches[0]["body"]) == path.read_bytes()
    assert_published_after_uploads(workflow)
    assert {
        path.name: path.read_bytes()
        for path in (bundle.wheel, bundle.sdist, bundle.manifest, workflow.receipt)
    } == retained_files
    assert snapshot(retained_dist) == snapshot(downloaded) == original


def test_https_fixture_preserves_real_redirect_processing_and_raw_bytes(workflow):
    """Adapter self-check only; no product behavior is replaced or credited."""
    first = "https://files.pythonhosted.org/packages/first"
    second = "https://files.pythonhosted.org/packages/second"
    payload = b"PK\x03\x04\x00\xffraw bytes\x00"
    workflow.config["routes"]["GET " + first] = {"status": 302, "headers": {"Location": second}}
    workflow.config["routes"]["GET " + second] = {"raw": base64.b64encode(payload).decode()}
    probe = workflow.root / "external redirect probe.py"
    probe.write_text(
        "import urllib.request\n"
        "with urllib.request.urlopen(" + repr(first) + ") as response:\n"
        "    assert response.geturl() == " + repr(second) + "\n"
        "    assert response.read() == " + repr(payload) + "\n",
        encoding="utf-8",
    )
    write_json(workflow.config_file, workflow.config)
    write_json(workflow.bootstrap_config, {"environment": workflow.env})
    result = invoke(
        workflow.driver, probe, workflow.bootstrap_config, cwd=workflow.outside, env=clean_env()
    )
    require_success(result)
    assert [call["endpoint"] for call in workflow.calls()] == [first, second]


@pytest.mark.parametrize("preparation", ["new", "matching", "conflicting"])
def test_probe_blob_identity_freezes_preparation_and_recovery(workflow, preparation):
    """S4/S5: opaque fourth helper identity through external Git metadata."""
    helper = "tools/release_probe.py"
    original_blob = "9" * 40
    workflow.config["git"]["objects"] = {
        SOURCE_SHA + ":" + helper: original_blob,
        WORKFLOW_SHA + ":" + helper: original_blob,
    }
    before = snapshot(workflow.bundle.dist)
    if preparation == "new":
        require_success(workflow.run("resolve", "--version", "1.0.0"))
        assert workflow.outputs()["source_sha"] == SOURCE_SHA
        assert workflow.outputs()["resume"] == "false"
    else:
        recovery_context(workflow)
        args = ["--version", "1.0.0", "--resume", str(RUN_ID)]
        require_success(workflow.run("resolve", *args))
        output = workflow.outputs()
        assert output["source_sha"] == output["workflow_sha"] == SOURCE_SHA
        assert output["origin_run"] == str(RUN_ID)
        assert output["artifact_id"] == str(ARTIFACT_ID)
        assert output["artifact_digest"] == workflow.artifact["digest"]
        assert output["resume"] == "true"
        operations = [json.loads(line) for line in workflow.git_log.read_text().splitlines()]
        if preparation == "matching":
            for sha in (SOURCE_SHA, WORKFLOW_SHA):
                operation = ["rev-parse", sha + ":" + helper]
                if operation not in operations:
                    print("Missing declared external helper identity lookup:", operation)
                assert operation in operations
        assert_read_only(workflow)
        if preparation == "conflicting":
            workflow.clear()
            workflow.config["git"]["objects"][WORKFLOW_SHA + ":" + helper] = "a" * 40
            result = workflow.run("resolve", *args)
            code = result.returncode
            if code == 0:
                print(
                    "Conflicting external probe blob was accepted after matching recovery control"
                )
            assert code != 0, diagnostic(result)
            assert (result.stdout + result.stderr).strip(), "Conflict needs a diagnostic"
            assert not workflow.output.exists()
            operations = [json.loads(line) for line in workflow.git_log.read_text().splitlines()]
            for sha in (SOURCE_SHA, WORKFLOW_SHA):
                operation = ["rev-parse", sha + ":" + helper]
                if operation not in operations:
                    print("Missing declared external helper identity lookup:", operation)
                assert operation in operations
    assert_read_only(workflow)
    assert snapshot(workflow.bundle.dist) == before
