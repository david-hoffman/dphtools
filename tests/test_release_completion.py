"""Independent caller outcomes from the approved completion public packet."""

import json
from pathlib import Path
import re
import sys

import pytest

from .release_workflow_support import BOOTSTRAP, Workflow
from .test_release_reconcile import Registry
from .test_release_smoke import PROCESS_DRIVER, Smoke, clean_env, real_bundle, real_package
from .test_release_support import Bundle, SOURCE_SHA, snapshot, write_json
from .test_release_version import diagnostic, invoke, worker


def status(result, success=True):
    code = result.returncode
    message = diagnostic(result)
    if (code == 0) is not success:
        print(message, file=sys.stderr)
    assert (code == 0) is success, message


@pytest.mark.parametrize("version,host", [("1.0.0", "pypi.org"), ("1.0.0rc1", "test.pypi.org")])
def test_new_preparation_requires_available_selected_version(tmp_path, version, host):
    caller = Workflow(tmp_path)
    api = f"https://{host}/pypi/dphtools/{version}/json"
    caller.config["routes"]["GET " + api] = {"status": 404}
    caller.route("git/ref/tags/" + version, status=404)
    before = snapshot(caller.bundle.dist)
    status(caller.run("resolve", "--version", version))
    assert caller.outputs()["version"] == version
    assert caller.outputs()["source_sha"] == SOURCE_SHA
    assert any(c["endpoint"] == api for c in caller.calls())
    assert all(c["method"] == "GET" for c in caller.calls())
    caller.clear()
    caller.config["routes"]["GET " + api] = {
        "json": {"info": {"name": "dphtools", "version": version}, "urls": []}
    }
    result = caller.run("resolve", "--version", version)
    status(result, False)
    message = diagnostic(result).lower()
    assert re.search(r"resume|recover", message), "Rejection must direct original-run recovery"
    assert any(c["endpoint"] == api for c in caller.calls())
    assert all(c["method"] == "GET" for c in caller.calls())
    assert not caller.output.exists()
    assert snapshot(caller.bundle.dist) == before


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0rc1"])
def test_same_host_https_redirect_retains_original_downloads(worker, tmp_path, version):
    # Independent structural bytes; the real worker decides retention/staging.
    transport_root = tmp_path / "transport"
    transport_root.mkdir()
    caller = Workflow(transport_root)
    registry = Registry(worker, Bundle(tmp_path / "bundle", version))
    registry.publish([registry.bundle.wheel.name, registry.bundle.sdist.name])
    for url, item in registry.config.items():
        caller.config["routes"]["GET " + url] = {"raw": item["body"]}
    redirects = {}
    for url in [registry.api, *registry.urls.values()]:
        target = url + ("/redirected" if url == registry.api else "?retained=1")
        redirects[url] = target
        caller.config["routes"]["GET " + target] = caller.config["routes"]["GET " + url]
        caller.config["routes"]["GET " + url] = {"status": 302, "headers": {"Location": target}}
    write_json(caller.config_file, caller.config)
    write_json(caller.bootstrap_config, {"environment": caller.env})
    downloaded = tmp_path / "verified originals"
    before = snapshot(registry.bundle.dist)
    result = invoke(
        caller.driver,
        worker[0],
        caller.bootstrap_config,
        "reconcile",
        "--manifest",
        registry.bundle.manifest,
        "--dist",
        registry.bundle.dist,
        "--output",
        registry.output,
        "--downloaded",
        downloaded,
        "--require-complete",
        cwd=worker[1],
        env=clean_env(),
    )
    status(result)
    assert snapshot(downloaded) == before
    for path in registry.bundle.dist.iterdir():
        assert (downloaded / path.name).read_bytes() == path.read_bytes()
    assert snapshot(registry.output) == {}
    assert snapshot(registry.bundle.dist) == before
    calls = caller.calls()
    assert all(c["method"] == "GET" for c in calls)
    urls = [c["endpoint"] for c in calls]
    assert set(urls) == {*redirects, *redirects.values()}
    for original, target in redirects.items():
        assert urls.index(original) < urls.index(target)


@pytest.mark.parametrize("command", ["resolve", "bind"])
def test_imported_main_matches_cli_and_import_alone_has_no_remote_writes(tmp_path, command):
    caller = Workflow(tmp_path)
    args = ["--version", "1.0.0"] if command == "resolve" else caller.binding()
    status(caller.run(command, *args))
    expected = caller.outputs() if command == "resolve" else None
    caller.clear()
    loading = """import importlib.util
spec = importlib.util.spec_from_file_location("release_caller", entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
    trailer = 'runpy.run_path(entry, run_name="__main__")'
    assert BOOTSTRAP.count(trailer) == 1
    caller.driver.write_text(BOOTSTRAP.replace(trailer, loading), encoding="utf-8")
    status(caller.run(command, *args))
    assert caller.calls() == [], "Loading the entry must not call remote services"
    assert not caller.output.exists()
    caller.driver.write_text(
        BOOTSTRAP.replace(trailer, loading + "raise SystemExit(module.main())\n"),
        encoding="utf-8",
    )
    status(caller.run(command, *args))
    if expected is not None:
        assert caller.outputs() == expected
    assert all(c["method"] == "GET" for c in caller.calls())
    caller.clear()
    if command == "resolve":
        caller.env["GITHUB_REF"] = "refs/heads/feature"
    else:
        args[args.index("--source-sha") + 1] = "f" * 40
    status(caller.run(command, *args), False)
    assert all(c["method"] == "GET" for c in caller.calls())
    assert not caller.output.exists()


def test_real_installs_and_probes_strip_dummy_inherited_credentials(
    worker, tmp_path, real_package
):
    smoke = Smoke(worker, real_bundle(tmp_path / "bundle", real_package))
    keys = [
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "TWINE_USERNAME",
        "TWINE_PASSWORD",
        "PYPI_TOKEN",
        "TESTPYPI_TOKEN",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    ]
    # Inject only dummy values after invoke's ambient credential removal. Keep the
    # offline dependency source. Record names only, never ambient values.
    injection = (
        "import os\nos.environ.update("
        + repr(
            {
                key: (
                    "https://dummy:sentinel@example.invalid/simple"
                    if key.startswith("PIP_")
                    else "release-completion-dummy-sentinel"
                )
                for key in keys
            }
        )
        + ")\nos.environ['PIP_NO_CACHE_DIR'] = '1'\n"
    )
    marker = '        self.cwd_value = str(kwargs.get("cwd") or Path.cwd())'
    assert PROCESS_DRIVER.count(marker) == 1
    observe = (
        """        record({"event": "credential-environment", "stage": stage,
                "present": [key for key in """
        + repr(keys)
        + """ if key in self.env_values
                            and (not key.startswith("PIP_") or "@" in self.env_values[key])]})
"""
    )
    smoke.driver.write_text(
        injection + PROCESS_DRIVER.replace(marker, observe + marker), encoding="utf-8"
    )
    before = snapshot(smoke.bundle.dist)
    status(smoke.run())
    records = smoke.calls()
    environments = [r for r in records if r["event"] == "credential-environment"]
    for stage in ("install", "check", "probe"):
        observed = [r for r in environments if r["stage"] == stage]
        assert observed, "Observe actual child environments at each public stage"
        if any(r["present"] for r in observed):
            print("Credential observation:", observed, file=sys.stderr)
        assert all(not r["present"] for r in observed), "Inherited credentials reached a child"
    installed = [r for r in records if r["event"] == "installed-check"]
    assert len(installed) == 2
    assert {r["artifact"] for r in installed} == set(smoke.config["artifacts"])
    assert len({r["environment"] for r in installed}) == 2
    for item in installed:
        code = item["returncode"]
        assert code == 0, "Independent installed observer failed"
        result = item["result"]
        assert result["metadata_version"] == result["package_version"] == "1.0.0"
        assert result["sum"] == 6
        assert Path(result["origin"]).resolve().is_relative_to(Path(item["environment"]).resolve())
        assert (
            Path(result["utils_origin"])
            .resolve()
            .is_relative_to(Path(item["environment"]).resolve())
        )
        assert not Path(item["environment"]).exists()
    smoke.assert_source_install_without_git()
    assert snapshot(smoke.bundle.dist) == before
