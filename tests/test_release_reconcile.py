"""S4/S5: real reconciliation CLI, controlled standard external HTTP boundary.

No owned release code is imported or replaced by the test process. The child
bootstrap adapts urllib only and blocks real sockets. All JSON/archive validation,
digest decisions and staging still run through the real production entry point.
"""

import base64
import json
import os
import shutil

import pytest

from .test_release_support import Bundle, digest, rejected, snapshot, write_json
from . import test_release_version as version_cli
from .test_release_version import diagnostic, invoke

worker = version_cli.worker

HTTP_DRIVER = r"""
import base64
from email.message import Message
from io import BytesIO
import json
from pathlib import Path
import runpy
import socket
import sys
import urllib.error
import urllib.request
import warnings

entry, config_file, log_file, *args = sys.argv[1:]
config = json.loads(Path(config_file).read_text(encoding="utf-8"))
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f"{category.__name__}: {message}\n")
sys.excepthook = lambda kind, value, traceback: print(f"{kind.__name__}: {value}", file=sys.stderr)

class Response(BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.status = self.code = 200
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = "application/json" if url.endswith("/json") else "application/octet-stream"
        self.headers["Content-Length"] = str(len(data))
    def getcode(self):
        return self.status
    def geturl(self):
        return self.url
    def info(self):
        return self.headers

def open_url(request, data=None, *unused, **kwargs):
    url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
    method = request.get_method() if isinstance(request, urllib.request.Request) else ("POST" if data else "GET")
    with Path(log_file).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"url": url, "method": method}) + "\n")
    if method != "GET":
        raise AssertionError("Reconciliation must not publish or mutate registry state")
    item = config.get(url)
    if item is None:
        raise AssertionError("Unconfigured external URL: " + url)
    if "transport_error" in item:
        raise urllib.error.URLError(item["transport_error"])
    status = item.get("status", 200)
    if status != 200:
        raise urllib.error.HTTPError(url, status, "controlled HTTP failure", {}, BytesIO(b"error"))
    return Response(base64.b64decode(item["body"]), url)

urllib.request.urlopen = open_url
urllib.request.OpenerDirector.open = lambda self, *args, **kwargs: open_url(*args, **kwargs)
def no_socket(*args, **kwargs):
    raise AssertionError("Fixture forbids unconfigured network access")
socket.create_connection = no_socket
socket.socket.connect = no_socket
sys.argv = [entry, *args]
sys.path[0] = str(Path(entry).resolve().parent)
runpy.run_path(entry, run_name="__main__")
"""


class Registry:
    def __init__(self, worker, bundle):
        self.worker = worker
        self.bundle = bundle
        self.driver = bundle.root / "http-boundary.py"
        self.driver.write_text(HTTP_DRIVER, encoding="utf-8")
        self.config_file = bundle.root / "http-config.json"
        self.log = bundle.root / "http.jsonl"
        self.output = bundle.root / "staging"
        self.output.mkdir()
        channel = bundle.expected()["channel"]
        host = "pypi.org" if channel == "pypi" else "test.pypi.org"
        self.api = f"https://{host}/pypi/dphtools/{bundle.version}/json"
        file_host = (
            "files.pythonhosted.org" if channel == "pypi" else "test-files.pythonhosted.org"
        )
        self.urls = {
            path.name: f"https://{file_host}/packages/fixture/" + path.name
            for path in (bundle.wheel, bundle.sdist)
        }
        self.config = {}
        self.publish([])

    def publish(self, filenames):
        records = []
        for name in filenames:
            content = (self.bundle.dist / name).read_bytes()
            url = self.urls[name]
            records.append(
                {
                    "filename": name,
                    "digests": {"sha256": digest(content)},
                    "size": len(content),
                    "url": url,
                }
            )
            self.config[url] = {"body": base64.b64encode(content).decode()}
        self.payload = {
            "info": {"name": "dphtools", "version": self.bundle.version},
            "urls": records,
        }
        self.save_payload()

    def save_payload(self):
        self.config[self.api] = {
            "body": base64.b64encode(json.dumps(self.payload).encode()).decode()
        }

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def run(self, *args):
        write_json(self.config_file, self.config)
        entry, outside = self.worker
        env = dict(os.environ)
        for key in list(env):
            if any(word in key.upper() for word in ("TOKEN", "PASSWORD", "CREDENTIAL")):
                env.pop(key)
        return invoke(
            self.driver,
            entry,
            self.config_file,
            self.log,
            "reconcile",
            "--manifest",
            self.bundle.manifest,
            "--dist",
            self.bundle.dist,
            "--output",
            self.output,
            *args,
            cwd=outside,
            env=env,
        )

    def ready(self):
        result = self.run()
        assert result.returncode == 0, "Valid reconciliation precondition: " + diagnostic(result)
        assert snapshot(self.output) == snapshot(self.bundle.dist)
        assert self.calls() == [{"url": self.api, "method": "GET"}]
        shutil.rmtree(self.output)
        self.output.mkdir()
        self.log.unlink()


@pytest.fixture
def registry(worker, tmp_path):
    return Registry(worker, Bundle(tmp_path / "bundle"))


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0rc1"])
@pytest.mark.parametrize("published", ["none", "wheel", "sdist", "both", "404"])
def test_reconcile_stages_only_absent_original_bytes(worker, tmp_path, version, published):
    bundle = Bundle(tmp_path / "bundle", version)
    registry = Registry(worker, bundle)
    names = {
        "none": [],
        "404": [],
        "wheel": [bundle.wheel.name],
        "sdist": [bundle.sdist.name],
        "both": [bundle.wheel.name, bundle.sdist.name],
    }[published]
    registry.publish(names)
    if published == "404":
        registry.config[registry.api] = {"status": 404}
    before = snapshot(bundle.dist)
    result = registry.run()
    assert result.returncode == 0, diagnostic(result)
    assert snapshot(registry.output) == {
        name: value for name, value in before.items() if name not in names
    }
    assert snapshot(bundle.dist) == before
    calls = registry.calls()
    assert calls and calls[0] == {"url": registry.api, "method": "GET"}
    allowed = {registry.api, *(registry.urls[name] for name in names)}
    assert all(call["method"] == "GET" and call["url"] in allowed for call in calls)


@pytest.mark.parametrize("published", ["none", "wheel", "sdist", "both"])
def test_require_complete_fails_until_both_original_files_are_published(registry, published):
    registry.ready()
    bundle = registry.bundle
    names = {
        "none": [],
        "wheel": [bundle.wheel.name],
        "sdist": [bundle.sdist.name],
        "both": [bundle.wheel.name, bundle.sdist.name],
    }[published]
    registry.publish(names)
    result = registry.run("--require-complete")
    if published == "both":
        assert result.returncode == 0, diagnostic(result)
        assert snapshot(registry.output) == {}
    else:
        rejected(result)
    assert registry.calls()[0]["url"] == registry.api
    assert all(call["method"] == "GET" for call in registry.calls())


@pytest.mark.parametrize(
    "fault",
    ["401", "403", "429", "500", "transport", "malformed-json", "missing-urls", "wrong-type"],
)
def test_registry_failure_is_not_interpreted_as_no_publication(registry, fault):
    registry.ready()
    if fault.isdecimal():
        registry.config[registry.api] = {"status": int(fault)}
    elif fault == "transport":
        registry.config[registry.api] = {"transport_error": "controlled connection failure"}
    else:
        body = {"malformed-json": b"{", "missing-urls": b"{}", "wrong-type": b'{"urls": {}}'}[
            fault
        ]
        registry.config[registry.api] = {"body": base64.b64encode(body).decode()}
    rejected(registry.run())
    assert snapshot(registry.output) == {}, "Ambiguous registry state must not stage upload inputs"
    assert registry.calls() == [{"url": registry.api, "method": "GET"}]


@pytest.mark.parametrize(
    "fault", ["digest", "unexpected-file", "missing-digest", "duplicate-file", "malformed-digest"]
)
def test_published_identity_conflicts_fail_without_staging(registry, fault):
    registry.ready()
    registry.publish([registry.bundle.wheel.name])
    record = registry.payload["urls"][0]
    if fault == "digest":
        record["digests"]["sha256"] = "0" * 64
    elif fault == "unexpected-file":
        record["filename"] = "unexpected.whl"
    elif fault == "missing-digest":
        del record["digests"]
    elif fault == "duplicate-file":
        registry.payload["urls"].append(record.copy())
    else:
        record["digests"]["sha256"] = "not-a-digest"
    registry.save_payload()
    rejected(registry.run())
    assert snapshot(registry.output) == {}
    assert registry.calls()[0]["url"] == registry.api


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/private",
        "https://127.0.0.1/private",
        "https://files.pythonhosted.org.evil.invalid/file",
        "https://example.invalid/file",
        "https://files.pythonhosted.org@127.0.0.1/file",
    ],
)
def test_untrusted_registry_urls_are_rejected_without_fetching(registry, unsafe_url):
    registry.ready()
    registry.publish([registry.bundle.wheel.name])
    registry.payload["urls"][0]["url"] = unsafe_url
    registry.save_payload()
    rejected(registry.run())
    assert all(call["url"] != unsafe_url for call in registry.calls())
    assert snapshot(registry.output) == {}


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0rc1"])
def test_published_download_urls_must_belong_to_selected_registry(worker, tmp_path, version):
    registry = Registry(worker, Bundle(tmp_path / "bundle", version))
    registry.ready()
    registry.publish([registry.bundle.wheel.name])
    host = "test-files.pythonhosted.org" if version == "1.0.0" else "files.pythonhosted.org"
    foreign_url = f"https://{host}/packages/fixture/{registry.bundle.wheel.name}"
    registry.payload["urls"][0]["url"] = foreign_url
    registry.save_payload()
    rejected(registry.run())
    assert all(call["url"] != foreign_url for call in registry.calls())
    assert snapshot(registry.output) == {}


def test_downloaded_published_bytes_must_match_claimed_digest(registry):
    registry.ready()
    registry.publish([registry.bundle.wheel.name, registry.bundle.sdist.name])
    for url in registry.urls.values():
        registry.config[url] = {"body": base64.b64encode(b"bytes disagree with metadata").decode()}
    result = registry.run("--require-complete")
    downloads = [call for call in registry.calls() if call["url"] != registry.api]
    # The contract requires verification of ANY downloaded bytes; it does not
    # require reconcile itself to download files. Hosted downloaded-install checks
    # remain an independent S4/S5 workflow requirement in either case.
    if downloads:
        rejected(result)
    else:
        assert result.returncode == 0, diagnostic(result)
    assert snapshot(registry.output) == {}


def test_nonempty_staging_directory_is_never_trusted_or_overwritten(registry):
    registry.ready()
    sentinel = registry.output / registry.bundle.wheel.name
    sentinel.write_bytes(b"untrusted previous attempt")
    before = snapshot(registry.output)
    rejected(registry.run())
    assert snapshot(registry.output) == before
    assert all(call["method"] == "GET" for call in registry.calls())


@pytest.mark.parametrize("fault", ["missing-manifest", "missing-wheel", "modified-sdist"])
def test_recovery_refuses_missing_or_changed_retained_bundle_without_rebuild(registry, fault):
    registry.ready()
    bundle = registry.bundle
    if fault == "missing-manifest":
        bundle.manifest.unlink()
    elif fault == "missing-wheel":
        bundle.wheel.unlink()
    else:
        bundle.sdist.write_bytes(b"changed after approval")
    before = snapshot(bundle.dist)
    rejected(registry.run())
    assert registry.calls() == [], "Validate retained identity before registry access"
    assert snapshot(registry.output) == {}
    assert snapshot(bundle.dist) == before


def test_http_fixture_proves_status_and_byte_transport_without_product(tmp_path):
    """Self-check only: distinguish broken HTTP adapter from release failures."""
    driver = tmp_path / "driver.py"
    driver.write_text(HTTP_DRIVER, encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import urllib.request, urllib.error\n"
        "assert urllib.request.urlopen('https://fixture.invalid/good').read() == b'payload'\n"
        "try:\n    urllib.request.urlopen('https://fixture.invalid/missing')\n"
        "except urllib.error.HTTPError as error:\n    assert error.code == 404\n"
        "else:\n    raise AssertionError('404 became success')\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "https://fixture.invalid/good": {"body": base64.b64encode(b"payload").decode()},
            "https://fixture.invalid/missing": {"status": 404},
        },
    )
    log = tmp_path / "calls.jsonl"
    result = invoke(driver, probe, config, log, cwd=tmp_path)
    assert result.returncode == 0, diagnostic(result)
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2
