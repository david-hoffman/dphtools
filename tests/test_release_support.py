"""Independent fixtures for the release public contract, never runtime oracles.

Synthetic archives are structural inputs only. Smoke tests build the real package
from an opaque source copy instead of supplying a replacement implementation.
"""

import base64
import csv
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import stat
import sys
import sysconfig
import tarfile
import zipfile

from .test_release_version import diagnostic, invoke

REPOSITORY = "david-hoffman/dphtools"
SOURCE_SHA = "1234567890abcdef" * 2 + "12345678"
WORKFLOW_SHA = "abcdef0123456789" * 2 + "abcdef01"
RUN_ID = 731029
PLATFORMS = ("ubuntu-24.04", "macos-15", "windows-2025")
STEPS = (
    "format",
    "lint",
    "docstrings",
    "types",
    "audit",
    "build",
    "install",
    "coverage-erase",
    "tests",
    "coverage-combine",
    "coverage-json",
    "coverage-xml",
    "coverage-report",
    "report-validation",
)
DEPENDENCIES = ("numpy", "pandas", "scipy", "matplotlib", "scikit-image")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rejected(result):
    assert result.returncode != 0, diagnostic(result)
    assert (result.stdout + result.stderr).strip(), "A rejection needs a useful diagnostic"


def snapshot(directory):
    """Byte identity without inspecting opaque source content."""
    return {
        path.relative_to(directory).as_posix(): digest(path.read_bytes())
        for path in directory.rglob("*")
        if path.is_file()
    }


def install_executable(bin_dir, name, script):
    """Use the approved portable external-executable fixture convention."""
    bin_dir.mkdir(exist_ok=True)
    if os.name == "nt":
        wrapper = Path(sysconfig.get_path("scripts")) / "pytest.exe"
        assert wrapper.is_file(), "Fixture requires pytest's installed Windows launcher"
        with zipfile.ZipFile(wrapper) as archive:
            offset = min(info.header_offset for info in archive.infolist())
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("__main__.py", script)
        executable = bin_dir / f"{name}.exe"
        with wrapper.open("rb") as source, executable.open("wb") as target:
            target.write(source.read(offset))
            target.write(payload.getvalue())
    else:
        python = bin_dir / "python3"
        if not python.exists():
            python.symlink_to(sys.executable)
        executable = bin_dir / name
        executable.write_text("#!/usr/bin/env python3\n" + script, encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def metadata(version="1.0.0", name="dphtools", python=">=3.8", dependencies=DEPENDENCIES):
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    if python is not None:
        lines.append(f"Requires-Python: {python}")
    lines.extend(f"Requires-Dist: {requirement}" for requirement in dependencies)
    return ("\n".join(lines) + "\n\nStructural release fixture.\n").encode()


def write_wheel(path, version, package_metadata, extras=()):
    info = f"dphtools-{version}.dist-info"
    members = [
        ("dphtools/__init__.py", b"# Structural fixture; never used for smoke tests.\n"),
        (f"{info}/METADATA", package_metadata),
        (
            f"{info}/WHEEL",
            b"Wheel-Version: 1.0\nGenerator: release-tests\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n",
        ),
        *extras,
    ]
    record = StringIO(newline="")
    writer = csv.writer(record)
    for name, data in members:
        encoded_hash = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow((name, "sha256=" + encoded_hash.decode(), len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    members.append((f"{info}/RECORD", record.getvalue().encode()))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)


def write_sdist(path, version, package_metadata, extras=()):
    prefix = f"dphtools-{version}"
    members = [
        (f"{prefix}/PKG-INFO", package_metadata),
        (
            f"{prefix}/pyproject.toml",
            b"[build-system]\nrequires = ['setuptools']\n"
            b"build-backend = 'setuptools.build_meta'\n",
        ),
        (
            f"{prefix}/setup.cfg",
            b"[metadata]\nname = dphtools\nversion = " + version.encode() + b"\n",
        ),
        (
            f"{prefix}/dphtools/__init__.py",
            b"# Structural fixture; never used for smoke tests.\n",
        ),
        *extras,
    ]
    # Retain the declared requirements in build inputs as well as PKG-INFO.
    config_name, config_data = members[2]
    members[2] = (
        config_name,
        config_data + b"[options]\npackages = find:\n"
        b"python_requires = >=3.8\ninstall_requires =\n"
        + b"".join(b"    " + name.encode() + b"\n" for name in DEPENDENCIES),
    )
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, BytesIO(data))


class Bundle:
    """A specified manifest and real archives, independent of the manifest worker."""

    def __init__(self, root, version="1.0.0"):
        self.root = root
        root.mkdir()
        self.version = version
        self.dist = root / "retained dist"
        self.dist.mkdir()
        self.wheel = self.dist / f"dphtools-{version}-py3-none-any.whl"
        self.sdist = self.dist / f"dphtools-{version}.tar.gz"
        self.reports = root / "verification reports"
        self.reports.mkdir()
        self.notes = root / "release notes.txt"
        self.notes.write_text("Release fixture: café.\nPreserve these notes.\n", encoding="utf-8")
        self.manifest = root / "manifest.json"
        self.generated = root / "generated-manifest.json"
        self.rewrite_archive("wheel")
        self.rewrite_archive("sdist")
        for platform in PLATFORMS:
            directory = self.reports / platform
            directory.mkdir()
            native = {
                "ubuntu-24.04": "Linux-6.8.0-x86_64",
                "macos-15": "macOS-15.0-arm64",
                "windows-2025": "Windows-11-10.0.26100-SP0",
            }[platform]
            write_json(
                directory / "checks.json",
                {
                    "document_version": "1.0",
                    "mode": "full",
                    "python": "3.13.12",
                    "platform": native,
                    "steps": [
                        {
                            "name": name,
                            "command": None if name == "report-validation" else ["fixture", name],
                            "returncode": 0,
                        }
                        for name in STEPS
                    ],
                    "failed": [],
                    "measurement_limits": [],
                },
            )
        self.refresh_manifest()

    def rewrite_archive(self, kind, *, extras=(), **metadata_changes):
        package_metadata = metadata(self.version, **metadata_changes)
        writer = write_wheel if kind == "wheel" else write_sdist
        writer(getattr(self, kind), self.version, package_metadata, extras)

    def expected(self):
        return {
            "schema_version": 1,
            "repository": REPOSITORY,
            "version": self.version,
            "channel": "testpypi" if any(c in self.version for c in ("a", "b", "r")) else "pypi",
            "source_ref": "refs/heads/main",
            "source_sha": SOURCE_SHA,
            "workflow_sha": WORKFLOW_SHA,
            "run_id": RUN_ID,
            "files": [
                {
                    "filename": path.name,
                    "size": path.stat().st_size,
                    "sha256": digest(path.read_bytes()),
                }
                for path in (self.wheel, self.sdist)
            ],
            "verification": {
                platform: digest((self.reports / platform / "checks.json").read_bytes())
                for platform in PLATFORMS
            },
            "notes": self.notes.read_text(encoding="utf-8"),
        }

    def refresh_manifest(self):
        write_json(self.manifest, self.expected())

    def manifest_args(self):
        return [
            "manifest",
            "--version",
            self.version,
            "--source-sha",
            SOURCE_SHA,
            "--workflow-sha",
            WORKFLOW_SHA,
            "--run-id",
            str(RUN_ID),
            "--dist",
            self.dist,
            "--reports",
            self.reports,
            "--notes",
            self.notes,
            "--output",
            self.generated,
        ]

    def run(self, worker, command, *extra):
        entry, outside = worker
        return invoke(
            entry, command, "--manifest", self.manifest, "--dist", self.dist, *extra, cwd=outside
        )

    def ready(self, worker):
        result = self.run(worker, "check")
        assert result.returncode == 0, "Valid bundle precondition: " + diagnostic(result)


def test_structural_fixtures_match_the_clarified_packet(tmp_path):
    """Fixture self-check, not product evidence: archives, reports and independent hashes."""
    bundle = Bundle(tmp_path / "fixture")
    with zipfile.ZipFile(bundle.wheel) as archive:
        assert archive.testzip() is None
        assert archive.read("dphtools-1.0.0.dist-info/METADATA") == metadata()
    with tarfile.open(bundle.sdist) as archive:
        assert archive.extractfile("dphtools-1.0.0/PKG-INFO").read() == metadata()
    payload = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert payload == bundle.expected()
    for platform in PLATFORMS:
        report = json.loads(
            (bundle.reports / platform / "checks.json").read_text(encoding="utf-8")
        )
        assert tuple(step["name"] for step in report["steps"]) == STEPS
        assert len(set(step["name"] for step in report["steps"])) == 14
