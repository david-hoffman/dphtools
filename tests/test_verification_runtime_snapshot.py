"""Keep fresh runtime identity bytes, ordering, aliases, and I/O failures visible."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = """
import hashlib, json, os, sys
from pathlib import Path
from coverage.config import CoverageConfig
sys.path.insert(0, sys.argv[1])
from verification_reuse import runtime_identity
root, prefix = map(Path, sys.argv[2:])
sys.prefix = sys.base_prefix = str(prefix)
sys.path = [str(root), str(root / 'tools')]
def expected(files, aliases=()):
    entries = [(str(path), hashlib.sha256(content).hexdigest()) for path, content in files]
    entries.sort(key=lambda entry: Path(entry[0]))
    aliases = sorted([(str(path), str(target)) for path, target in aliases],
                     key=lambda entry: Path(entry[0]))
    encoded = json.dumps({'files': entries, 'directory_aliases': aliases}, sort_keys=True)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()
"""


def snapshot_probe(tmp_path, script):
    """Execute the real identity function from its private byte and I/O fixture."""
    root = tmp_path / "repository"
    tools = root / "tools"
    tools.mkdir(parents=True)
    for name in ("verification_reuse.py", "verification_inputs.py"):
        shutil.copy2(ROOT / "tools" / name, tools / name)
    prefix = tmp_path / "runtime"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONUSERBASE"] = str(tmp_path / "private-user-base")
    result = subprocess.run(
        [sys.executable, "-c", PREAMBLE + script, str(tools), str(root), str(prefix)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_snapshot_rechecks_small_and_large_bytes_with_file_and_directory_aliases(tmp_path):
    """An unchanged ordered snapshot binds every real file and directory alias."""
    snapshot_probe(
        tmp_path,
        """
prefix.mkdir()
directory = prefix / 'a'
directory.mkdir()
contents = {
    directory / 'child': b'nested input',
    prefix / 'a-z': b'sibling input',
    prefix / '.pth': b'a dotfile without an extension',
    prefix / 'large.bin': b'L' * 65536,
}
for path, content in contents.items():
    path.write_bytes(content)
file_alias = prefix / 'file-alias'
file_alias.symlink_to(prefix / 'a-z')
contents[file_alias] = contents[prefix / 'a-z']
directory_alias = prefix / 'directory-alias'
directory_alias.symlink_to(directory, target_is_directory=True)
(prefix / 'broken-alias').symlink_to(prefix / 'missing')
(prefix / 'looping-alias').symlink_to(prefix / 'looping-alias')
first = runtime_identity(root)
assert first['runtime_files'] == len(contents)
assert first['runtime_digest'] == expected(contents.items(), [(directory_alias, directory)])
assert runtime_identity(root) == first
contents[prefix / 'a-z'] = b'changed sibling'
(prefix / 'a-z').write_bytes(contents[prefix / 'a-z'])
contents[file_alias] = contents[prefix / 'a-z']
contents[prefix / 'large.bin'] = b'Q' * 65536
(prefix / 'large.bin').write_bytes(contents[prefix / 'large.bin'])
changed = runtime_identity(root)
assert changed['runtime_digest'] == expected(contents.items(), [(directory_alias, directory)])
assert changed != first
""",
    )


def test_snapshot_retains_readable_files_when_directory_scan_is_denied(tmp_path):
    """An OS scan failure retains the original traversal's readable-file scope."""
    snapshot_probe(
        tmp_path,
        """
prefix.mkdir()
denied = prefix / 'denied'
denied.mkdir()
(denied / 'unreachable').write_bytes(b'unreachable input')
readable = prefix / 'readable'
readable.write_bytes(b'reachable input')
real_scandir = os.scandir
def scandir(path):
    if Path(path) == denied:
        raise PermissionError('directory scan denied')
    return real_scandir(path)
os.scandir = scandir
actual = runtime_identity(root)
assert actual['runtime_files'] == 1
assert actual['runtime_digest'] == expected([(readable, b'reachable input')])
""",
    )


def test_snapshot_never_accepts_partial_data_after_directory_iteration_fails(tmp_path):
    """An interrupted OS directory iterator cannot yield reusable partial data."""
    snapshot_probe(
        tmp_path,
        """
prefix.mkdir()
(prefix / 'first').write_bytes(b'first input')
(prefix / 'second').write_bytes(b'second input')
real_scandir = os.scandir
class FailedScan:
    def __init__(self, directory):
        self.directory = directory
        self.started = False
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.directory.close()
    def __iter__(self):
        return self
    def __next__(self):
        if self.started:
            raise PermissionError('directory iteration failed')
        self.started = True
        return next(self.directory)
def scandir(path):
    directory = real_scandir(path)
    return FailedScan(directory) if Path(path) == prefix else directory
os.scandir = scandir
try:
    runtime_identity(root)
except PermissionError as error:
    assert str(error) == 'directory iteration failed'
else:
    raise AssertionError('A partial directory scan supplied reusable evidence')
""",
    )


@pytest.mark.parametrize("size", [4, 65536], ids=["small-read", "large-read"])
def test_snapshot_never_turns_a_failed_file_read_into_reusable_evidence(tmp_path, size):
    """The actual OS read failure propagates for small and substantial files."""
    snapshot_probe(
        tmp_path,
        """
prefix.mkdir()
unreadable = prefix / 'unreadable'
unreadable.write_bytes(b'X' * SIZE)
def refuse_open(event, arguments):
    if event == 'open' and arguments[0] == str(unreadable):
        raise PermissionError('runtime read denied')
sys.addaudithook(refuse_open)
try:
    runtime_identity(root)
except PermissionError as error:
    assert str(error) == 'runtime read denied'
else:
    raise AssertionError('A failed file read supplied reusable evidence')
""".replace("SIZE", str(size)),
    )


def test_snapshot_preserves_empty_traversal_for_a_prefix_that_is_a_file(tmp_path):
    """A nondirectory prefix retains the existing empty traversal result."""
    snapshot_probe(
        tmp_path,
        """
prefix.write_bytes(b'a nondirectory prefix')
actual = runtime_identity(root)
assert actual['runtime_files'] == 0
assert actual['runtime_digest'] == expected([])
""",
    )


@pytest.mark.parametrize("name", ["sitecustomize", "usercustomize"])
def test_snapshot_rejects_bare_customization_directories(tmp_path, name):
    """Uncontrolled startup directories cannot provide reusable runtime identity."""
    snapshot_probe(
        tmp_path,
        """
prefix.mkdir()
(prefix / NAME).mkdir()
assert runtime_identity(root) is None
""".replace("NAME", repr(name)),
    )
