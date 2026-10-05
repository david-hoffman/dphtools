"""Share verified immutable package bytes, while each installer stays private."""

import pytest

from .test_release_smoke import build_real_package


@pytest.fixture(scope="session")
def real_package(tmp_path_factory):
    artifacts = build_real_package(tmp_path_factory)
    yield artifacts
    artifacts.verify()
