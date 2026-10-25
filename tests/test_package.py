"""Package-level checks that guard against packaging drift."""

from importlib.metadata import version

import docproof


def test_runtime_version_matches_installed_metadata() -> None:
    # The version lives in two places; a release that bumps one and not the other ships a lie.
    assert docproof.__version__ == version("docproof")
