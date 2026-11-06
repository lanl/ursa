"""Shared TUI setup; individual tests retain control of the event loop."""

import pytest

from tests.cli._app_helpers import make_runtime
from ursa.cli.tui.app import UrsaTextualApp


@pytest.fixture
def app(tmp_path):
    return UrsaTextualApp(make_runtime(tmp_path))
