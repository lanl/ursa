"""Shared TUI setup; individual tests retain control of the event loop."""

import pytest

from tests.cli._app_helpers import make_runtime
from ursa.cli.tui.app import UrsaTextualApp


@pytest.fixture
def app(tmp_path):
    return UrsaTextualApp(make_runtime(tmp_path))


@pytest.fixture
def model_discovery(monkeypatch):
    """Install a catalog provider without contacting a real model endpoint."""

    def install(discover):
        monkeypatch.setattr(
            "ursa.cli.tui.widgets.list_provider_models", discover
        )

    return install
