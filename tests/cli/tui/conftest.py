"""Shared TUI setup; individual tests retain control of the event loop."""

from contextlib import asynccontextmanager

import pytest
from textual.containers import VerticalScroll

from tests.cli._app_helpers import make_runtime
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.turn import Turn


@pytest.fixture
def app(tmp_path):
    return UrsaTextualApp(make_runtime(tmp_path))


@pytest.fixture
def finished_agent(app):
    async def run_agent(_name, _prompt, callbacks=None):
        return "Finished"

    app.runtime.run_agent = run_agent


@pytest.fixture
def notifications(app, monkeypatch):
    received = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: received.append((message, kwargs)),
    )
    return received


@pytest.fixture
def model_discovery(monkeypatch):
    """Install a catalog provider without contacting a real model endpoint."""

    def install(discover):
        monkeypatch.setattr(
            "ursa.cli.tui.widgets.list_provider_models", discover
        )

    return install


@pytest.fixture
def event_turn(app, tmp_path):
    """Run a mounted conversation turn for tests that drive events directly."""

    @asynccontextmanager
    async def mounted(prompt, *, size):
        async with app.run_test(size=size) as pilot:
            turn = Turn(prompt, tmp_path)
            await app.query_one("#conversation", VerticalScroll).mount(turn)
            yield pilot, turn

    return mounted
