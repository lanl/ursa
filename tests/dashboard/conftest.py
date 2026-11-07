from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_dashboard_config_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dashboard tests independent of the developer's URSA config."""
    monkeypatch.setattr(
        "ursa_dashboard.settings.system_config_paths", lambda: []
    )
    monkeypatch.setattr("ursa_dashboard.settings.user_config_paths", lambda: [])
    monkeypatch.setattr(
        "ursa_dashboard.settings.dashboard_environment_config_layer",
        lambda: {},
    )
