from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_dashboard_config_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Keep dashboard tests independent of the developer's URSA config."""
    user_config = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "ursa_dashboard.settings.system_config_paths", lambda: []
    )
    monkeypatch.setattr(
        "ursa_dashboard.settings.user_config_paths", lambda: [user_config]
    )
    # The Settings tabs now write model defaults through UserConfigStore, whose
    # module imports user_config_paths separately from the resolver.
    monkeypatch.setattr(
        "ursa_dashboard.user_config.user_config_paths", lambda: [user_config]
    )
    monkeypatch.setattr(
        "ursa_dashboard.settings.dashboard_environment_config_layer",
        lambda: {},
    )
