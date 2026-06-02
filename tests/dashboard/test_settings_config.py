from __future__ import annotations

import json

import pytest

from ursa.security import GroupBaseURLPolicyError
from ursa_dashboard.settings import (
    SettingsStore,
    apply_dashboard_config,
    dashboard_llm_patch_from_ursa_config,
)


def test_dashboard_config_maps_cli_llm_model_to_dashboard_settings(tmp_path):
    cfg_path = tmp_path / "endpoint.yaml"
    cfg_path.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-test",
            "  base_url: https://models.example.org/v1",
            "  api_key_env: SAFE_API_KEY",
            "  max_completion_tokens: 4096",
            "  temperature: 0.3",
            "  seed: 123",
            "  model_kwargs:",
            "    use_responses_api: true",
        ]),
        encoding="utf-8",
    )

    patch = dashboard_llm_patch_from_ursa_config(cfg_path)

    assert patch == {
        "llm": {
            "model": "openai:gpt-test",
            "base_url": "https://models.example.org/v1",
            "api_key_env_var": "SAFE_API_KEY",
            "max_tokens": 4096,
            "temperature": 0.3,
            "model_kwargs": {
                "seed": 123,
                "use_responses_api": True,
            },
        }
    }


def test_dashboard_config_rejects_raw_api_key(tmp_path):
    cfg_path = tmp_path / "endpoint.yaml"
    cfg_path.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-test",
            "  api_key: secret-value",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not store raw"):
        dashboard_llm_patch_from_ursa_config(cfg_path)


def test_apply_dashboard_config_validates_group_before_persisting(
    tmp_path, monkeypatch
):
    import ursa.security as security

    groups_dir = tmp_path / "ursa"
    group_dir = groups_dir / "my_safety_group"
    group_dir.mkdir(parents=True)
    (group_dir / "group.yaml").write_text(
        "allowed_base_urls:\n  - https://safe.example.org/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(security, "AGENT_GROUPS_DIR", groups_dir)

    store = SettingsStore(tmp_path / "workspace")
    original = store.load()

    safe_cfg = tmp_path / "safe_endpoint.yaml"
    safe_cfg.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-safe",
            "  base_url: https://safe.example.org/v1",
        ]),
        encoding="utf-8",
    )

    settings = apply_dashboard_config(store, safe_cfg, group="my_safety_group")
    assert settings.llm.model == "openai:gpt-safe"
    assert settings.llm.base_url == "https://safe.example.org/v1"

    unsafe_cfg = tmp_path / "unsafe_endpoint.yaml"
    unsafe_cfg.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-unsafe",
            "  base_url: https://unsafe.example.org/v1",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(GroupBaseURLPolicyError):
        apply_dashboard_config(store, unsafe_cfg, group="my_safety_group")

    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["llm"]["model"] == "openai:gpt-safe"
    assert persisted["llm"]["base_url"] == "https://safe.example.org/v1"
    assert persisted["updated_at"] != original.updated_at
