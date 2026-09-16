from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ursa.security import GroupBaseURLPolicyError
from ursa.util.inference_providers import ProviderModel
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.settings import (
    DashboardConfigResolver,
    GlobalSettings,
    LLMSettings,
    SettingsStore,
    apply_dashboard_config,
    dashboard_llm_patch_from_ursa_config,
)


def test_dashboard_config_precedence_places_settings_below_user(
    tmp_path, monkeypatch
):
    system = tmp_path / "system.yaml"
    user = tmp_path / "user.yaml"
    explicit = tmp_path / "explicit.yaml"
    system.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:system-model",
            "emb_model:",
            "  model: openai:system-embedding",
        ])
    )
    user.write_text("llm_model:\n  model: openai:user-model\n")
    explicit.write_text("llm_model:\n  model: openai:launch-model\n")
    monkeypatch.setattr(
        "ursa_dashboard.settings.system_config_paths", lambda: [system]
    )
    monkeypatch.setattr(
        "ursa_dashboard.settings.user_config_paths", lambda: [user]
    )
    monkeypatch.setattr(
        "ursa_dashboard.settings.dashboard_environment_config_layer",
        lambda: {"llm_model": {"model": "openai:env-model"}},
    )
    stored = GlobalSettings(
        llm=LLMSettings(
            model="openai:dashboard-model", credential_source="none"
        )
    )

    effective, _config = DashboardConfigResolver(
        group="default", explicit_config=explicit
    ).resolve(stored)
    assert effective.llm.model == "openai:launch-model"
    assert effective.embedding.model is None

    effective, _config = DashboardConfigResolver(group="default").resolve(
        stored
    )
    assert effective.llm.model == "openai:env-model"

    monkeypatch.setattr(
        "ursa_dashboard.settings.dashboard_environment_config_layer",
        lambda: {},
    )
    effective, _config = DashboardConfigResolver(group="default").resolve(
        stored
    )
    assert effective.llm.model == "openai:user-model"

    monkeypatch.setattr("ursa_dashboard.settings.user_config_paths", lambda: [])
    effective, _config = DashboardConfigResolver(group="default").resolve(
        stored
    )
    assert effective.llm.model == "openai:dashboard-model"


def test_user_named_provider_overrides_dashboard_direct_endpoint(
    tmp_path, monkeypatch
):
    user = tmp_path / "user.yaml"
    user.write_text(
        "\n".join([
            "inference_providers:",
            "  user_gateway:",
            "    base_url: https://user-models.example/v1",
            "llm_model:",
            "  model: openai:user-model",
            "  inference_provider: user_gateway",
        ])
    )
    monkeypatch.setattr(
        "ursa_dashboard.settings.user_config_paths", lambda: [user]
    )
    stored = GlobalSettings(
        llm=LLMSettings(
            model="openai:dashboard-model",
            base_url="https://dashboard-models.example/v1",
            credential_source="none",
        )
    )

    effective, _config = DashboardConfigResolver(group="default").resolve(
        stored
    )

    assert effective.llm.inference_provider == "user_gateway"
    assert effective.llm.base_url == "https://user-models.example/v1"
    assert effective.llm.model == "openai:user-model"


def test_dashboard_config_maps_cli_llm_model_to_dashboard_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SAFE_API_KEY", "secret")
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

    patch = dashboard_llm_patch_from_ursa_config(cfg_path, group="default")

    assert patch == {
        "llm": {
            "model": "openai:gpt-test",
            "base_url": "https://models.example.org/v1",
            "api_key_env": "SAFE_API_KEY",
            "credential_source": "environment",
            "max_tokens": 4096,
            "temperature": 0.3,
            "model_kwargs": {
                "seed": 123,
                "use_responses_api": True,
            },
        }
    }


def test_dashboard_config_maps_cli_emb_model_to_dashboard_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SAFE_EMBEDDING_KEY", "secret")
    cfg_path = tmp_path / "endpoint.yaml"
    cfg_path.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-test",
            "emb_model:",
            "  model: openai:text-embedding-3-large",
            "  base_url: https://models.example.org/v1",
            "  api_key_env: SAFE_EMBEDDING_KEY",
            "  dimensions: 1024",
            "  model_kwargs:",
            "    timeout: 60",
        ]),
        encoding="utf-8",
    )

    patch = dashboard_llm_patch_from_ursa_config(cfg_path, group="default")

    assert patch["embedding"] == {
        "model": "openai:text-embedding-3-large",
        "base_url": "https://models.example.org/v1",
        "api_key_env": "SAFE_EMBEDDING_KEY",
        "credential_source": "environment",
        "model_kwargs": {"dimensions": 1024, "timeout": 60},
    }


def test_dashboard_config_preserves_named_inference_provider(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LAB_MODELS_KEY", "secret")
    cfg_path = tmp_path / "providers.yaml"
    cfg_path.write_text(
        "\n".join([
            "inference_providers:",
            "  research_lab:",
            "    base_url: https://models.example.org/v1",
            "    api_key:",
            "      env: LAB_MODELS_KEY",
            "llm_model:",
            "  model: openai:lab-chat",
            "  inference_provider: research_lab",
        ]),
        encoding="utf-8",
    )

    patch = dashboard_llm_patch_from_ursa_config(cfg_path, group="default")

    assert patch["llm"]["inference_provider"] == "research_lab"
    assert patch["llm"]["base_url"] == "https://models.example.org/v1"
    assert patch["llm"]["api_key_env"] == "LAB_MODELS_KEY"
    assert patch["llm"]["credential_source"] == "environment"


def test_dashboard_provider_catalog_is_sanitized_and_canonicalizes_endpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "ursa-cache")
    monkeypatch.setenv(
        "URSA_DASHBOARD_WORKSPACE_ROOT", str(tmp_path / "dashboard")
    )
    monkeypatch.setenv("LAB_MODELS_KEY", "secret-for-test")
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        "\n".join([
            "inference_providers:",
            "  research_lab:",
            "    base_url: https://models.example.org/v1",
            "    api_key:",
            "      env: LAB_MODELS_KEY",
            "llm_model:",
            "  model: openai:lab-chat",
            "  inference_provider: research_lab",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("URSA_DASHBOARD_CONFIG", str(config_path))
    monkeypatch.setattr(
        "ursa_dashboard.app.list_provider_models",
        lambda _provider: [
            ProviderModel("lab-embed", "openai", type="embedding"),
            ProviderModel("lab-chat-realtime", "openai"),
            ProviderModel("lab-chat", "openai"),
            ProviderModel("whisper-1", "openai"),
        ],
    )

    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        catalog = client.get("/inference-providers")
        assert catalog.status_code == 200
        research_lab = next(
            item
            for item in catalog.json()["providers"]
            if item["name"] == "research_lab"
        )
        assert research_lab["base_url"] == "https://models.example.org/v1"
        assert research_lab["api_key_env"] == "LAB_MODELS_KEY"
        assert "secret-for-test" not in catalog.text

        models = client.get(
            "/inference-provider-models",
            params={"provider": "research_lab", "kind": "chat"},
        )
        assert models.status_code == 200
        assert [item["qualified_name"] for item in models.json()["models"]] == [
            "openai:lab-chat",
            "openai:lab-embed",
            "openai:whisper-1",
            "openai:lab-chat-realtime",
        ]

        updated = client.patch(
            "/settings",
            json={
                "patch": {
                    "llm": {
                        "inference_provider": "research_lab",
                        "base_url": "https://untrusted.example/v1",
                        "model": "openai:lab-chat",
                    }
                }
            },
        )
        assert updated.status_code == 200
        assert updated.json()["settings"]["llm"]["base_url"] == (
            "https://models.example.org/v1"
        )
        assert (
            updated.json()["settings"]["llm"]["inference_provider"]
            == "research_lab"
        )

        unknown = client.patch(
            "/settings",
            json={"patch": {"llm": {"inference_provider": "missing"}}},
        )
        assert unknown.status_code == 400


def test_explicit_dashboard_config_is_not_copied_into_settings_json(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "ursa-cache")
    monkeypatch.setenv(
        "URSA_DASHBOARD_WORKSPACE_ROOT", str(tmp_path / "dashboard")
    )
    config_path = tmp_path / "launch.yaml"
    config_path.write_text("llm_model:\n  model: openai:launch-only\n")
    monkeypatch.setenv("URSA_DASHBOARD_CONFIG", str(config_path))

    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        effective = client.get("/settings").json()["settings"]
        assert effective["llm"]["model"] == "openai:launch-only"

    stored = SettingsStore(tmp_path / "dashboard").load()
    assert stored.llm.model != "openai:launch-only"


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
        dashboard_llm_patch_from_ursa_config(cfg_path, group="default")


def test_dashboard_config_rejects_raw_embedding_api_key(tmp_path):
    cfg_path = tmp_path / "endpoint.yaml"
    cfg_path.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-test",
            "emb_model:",
            "  model: openai:text-embedding-3-large",
            "  api_key: secret-value",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="does not store raw emb_model.api_key"
    ):
        dashboard_llm_patch_from_ursa_config(cfg_path, group="default")


def test_dashboard_patch_enforces_selected_group_not_config_file_group(
    tmp_path, monkeypatch
):
    cfg_path = tmp_path / "endpoint.yaml"
    cfg_path.write_text(
        "\n".join([
            "group: source-group",
            "llm_model:",
            "  model: openai:gpt-test",
            "  base_url: https://models.example.org/v1",
        ]),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        "ursa.cli.config.enforce_group_base_url_policy",
        lambda base_url, group: calls.append((base_url, group)),
    )

    patch = dashboard_llm_patch_from_ursa_config(
        cfg_path, group="dashboard-group"
    )

    assert patch["llm"]["base_url"] == "https://models.example.org/v1"
    assert calls == [("https://models.example.org/v1", "dashboard-group")]


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
    monkeypatch.setattr(security, "URSA_CACHE_DIR", groups_dir)

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

    model_only_cfg = tmp_path / "model_only.yaml"
    model_only_cfg.write_text(
        "llm_model:\n  model: openai:gpt-updated\n",
        encoding="utf-8",
    )
    settings = apply_dashboard_config(
        store, model_only_cfg, group="my_safety_group"
    )
    assert settings.llm.model == "openai:gpt-updated"
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
    assert persisted["llm"]["model"] == "openai:gpt-updated"
    assert persisted["llm"]["base_url"] == "https://safe.example.org/v1"
    assert persisted["updated_at"] != original.updated_at
