from __future__ import annotations

from copy import deepcopy

import pytest
import yaml
from fastapi.testclient import TestClient

from ursa.cli.config import UrsaConfig
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import (
    CredentialConfigurationError,
    CredentialStoreError,
    MemoryCredentialStore,
    resolve_api_key,
)
from ursa_dashboard.user_config import STARTER_CONFIG


@pytest.fixture
def config_client(tmp_path, monkeypatch):
    config_path = tmp_path / "user" / "config.yaml"
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(
        "URSA_DASHBOARD_WORKSPACE_ROOT", str(tmp_path / "dashboard")
    )
    monkeypatch.delenv("URSA_DASHBOARD_CONFIG", raising=False)
    monkeypatch.setattr(
        "ursa_dashboard.settings.user_config_paths", lambda: [config_path]
    )
    monkeypatch.setattr(
        "ursa_dashboard.user_config.user_config_paths", lambda: [config_path]
    )
    secrets = MemoryCredentialStore()
    monkeypatch.setattr(
        "ursa_dashboard.app.KeyringCredentialStore", lambda **_: secrets
    )
    monkeypatch.setattr(
        "ursa_dashboard.credentials.KeyringCredentialStore", lambda **_: secrets
    )
    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        yield client, config_path, secrets


def edit_payload(client):
    draft = client.get("/user-config").json()
    return {
        key: draft[key]
        for key in ("revision", "providers", "llm_model", "emb_model")
    }


def test_starter_and_secure_save_reload_catalog_and_session_credentials(
    config_client,
):
    client, path, secrets = config_client
    draft = edit_payload(client)
    assert draft["llm_model"]["model"] == "openai:gpt-5.6-terra"
    assert draft["llm_model"]["options"] == {"reasoning": {"effort": "medium"}}
    assert not path.exists()
    provider = draft["providers"][0]
    provider.update(
        name="my_lab",
        base_url="https://models.example/v1",
        credential_mode="keyring",
        api_key="test-secret-only",
    )
    for name in ("llm_model", "emb_model"):
        draft[name]["inference_provider"] = "my_lab"
    response = client.put("/user-config", json=draft)
    assert response.status_code == 200, response.text
    assert "test-secret-only" not in response.text
    assert "test-secret-only" not in path.read_text()
    saved = yaml.safe_load(path.read_text())
    reference = saved["inference_providers"]["my_lab"]["api_key"]["keyring"]
    assert secrets.get_secret(reference) == "test-secret-only"
    assert UrsaConfig.from_file(path).llm_model.inference_provider == "my_lab"
    catalog = client.get("/inference-providers").json()["providers"]
    assert any(item["name"] == "my_lab" for item in catalog)
    effective = client.get("/settings").json()["settings"]["llm"]
    assert effective["base_url"] == "https://models.example/v1"
    assert effective["credential_source"] == "keyring"
    assert (
        resolve_api_key(effective, group="default", kind="llm", store=secrets)
        == "test-secret-only"
    )
    session = client.post(
        "/sessions",
        json={"agent_id": "chat_agent", "workspace_mode": "temporary"},
    ).json()["session"]
    assert session["llm"]["api_key_keyring"] == reference
    assert client.get(
        f"/sessions/{session['session_id']}/credentials/status"
    ).json()["llm"]["usable"]
    # Replacing a provider key does not invalidate a previous session snapshot.
    replacement = edit_payload(client)
    next(p for p in replacement["providers"] if p["name"] == "my_lab").update(
        credential_mode="keyring", api_key="replacement-secret"
    )
    assert client.put("/user-config", json=replacement).status_code == 200
    assert (
        resolve_api_key(
            session["llm"], group="default", kind="llm", store=secrets
        )
        == "test-secret-only"
    )
    changed_endpoint = session["llm"] | {
        "base_url": "https://another.example/v1"
    }
    with pytest.raises(CredentialConfigurationError, match="not approved"):
        resolve_api_key(
            changed_endpoint, group="default", kind="llm", store=secrets
        )


def test_existing_config_preserves_unrelated_fields_secrets_and_backup(
    config_client,
):
    client, path, secrets = config_client
    path.parent.mkdir()
    original = deepcopy(STARTER_CONFIG)
    original["agent_config"] = {"chat": {"custom_option": True}}
    original["inference_providers"]["openai"]["ssl_verify"] = False
    original["inference_providers"]["openai"]["api_key"] = "existing-secret"
    path.write_text(
        "# Keep an exact backup of this comment\n" + yaml.safe_dump(original)
    )
    original_bytes = path.read_bytes()
    response = client.get("/user-config")
    assert response.status_code == 200
    assert "existing-secret" not in response.text
    draft = edit_payload(client)
    draft["llm_model"]["model"] = "openai:new-model"
    updated = client.put("/user-config", json=draft)
    assert updated.status_code == 200, updated.text
    saved = yaml.safe_load(path.read_text())
    assert saved["agent_config"] == original["agent_config"]
    provider = saved["inference_providers"]["openai"]
    assert provider["ssl_verify"] is False
    assert (
        provider["base_url"]
        == original["inference_providers"]["openai"]["base_url"]
    )
    assert (
        secrets.get_secret(provider["api_key"]["keyring"]) == "existing-secret"
    )
    assert "existing-secret" not in path.read_text()
    assert (
        next(path.parent.glob("config.backup-*.yaml")).read_bytes()
        == original_bytes
    )


def test_concurrent_edit_is_not_overwritten(config_client):
    client, path, _secrets = config_client
    draft = edit_payload(client)
    path.parent.mkdir()
    path.write_text("llm_model:\n  model: openai:external-change\n")
    before = path.read_bytes()
    response = client.put("/user-config", json=draft)
    assert response.status_code == 409
    assert path.read_bytes() == before


def test_failed_secret_write_does_not_create_config(config_client, monkeypatch):
    client, path, secrets = config_client
    draft = edit_payload(client)
    draft["providers"][0].update(
        credential_mode="keyring", api_key="do-not-log"
    )

    def fail(*_args):
        raise CredentialStoreError("System keyring unavailable")

    monkeypatch.setattr(secrets, "set_secret", fail)
    response = client.put("/user-config", json=draft)
    assert response.status_code == 400
    assert "do-not-log" not in response.text
    assert not path.exists()


def test_model_test_uses_unsaved_key_without_persisting(
    config_client, monkeypatch
):
    client, path, secrets = config_client
    draft = edit_payload(client)
    draft["providers"][0].update(
        credential_mode="keyring", api_key="draft-secret"
    )
    calls = []

    async def probe(config, kind):
        calls.append((config.llm_model.api_key.get_secret_value(), kind))

    monkeypatch.setattr("ursa_dashboard.app.probe_model", probe)
    response = client.post("/user-config/test", json=draft)
    assert response.status_code == 200, response.text
    assert calls == [("draft-secret", "chat")]
    assert not path.exists()
    assert secrets._values == {}

    async def failed_probe(*_args):
        raise ValueError("upstream echoed draft-secret")

    monkeypatch.setattr("ursa_dashboard.app.probe_model", failed_probe)
    failed = client.post("/user-config/test", json=draft)
    assert failed.status_code == 400
    assert "draft-secret" not in failed.text


def test_invalid_config_and_cross_origin_write_are_rejected(config_client):
    client, path, _secrets = config_client
    draft = edit_payload(client)
    draft["llm_model"]["inference_provider"] = "missing"
    response = client.put("/user-config", json=draft)
    assert response.status_code == 400
    assert not path.exists()
    response = client.put(
        "/user-config",
        json=edit_payload(client),
        headers={"origin": "https://not-the-dashboard.example"},
    )
    assert response.status_code == 403
    assert not path.exists()


def test_walkthrough_dismissal_persists(config_client):
    client, _path, _secrets = config_client
    assert not client.get("/settings").json()["settings"]["ui"][
        "walkthrough_seen"
    ]
    assert (
        client.patch(
            "/settings", json={"patch": {"ui": {"walkthrough_seen": True}}}
        ).status_code
        == 200
    )
    assert client.get("/settings").json()["settings"]["ui"]["walkthrough_seen"]


def test_default_config_keeps_dotfile_symlink(config_client, tmp_path):
    client, path, _secrets = config_client
    target = tmp_path / "dotfiles" / "ursa.yaml"
    target.parent.mkdir()
    target.write_text(yaml.safe_dump(STARTER_CONFIG))
    path.parent.mkdir()
    path.symlink_to(target)
    draft = edit_payload(client)
    draft["llm_model"]["model"] = "openai:changed-through-link"
    response = client.put("/user-config", json=draft)
    assert response.status_code == 200, response.text
    assert path.is_symlink()
    assert (
        yaml.safe_load(target.read_text())["llm_model"]["model"]
        == "openai:changed-through-link"
    )
    assert next(target.parent.glob("config.backup-*.yaml")).is_file()


def test_model_discovery_uses_draft_and_shared_sorter(
    config_client, monkeypatch
):
    from ursa.util.inference_providers import ProviderModel

    client, path, _secrets = config_client
    draft = edit_payload(client)
    draft["providers"][0].update(
        credential_mode="keyring", api_key="temporary-discovery-key"
    )

    def models(config):
        assert config.api_key.get_secret_value() == "temporary-discovery-key"
        return [
            ProviderModel("whisper-1", "openai"),
            ProviderModel("gpt-test", "openai"),
        ]

    monkeypatch.setattr("ursa_dashboard.app.list_provider_models", models)
    response = client.post("/user-config/models", json=draft)
    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["models"]] == [
        "gpt-test",
        "whisper-1",
    ]
    assert "temporary-discovery-key" not in response.text
    assert not path.exists()


@pytest.mark.asyncio
async def test_connection_probe_invokes_model_with_bounded_output(monkeypatch):
    from ursa.cli.config import ChatModelConfig
    from ursa_dashboard.user_config import probe_model

    calls = []

    class Model:
        async def ainvoke(self, prompt):
            calls.append(prompt)
            return "OK"

    def init(model):
        assert model.max_completion_tokens == 256
        assert model.max_retries == 0
        return Model()

    monkeypatch.setattr(ChatModelConfig, "init_chat_model", init)
    config = UrsaConfig()
    await probe_model(config, "chat")
    assert calls == ["Reply with just the word OK."]
    assert config.llm_model.max_completion_tokens is None
