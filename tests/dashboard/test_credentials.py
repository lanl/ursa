from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import (
    CredentialConfigurationError,
    MemoryCredentialStore,
    assert_no_raw_api_key,
    credential_id,
    credential_target,
    resolve_api_key,
    store_api_key,
)
from ursa_dashboard.run_manager import RunManager
from ursa_dashboard.settings import SettingsStore


def test_stored_credential_is_bound_to_endpoint() -> None:
    store = MemoryCredentialStore()
    config = {
        "model": "openai:gpt-test",
        "base_url": "https://models.example.org/v1",
        "credential_source": "stored",
        "credential_id": credential_id("default", "llm"),
        "credential_target": "origin:https://models.example.org",
    }
    store_api_key(
        store,
        credential_id=credential_id("default", "llm"),
        target="origin:https://models.example.org",
        value="secret-value",
    )

    assert credential_target(config) == "origin:https://models.example.org"
    assert (
        resolve_api_key(
            config,
            group="default",
            kind="llm",
            store=store,
        )
        == "secret-value"
    )

    changed = config | {
        "base_url": "https://other.example.org/v1",
        "credential_target": "origin:https://other.example.org",
    }
    with pytest.raises(CredentialConfigurationError, match="not approved"):
        resolve_api_key(
            changed,
            group="default",
            kind="llm",
            store=store,
        )


def test_literal_api_keys_are_rejected_recursively() -> None:
    with pytest.raises(CredentialConfigurationError, match="literal API key"):
        assert_no_raw_api_key(
            {"model_kwargs": {"api_key": "secret-value"}},
            context="test",
        )


def test_embedding_can_reuse_llm_key_only_for_same_target() -> None:
    store = MemoryCredentialStore()
    store_api_key(
        store,
        credential_id=credential_id("default", "llm"),
        target="origin:https://models.example.org",
        value="shared-secret",
    )
    config = {
        "model": "openai:text-embedding-test",
        "base_url": "https://models.example.org/embeddings",
        "credential_source": "llm",
    }

    assert (
        resolve_api_key(
            config,
            group="default",
            kind="embedding",
            store=store,
        )
        == "shared-secret"
    )

    with pytest.raises(CredentialConfigurationError, match="not approved"):
        resolve_api_key(
            config | {"base_url": "https://other.example.org/v1"},
            group="default",
            kind="embedding",
            store=store,
        )


def test_new_settings_default_to_secure_storage_and_legacy_uses_env(
    tmp_path,
) -> None:
    store = SettingsStore(tmp_path / "dashboard")
    assert store.load().llm.credential_source == "stored"

    data = json.loads(store.path.read_text(encoding="utf-8"))
    data["llm"].pop("credential_source")
    store.path.write_text(json.dumps(data), encoding="utf-8")

    assert store.load().llm.credential_source == "environment"


def test_worker_environment_excludes_model_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_MODEL_KEY", "secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "another-secret")
    monkeypatch.setenv("GH_TOKEN", "token-secret")
    monkeypatch.setenv("SAFE_SETTING", "kept")
    rec = {
        "llm": {"api_key_env": "MY_MODEL_KEY"},
        "embedding": {},
    }

    env = RunManager._worker_environment(rec, project_root="/project")

    assert "MY_MODEL_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "GH_TOKEN" not in env
    assert env["SAFE_SETTING"] == "kept"
    assert env["PYTHONPATH"].split(":", 1)[0] == "/project"


def test_credential_api_never_persists_or_returns_raw_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")
    store = MemoryCredentialStore()
    secret = "ursa-test-secret-never-persist"

    with TestClient(create_app(credential_store=store)) as client:
        settings_response = client.patch(
            "/settings",
            json={
                "patch": {
                    "llm": {
                        "model": "openai:gpt-test",
                        "base_url": "https://models.example.org/v1",
                        "credential_source": "none",
                    }
                }
            },
        )
        assert settings_response.status_code == 200

        response = client.put("/credentials/llm", json={"api_key": secret})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "kind": "llm",
            "source": "stored",
            "configured": True,
            "usable": True,
            "needs_reentry": False,
            "target": "origin:https://models.example.org",
        }
        assert secret not in response.text

        settings = client.get("/settings")
        assert settings.status_code == 200
        assert secret not in settings.text
        assert settings.json()["settings"]["llm"]["credential_source"] == (
            "stored"
        )

        mismatch = client.patch(
            "/settings",
            json={
                "patch": {"llm": {"base_url": "https://other.example.org/v1"}}
            },
        )
        assert mismatch.status_code == 200
        status = client.get("/credentials/status").json()["llm"]
        assert status["configured"] is True
        assert status["usable"] is False
        assert status["needs_reentry"] is True

        deleted = client.delete("/credentials/llm")
        assert deleted.status_code == 200
        assert deleted.json()["source"] == "none"
        assert deleted.json()["configured"] is False

    dashboard_root = tmp_path / "ursa" / "default" / "dashboard"
    for path in dashboard_root.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(
                encoding="utf-8", errors="ignore"
            )


@pytest.mark.parametrize(
    "payload",
    [
        {"patch": {"llm": {"api_key": "secret-value"}}},
        {"patch": {"llm": {"model_kwargs": {"api_key": "secret-value"}}}},
    ],
)
def test_settings_api_rejects_literal_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path, payload: dict
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")

    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        response = client.patch("/settings", json=payload)

    assert response.status_code == 400
    assert "literal API key" in response.json()["detail"]
    assert "secret-value" not in response.text


def test_settings_api_rejects_forged_credential_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")

    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        response = client.patch(
            "/settings",
            json={
                "patch": {
                    "llm": {
                        "credential_id": "dashboard:default:llm",
                        "credential_target": "origin:https://evil.example",
                    }
                }
            },
        )

    assert response.status_code == 400
    assert "server-managed credential metadata" in response.json()["detail"]


def test_run_api_rejects_literal_key_before_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")

    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        response = client.post(
            "/runs",
            json={
                "agent_id": "chat_agent",
                "params": {"prompt": "hello"},
                "llm": {"api_key": "secret-value"},
            },
        )

    assert response.status_code == 400
    meta_dir = tmp_path / "ursa" / "default" / "dashboard" / "_meta"
    persisted = ""
    for path in meta_dir.rglob("*"):
        if path.is_file():
            persisted += path.read_text(encoding="utf-8", errors="ignore")
    assert "secret-value" not in persisted


def test_settings_file_contains_only_credential_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")
    store = MemoryCredentialStore()

    with TestClient(create_app(credential_store=store)) as client:
        client.patch(
            "/settings",
            json={"patch": {"llm": {"credential_source": "none"}}},
        )
        client.put("/credentials/llm", json={"api_key": "secret-value"})

    settings_path = (
        tmp_path / "ursa" / "default" / "dashboard" / "_meta" / "settings.json"
    )
    persisted = json.loads(settings_path.read_text(encoding="utf-8"))
    assert persisted["llm"]["credential_source"] == "stored"
    assert "api_key" not in persisted["llm"]
    assert "secret-value" not in settings_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_run_manager_delivers_key_only_through_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    store = MemoryCredentialStore()
    secret = "pipe-only-secret"
    secret_id = credential_id("default", "llm")
    store_api_key(
        store,
        credential_id=secret_id,
        target="origin:https://models.example.org",
        value=secret,
    )
    manager = RunManager(
        dashboard_root=tmp_path / "dashboard",
        credential_store=store,
        dashboard_group="default",
    )
    captured: dict = {}

    class FakeStdin:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, value: bytes) -> None:
            self.data.extend(value)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 4321
            self.returncode = 0
            self.stdin = FakeStdin()
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            return self.returncode

    async def fake_subprocess(*cmd, **kwargs):
        process = FakeProcess()
        captured.update({"cmd": cmd, "kwargs": kwargs, "process": process})
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)

    llm = {
        "model": "openai:gpt-test",
        "base_url": "https://models.example.org/v1",
        "credential_source": "stored",
        "credential_id": secret_id,
        "credential_target": "origin:https://models.example.org",
    }
    run = await manager.create_run(
        agent_id="chat_agent",
        params={"prompt": "hello"},
        agent_init={},
        llm=llm,
        runner={},
        extra={
            "embedding": {
                "model": None,
                "credential_source": "none",
            }
        },
    )
    await manager._execute_run(run["run_id"])

    process = captured["process"]
    payload = json.loads(bytes(process.stdin.data).decode("utf-8"))
    assert payload == {"llm_api_key": secret, "embedding_api_key": None}
    assert secret not in " ".join(captured["cmd"])
    assert secret not in json.dumps(captured["kwargs"]["env"])

    for path in (tmp_path / "dashboard").rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(
                encoding="utf-8", errors="ignore"
            )
