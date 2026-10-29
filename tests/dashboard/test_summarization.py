"""Dashboard settings, metadata preview, and worker summarization defaults."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ursa import security
from ursa.util.model_context import summarization_threshold
from ursa_dashboard import worker_main
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.registry import REGISTRY
from ursa_dashboard.settings import SettingsStore, apply_dashboard_config


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("URSA_DASHBOARD_CONFIG", raising=False)
    monkeypatch.delenv("URSA_DASHBOARD_GROUP", raising=False)
    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        yield client


@pytest.mark.parametrize(
    "window,expected", [(128000, 89600), (8192, 5735), (None, 12000)]
)
def test_saved_settings_preview(client, monkeypatch, window, expected):
    seen = {}

    def init_llm(config, *, api_key_override):
        seen.update(config)
        assert api_key_override is None
        return SimpleNamespace(profile={"max_input_tokens": window})

    monkeypatch.setattr(worker_main, "_init_llm", init_llm)
    saved = client.patch(
        "/settings",
        json={
            "patch": {
                "llm": {
                    "model": "openai:custom",
                    "credential_source": "none",
                    "tokens_before_summarize": 12000,
                }
            }
        },
    )
    assert saved.status_code == 200
    preview = client.get("/settings/summarization")
    assert preview.status_code == 200
    assert preview.json() == {
        "context_window": window,
        "tokens_before_summarize": expected,
    }
    assert seen["model"] == "openai:custom"
    assert (
        client.get("/settings").json()["settings"]["llm"][
            "tokens_before_summarize"
        ]
        == 12000
    )


def test_session_preview_uses_session_settings(client, monkeypatch, tmp_path):
    client.patch(
        "/settings", json={"patch": {"llm": {"credential_source": "none"}}}
    )
    session = client.post(
        "/sessions",
        json={
            "agent_id": "chat_agent",
            "workspace_mode": "folder",
            "workspace_path": str(tmp_path / "workspace"),
        },
    ).json()["session"]
    session_id = session["session_id"]
    assert (
        client.patch(
            f"/sessions/{session_id}",
            json={
                "llm": {
                    "model": "openai:session-model",
                    "tokens_before_summarize": 18000,
                }
            },
        ).status_code
        == 200
    )

    def init_llm(config, **kwargs):
        assert config["model"] == "openai:session-model"
        return SimpleNamespace()

    monkeypatch.setattr(worker_main, "_init_llm", init_llm)
    preview = client.get(
        "/settings/summarization", params={"session_id": session_id}
    )
    assert preview.json() == {
        "context_window": None,
        "tokens_before_summarize": 18000,
    }
    assert (
        client.get("/settings").json()["settings"]["llm"][
            "tokens_before_summarize"
        ]
        == 50000
    )
    assert (
        client.patch(
            f"/sessions/{session_id}",
            json={
                "llm": {
                    "tokens_before_summarize": 0,
                }
            },
        ).status_code
        == 400
    )


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "bad"])
def test_invalid_fallback_is_not_persisted(client, invalid):
    response = client.patch(
        "/settings",
        json={
            "patch": {
                "llm": {
                    "tokens_before_summarize": invalid,
                }
            }
        },
    )
    assert response.status_code == 400
    assert (
        client.get("/settings").json()["settings"]["llm"][
            "tokens_before_summarize"
        ]
        == 50000
    )


def test_preview_respects_stored_credential_binding(client, monkeypatch):
    client.patch(
        "/settings",
        json={
            "patch": {
                "llm": {
                    "model": "openai:test",
                    "base_url": "https://first.example/v1",
                }
            }
        },
    )
    assert (
        client.put(
            "/credentials/llm", json={"api_key": "test-secret"}
        ).status_code
        == 200
    )
    client.patch(
        "/settings",
        json={
            "patch": {
                "llm": {
                    "base_url": "https://second.example/v1",
                }
            }
        },
    )

    def unexpected_init(*args, **kwargs):
        pytest.fail(
            "Must reject mismatched credential before model initialization"
        )

    monkeypatch.setattr(worker_main, "_init_llm", unexpected_init)
    response = client.get("/settings/summarization")
    assert response.status_code == 400
    assert "test-secret" not in response.text


def test_config_import_preserves_explicit_fallback(tmp_path):
    store = SettingsStore(tmp_path / "dashboard")
    config = tmp_path / "config.yaml"
    config.write_text(
        "tokens_before_summarize: 80000\nllm_model:\n  model: openai:test\n"
    )
    assert (
        apply_dashboard_config(
            store, config, group="default"
        ).llm.tokens_before_summarize
        == 80000
    )
    config.write_text("llm_model:\n  model: openai:other\n")
    assert (
        apply_dashboard_config(
            store, config, group="default"
        ).llm.tokens_before_summarize
        == 80000
    )


@pytest.mark.parametrize(
    "override,window,expected",
    [(None, None, 12000), (9000, None, 9000), (9000, 128000, 89600)],
)
def test_worker_applies_fallback_and_agent_override(
    tmp_path, monkeypatch, override, window, expected
):
    model = SimpleNamespace(profile={"max_input_tokens": window})
    monkeypatch.setattr(worker_main, "_init_llm", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        worker_main, "_init_embedding", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(worker_main, "inject_truststore_into_ssl", lambda: None)
    seen = {}

    class Adapter:
        async def ainvoke(self, **kwargs):
            return "done"

    def build_adapter(llm, config):
        seen["threshold"] = summarization_threshold(
            llm, config["tokens_before_summarize"]
        )
        return Adapter()

    monkeypatch.setitem(
        REGISTRY,
        "test",
        SimpleNamespace(build_adapter=build_adapter, build_inputs=lambda p: p),
    )
    argv = [
        "worker",
        "--agent-id",
        "test",
        "--run-id",
        "test",
        "--workspace-dir",
        str(tmp_path),
        "--output-json",
        str(tmp_path / "output.json"),
    ]
    for name, value in {
        "params": {},
        "agent-init": {}
        if override is None
        else {"tokens_before_summarize": override},
        "llm": {"model": "openai:test", "tokens_before_summarize": 12000},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        argv.extend([f"--{name}-json", str(path)])
    monkeypatch.setattr("sys.argv", argv)
    assert worker_main.main() == 0
    assert seen["threshold"] == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_worker_loads_shared_model_metadata(monkeypatch, explicit):
    model = SimpleNamespace()
    seen = {}

    def init_model(**kwargs):
        assert "tokens_before_summarize" not in kwargs
        return model

    def load_context(llm, *, explicit_profile):
        assert llm is model
        seen["explicit"] = explicit_profile
        llm.profile = {"max_input_tokens": 128000}

    monkeypatch.setattr("langchain.chat_models.init_chat_model", init_model)
    monkeypatch.setattr(
        "ursa.util.model_metadata.load_model_context", load_context
    )
    llm = worker_main._init_llm({
        "model": "openai:test",
        "tokens_before_summarize": 12000,
        "model_kwargs": {"profile": {"max_input_tokens": 128000}}
        if explicit
        else {},
    })
    assert seen["explicit"] is explicit
    assert summarization_threshold(llm, 12000) == 89600
