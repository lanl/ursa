import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ursa.cli import doctor as doctor_mod
from ursa.cli import main
from ursa.cli.config import UrsaConfig
from ursa.cli.doctor import (
    CheckResult,
    CheckStatus,
    DoctorContext,
    check_embedding_models,
    check_inference_providers,
    check_llm_models,
    check_mcp_servers,
    run_checks,
)


@pytest.fixture(autouse=True)
def _inline_health_probes(monkeypatch):
    def run(probe, *args, timeout):
        result = probe(*args)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    monkeypatch.setattr(doctor_mod, "run_probe_with_deadline", run)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)


def test_runner_contains_section_failures(monkeypatch):
    def exits(_context):
        raise SystemExit("check tried to exit")

    monkeypatch.setattr(doctor_mod, "check_configuration", exits)
    report = run_checks(DoctorContext(config=None))

    assert report.checks[0].status is CheckStatus.ERROR
    assert all(r.status is CheckStatus.WARN for r in report.checks[1:])


def test_provider_check_does_not_probe_an_unused_openai_default(monkeypatch):
    config = (
        UrsaConfig()
        .model_merge({
            "inference_providers": {"anthropic": {}},
            "llm_model": {
                "model": "claude-test",
                "model_provider": "anthropic",
                "inference_provider": "anthropic",
            },
        })
        .resolve()
    )
    listed = []
    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models",
        lambda provider: listed.append(provider) or [],
    )

    result = check_inference_providers(DoctorContext(config=config))

    assert list(config.inference_providers) == ["openai", "anthropic"]
    assert result.status is CheckStatus.OK
    assert listed == [config.inference_providers["anthropic"]]


def test_doctor_resolves_shared_key_once_and_redacts_failures(monkeypatch):
    token = "very-secret-token"
    reads = []
    backend = SimpleNamespace(
        get_password=lambda service, username: reads.append((service, username))
        or token
    )
    monkeypatch.setattr("ursa.util.secrets.get_keyring", lambda: backend)
    config = UrsaConfig.model_validate({
        "inference_providers": {"shared": {"api_key": {"keyring": True}}},
        "llm_model": {
            "model": "openai:test",
            "inference_provider": "shared",
        },
    }).resolve()

    def reject(provider):
        raise RuntimeError(
            f"provider rejected {provider.api_key.get_secret_value()}"
        )

    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models", reject
    )
    monkeypatch.setattr(
        type(config.llm_model),
        "init_chat_model",
        lambda _self: _FakeChatModel(),
    )

    context = DoctorContext(config=config)
    provider = check_inference_providers(context)
    llm = check_llm_models(context)

    assert reads == [("ursa", "shared")]
    assert token not in json.dumps([provider.as_dict(), llm.as_dict()])
    assert provider.checks[0].message == "provider rejected [REDACTED]"


class _FakeChatModel:
    def __init__(self):
        self.tool = False

    def invoke(self, _prompt):
        call = {"name": "_ursa_health_echo", "args": {"value": "ursa"}}
        return SimpleNamespace(tool_calls=[call] if self.tool else [])

    def bind_tools(self, tools, *, tool_choice):
        assert tools[0].name == tool_choice == "_ursa_health_echo"
        self.tool = True
        return self

    def with_structured_output(self, schema, **_kwargs):
        return SimpleNamespace(invoke=lambda _prompt: schema(status="ok"))


def _model_config(model, **values):
    return SimpleNamespace(
        model=model,
        inference_provider="test-provider",
        **values,
    )


def _chat_config(model):
    return _model_config("chat-test", init_chat_model=lambda: model)


def _context(**config):
    return DoctorContext(config=SimpleNamespace(**config))


def test_every_external_probe_uses_the_bounded_runner(monkeypatch):
    calls = []

    def timeout(probe, *args, timeout):
        capability = args[-1] if probe.__name__ == "probe_chat_config" else None
        calls.append((probe.__name__, capability, timeout))
        raise TimeoutError("timed out")

    monkeypatch.setattr(doctor_mod, "run_probe_with_deadline", timeout)
    context = _context(
        inference_providers={"first": object(), "second": object()},
        llm_model=_chat_config(_FakeChatModel()),
        emb_model=_model_config("embedding-test"),
        mcp_servers={"first": object(), "second": object()},
    )

    results = [
        check(context)
        for check in (
            check_inference_providers,
            check_llm_models,
            check_embedding_models,
            check_mcp_servers,
        )
    ]
    assert all(result.status is CheckStatus.ERROR for result in results)
    assert calls == [
        ("probe_inference_provider", None, 30.0),
        ("probe_inference_provider", None, 30.0),
        ("probe_chat_config", "chat", 30.0),
        ("probe_chat_config", "tool", 30.0),
        ("probe_chat_config", "structured", 30.0),
        ("probe_embedding_model", None, 30.0),
        ("probe_mcp_server", None, 30.0),
        ("probe_mcp_server", None, 30.0),
    ]


def test_llm_probe_failure_does_not_suppress_later_probes():
    class ChatFailureModel(_FakeChatModel):
        def invoke(self, prompt):
            if not self.tool:
                raise RuntimeError("chat failed")
            return super().invoke(prompt)

    result = check_llm_models(
        _context(llm_model=_chat_config(ChatFailureModel()))
    )

    probes = result.checks[0].checks
    assert [(probe.name, probe.status) for probe in probes] == [
        ("Chat invocation", CheckStatus.ERROR),
        ("Forced tool use", CheckStatus.OK),
        ("Structured output", CheckStatus.OK),
    ]


@pytest.mark.parametrize(
    ("vector", "status"),
    [([0.1, 0.2], CheckStatus.OK), ([], CheckStatus.ERROR)],
)
def test_embedding_check_calls_model(vector, status):
    embedding = SimpleNamespace(embed_query=lambda _text: vector)
    config = _model_config("embedding-test", init_embedding=lambda: embedding)

    result = check_embedding_models(_context(emb_model=config))

    assert result.status is status


_MCP_TOKEN = "mcp-secret-token"


def _http_mcp(authorization):
    return {
        "transport": "streamable-http",
        "url": "https://example.test/mcp",
        "headers": {"Authorization": authorization},
    }


@pytest.mark.parametrize(
    "server",
    [
        _http_mcp({"env": "MCP_TOKEN", "template": "Bearer %s"}),
        _http_mcp(f"Bearer {_MCP_TOKEN}"),
        {
            "transport": "stdio",
            "command": "mcp-tool",
            "env": {"TOOL_TOKEN": _MCP_TOKEN},
        },
    ],
)
def test_mcp_redacts_human_and_json_output(monkeypatch, capsys, server):
    monkeypatch.setenv("MCP_TOKEN", _MCP_TOKEN)
    config = UrsaConfig.model_validate({
        "mcp_servers": {"tools": server}
    }).resolve()

    def reject(servers):
        configured = servers["tools"]
        values = getattr(configured, "headers", None) or configured.env
        raise RuntimeError(f"rejected {next(iter(values.values()))}")

    monkeypatch.setattr("ursa.util.mcp.start_mcp_client", reject)
    result = check_mcp_servers(DoctorContext(config=config))
    report = CheckResult.group("URSA Doctor", [result])
    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)

    assert doctor_mod.run_doctor(SimpleNamespace(json=True), config=config) == 1
    json_detail = capsys.readouterr().out
    assert doctor_mod.run_doctor(config=config) == 1
    human_detail = capsys.readouterr().out

    assert _MCP_TOKEN not in json_detail + human_detail
    assert "rejected [REDACTED]" in json_detail
    assert "rejected [REDACTED]" in human_detail


def test_main_doctor_json_writes_one_document(monkeypatch, capsys):
    report = CheckResult.group(
        "URSA Doctor",
        [CheckResult("broken", CheckStatus.ERROR, "boom")],
    )
    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)

    with pytest.raises(SystemExit) as exit_info:
        main(["self", "doctor", "--json"])

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out) == report.as_dict()


def test_main_doctor_uses_explicit_config(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm_model:\n  model: openai:configured-model\n")
    run = MagicMock(return_value=0)
    monkeypatch.setattr(doctor_mod, "run_doctor", run)

    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(path), "self", "doctor"])

    assert exit_info.value.code == 0
    assert run.call_args.kwargs["config"].llm_model.model == "configured-model"


def test_main_doctor_does_not_echo_invalid_literal_secret(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "config.yaml"
    path.write_text(
        "llm_model:\n"
        "  model: openai:test\n"
        "  base_url: https://models.example/v1\n"
        "  inference_provider: openai\n"
        "  api_key: LEAKME\n"
    )

    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(path), "self", "doctor", "--json"])

    assert exit_info.value.code == 1
    assert "LEAKME" not in capsys.readouterr().out


def test_unavailable_config_skips_external_checks():
    context = DoctorContext(config=None, config_error=ValueError("bad config"))

    report = run_checks(context)

    assert report.status is CheckStatus.ERROR
    assert [result.name for result in report.checks] == [
        "Configuration",
        "Inference providers",
        "LLM model",
        "Embedding model",
        "MCP servers",
    ]
    assert report.checks[0].message == "bad config"
    assert all(r.status is CheckStatus.WARN for r in report.checks[1:])
