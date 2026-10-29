import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from ursa.cli import doctor as doctor_mod
from ursa.cli import main
from ursa.cli.config import UrsaConfig
from ursa.cli.doctor import (
    CheckStatus,
    DoctorContext,
    check_inference_providers,
    check_llm_models,
    check_mcp_server,
    check_mcp_servers,
)


@pytest.fixture(autouse=True)
def _inline_health_probes(monkeypatch):
    def run(probe, *args, timeout):
        result = probe(*args)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    monkeypatch.setattr(doctor_mod, "run_probe_with_deadline", run)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)


class _FakeChatModel:
    tool = False

    def invoke(self, _prompt):
        call = {"name": "_ursa_health_echo", "args": {"value": "ursa"}}
        return SimpleNamespace(tool_calls=[call] if self.tool else [])

    def bind_tools(self, tools, *, tool_choice):
        assert tools[0].name == tool_choice == "_ursa_health_echo"
        self.tool = True
        return self

    def with_structured_output(self, schema, **_kwargs):
        return SimpleNamespace(invoke=lambda _prompt: schema(status="ok"))


def test_shared_key_is_resolved_once_and_redacted(monkeypatch):
    token, reads = "very-secret-token", []
    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(
            get_password=lambda service, username: reads.append((
                service,
                username,
            ))
            or token
        ),
    )
    config = UrsaConfig.model_validate({
        "inference_providers": {"shared": {"api_key": {"keyring": True}}},
        "llm_model": {"model": "openai:test", "inference_provider": "shared"},
    }).resolve()
    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models",
        lambda provider: (_ for _ in ()).throw(
            RuntimeError(f"rejected {provider.api_key.get_secret_value()}")
        ),
    )
    monkeypatch.setattr(
        type(config.llm_model),
        "init_chat_model",
        lambda _self: _FakeChatModel(),
    )
    context = DoctorContext(config)
    results = check_inference_providers(context), check_llm_models(context)
    assert reads == [("ursa", "shared")]
    assert token not in json.dumps([result.as_dict() for result in results])
    assert results[0].checks[0].message == "rejected [REDACTED]"


def test_external_probes_are_bounded(monkeypatch):
    calls = []

    def timeout(probe, *args, timeout):
        calls.append((probe.__name__, timeout))
        raise TimeoutError("timed out")

    monkeypatch.setattr(doctor_mod, "run_probe_with_deadline", timeout)
    context = DoctorContext(
        SimpleNamespace(inference_providers={"provider": object()})
    )
    assert check_inference_providers(context).status is CheckStatus.ERROR
    assert calls == [("probe_inference_provider", 30.0)]


def test_non_mcp_http_endpoint_is_a_normal_check_failure(monkeypatch):
    class Client:
        async def get_tools(self, *, server_name):
            assert server_name == "not-mcp"
            raise ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [RuntimeError("405 Method Not Allowed")],
            )

    monkeypatch.setattr(
        "ursa.util.mcp.start_mcp_client", lambda _servers: Client()
    )
    result = check_mcp_server(
        "not-mcp",
        {"transport": "streamable_http", "url": "https://google.com"},
    )
    assert result.status is CheckStatus.ERROR
    assert result.message == (
        "The endpoint does not accept MCP requests. Check that the URL "
        "points to an MCP server."
    )


def test_complete_report_checks_every_section(monkeypatch):
    config = UrsaConfig.model_validate({
        "inference_providers": {"company": {}},
        "llm_model": {
            "model": "chat-test",
            "model_provider": "openai",
            "inference_provider": "company",
        },
        "emb_model": {
            "model": "embedding-test",
            "model_provider": "openai",
            "inference_provider": "company",
        },
        "mcp_servers": {"tools": {"transport": "stdio", "command": "tool"}},
    })
    monkeypatch.setattr(
        doctor_mod, "run_probe_with_deadline", lambda *_a, **_k: "ok"
    )
    report = doctor_mod.build_report(config)
    assert report.status is CheckStatus.OK
    assert "/".join(item.name for item in report.checks) == (
        "Configuration/Inference providers/LLM model/Embedding model/MCP servers"
    )
    assert doctor_mod.run_doctor(config=config) == 0


_TOKEN = "mcp-secret-token"


@pytest.mark.parametrize(
    "server",
    [
        {
            "transport": "streamable-http",
            "url": "https://example.test/mcp",
            "headers": {"Authorization": value},
        }
        for value in (
            {"env": "MCP_TOKEN", "template": "Bearer %s"},
            f"Bearer {_TOKEN}",
        )
    ]
    + [{"transport": "stdio", "command": "mcp-tool", "env": {"TOKEN": _TOKEN}}],
)
def test_mcp_redacts_all_secret_sources(monkeypatch, server):
    monkeypatch.setenv("MCP_TOKEN", _TOKEN)
    config = UrsaConfig.model_validate({
        "mcp_servers": {"tools": server}
    }).resolve()

    def reject(servers):
        configured = servers["tools"]
        values = getattr(configured, "headers", None) or configured.env
        raise RuntimeError(f"rejected {next(iter(values.values()))}")

    monkeypatch.setattr("ursa.util.mcp.start_mcp_client", reject)
    output = json.dumps(check_mcp_servers(DoctorContext(config)).as_dict())
    assert _TOKEN not in output
    assert "rejected [REDACTED]" in output


def test_invalid_literal_secret_is_not_echoed(monkeypatch, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(
        "llm_model:\n  model: openai:test\n  base_url: https://example.test/v1\n"
        "  inference_provider: openai\n  api_key: LEAKME\n"
    )
    with pytest.raises(SystemExit, match="1"):
        main(["--config", str(path), "self", "doctor", "--json"])
    output = capsys.readouterr().out
    assert "LEAKME" not in output
    assert json.loads(output)["status"] == "error"
