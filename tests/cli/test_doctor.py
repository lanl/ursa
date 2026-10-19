import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ursa.cli import build_parser, main
from ursa.cli import doctor as doctor_mod
from ursa.cli.config import UrsaConfig
from ursa.cli.doctor import (
    CheckResult,
    CheckStatus,
    DoctorContext,
    check_configuration,
    check_embedding_models,
    check_inference_providers,
    check_llm_models,
    check_mcp_servers,
    run_checks,
)


def test_doctor_parser_registers_json_without_running_checks(monkeypatch):
    checked = MagicMock(side_effect=AssertionError("check ran while parsing"))
    monkeypatch.setattr(doctor_mod, "run_checks", checked)

    parsed = build_parser().parse_args(["self", "doctor", "--json"])

    assert parsed.self.doctor.json is True
    checked.assert_not_called()


def test_configuration_reports_normal_resolution_or_its_failure():
    healthy = check_configuration(DoctorContext(config=SimpleNamespace()))
    broken = check_configuration(
        DoctorContext(config=None, config_error=ValueError("merge failed"))
    )

    assert healthy.status is CheckStatus.OK
    assert healthy.message == "Merged and resolved successfully"
    assert broken.status is CheckStatus.ERROR
    assert broken.message == "merge failed"


def test_runner_contains_failures_and_preserves_order():
    called = []

    def exits(_context):
        called.append("exit")
        raise SystemExit("check tried to exit")

    def healthy(_context):
        called.append("healthy")
        return CheckResult("healthy", CheckStatus.OK)

    report = run_checks(DoctorContext(config=None), checks=(exits, healthy))

    assert called == ["exit", "healthy"]
    assert [result.status for result in report.checks] == [
        CheckStatus.ERROR,
        CheckStatus.OK,
    ]


def test_provider_checks_validate_every_provider(monkeypatch):
    providers = {"good": object(), "bad": object()}

    def list_models(provider):
        if provider is providers["bad"]:
            raise RuntimeError("credentials rejected")
        return [object(), object()]

    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models", list_models
    )

    result = check_inference_providers(
        DoctorContext(config=SimpleNamespace(inference_providers=providers))
    )

    assert result.status is CheckStatus.ERROR
    assert [check.status for check in result.checks] == [
        CheckStatus.OK,
        CheckStatus.ERROR,
    ]
    assert "credentials rejected" in result.checks[1].message


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

    report = run_checks(
        DoctorContext(config=config),
        checks=(check_inference_providers, check_llm_models),
    )

    assert reads == [("ursa", "shared")]
    assert token not in json.dumps(report.as_dict())
    assert report.checks[0].checks[0].message == (
        "provider rejected [REDACTED]"
    )


class _FakeChatResponse:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class _FakeChatModel:
    def __init__(self):
        self.tool = False

    def invoke(self, _prompt):
        if self.tool:
            return _FakeChatResponse([
                {"name": "_ursa_health_echo", "args": {"value": "ursa"}}
            ])
        return _FakeChatResponse()

    def bind_tools(self, tools, *, tool_choice):
        assert tools[0].name == tool_choice == "_ursa_health_echo"
        self.tool = True
        return self

    def with_structured_output(self, schema, **_kwargs):
        return SimpleNamespace(invoke=lambda _prompt: schema(status="ok"))


def _chat_config(model):
    return SimpleNamespace(
        model="chat-test",
        inference_provider="test-provider",
        model_provider="test",
        base_url=None,
        init_chat_model=lambda: model,
    )


def test_llm_check_calls_chat_tool_use_and_structured_output():
    result = check_llm_models(
        DoctorContext(
            config=SimpleNamespace(llm_model=_chat_config(_FakeChatModel()))
        )
    )

    assert result.name == "LLM model"
    assert result.status is CheckStatus.OK
    assert [probe.name for probe in result.checks[0].checks] == [
        "Chat invocation",
        "Forced tool use",
        "Structured output",
    ]


def test_llm_probe_failure_does_not_suppress_later_probes():
    class ChatFailureModel(_FakeChatModel):
        def invoke(self, prompt):
            if not self.tool:
                raise RuntimeError("chat failed")
            return super().invoke(prompt)

    result = check_llm_models(
        DoctorContext(
            config=SimpleNamespace(llm_model=_chat_config(ChatFailureModel()))
        )
    )

    assert [probe.status for probe in result.checks[0].checks] == [
        CheckStatus.ERROR,
        CheckStatus.OK,
        CheckStatus.OK,
    ]


def test_llm_initialization_failure_is_reported():
    def fail():
        raise RuntimeError("missing SDK")

    config = _chat_config(_FakeChatModel())
    config.init_chat_model = fail

    result = check_llm_models(
        DoctorContext(config=SimpleNamespace(llm_model=config))
    )

    assert result.status is CheckStatus.ERROR
    assert result.checks[0].message == "missing SDK"


@pytest.mark.parametrize(
    ("vector", "status"),
    [([0.1, 0.2], CheckStatus.OK), ([], CheckStatus.ERROR)],
)
def test_embedding_check_calls_model(vector, status):
    embedding = SimpleNamespace(embed_query=lambda _text: vector)
    config = SimpleNamespace(
        model="embedding-test",
        inference_provider="test-provider",
        model_provider="test",
        base_url=None,
        init_embedding=lambda: embedding,
    )

    result = check_embedding_models(
        DoctorContext(config=SimpleNamespace(emb_model=config))
    )

    assert result.status is status


def test_mcp_check_connects_to_each_server(monkeypatch):
    class FakeClient:
        async def get_tools(self, *, server_name):
            if server_name == "bad":
                raise RuntimeError("cannot connect")
            return [object()]

    monkeypatch.setattr(
        "ursa.util.mcp.start_mcp_client", lambda _servers: FakeClient()
    )
    result = check_mcp_servers(
        DoctorContext(
            config=SimpleNamespace(
                mcp_servers={"good": object(), "bad": object()}
            )
        )
    )

    assert result.status is CheckStatus.ERROR
    assert [check.status for check in result.checks] == [
        CheckStatus.OK,
        CheckStatus.ERROR,
    ]


@pytest.mark.parametrize(
    "header",
    [
        {"env": "MCP_TOKEN", "template": "Bearer %s"},
        "Bearer mcp-secret-token",
    ],
)
def test_mcp_check_redacts_resolved_header_secret(monkeypatch, header):
    token = "mcp-secret-token"
    monkeypatch.setenv("MCP_TOKEN", token)
    config = UrsaConfig.model_validate({
        "mcp_servers": {
            "tools": {
                "transport": "streamable-http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": header},
            }
        }
    }).resolve()

    def reject(servers):
        header = servers["tools"].headers["Authorization"]
        raise RuntimeError(f"rejected header {header}")

    monkeypatch.setattr("ursa.util.mcp.start_mcp_client", reject)

    result = check_mcp_servers(DoctorContext(config=config))

    assert token not in str(result.as_dict())
    assert result.checks[0].message == "rejected header [REDACTED]"


def test_mcp_check_redacts_stdio_environment_values(monkeypatch, capsys):
    token = "stdio-secret-token"
    config = UrsaConfig.model_validate({
        "mcp_servers": {
            "tools": {
                "transport": "stdio",
                "command": "mcp-tool",
                "env": {"TOOL_TOKEN": token},
            }
        }
    }).resolve()

    def reject(servers):
        leaked = servers["tools"].env["TOOL_TOKEN"]
        raise RuntimeError(f"subprocess rejected {leaked}")

    monkeypatch.setattr("ursa.util.mcp.start_mcp_client", reject)

    result = check_mcp_servers(DoctorContext(config=config))
    report = CheckResult.group("URSA Doctor", [result])
    json_detail = json.dumps(report.as_dict())
    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)

    assert doctor_mod.run_doctor(config=config) == 1
    human_detail = capsys.readouterr().out

    assert token not in json_detail
    assert token not in human_detail
    assert "subprocess rejected [REDACTED]" in json_detail
    assert "subprocess rejected [REDACTED]" in human_detail


def test_main_doctor_json_writes_one_document(monkeypatch, capsys):
    report = CheckResult.group(
        "URSA Doctor",
        [CheckResult("broken", CheckStatus.ERROR, "boom")],
    )
    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    with pytest.raises(SystemExit) as exit_info:
        main(["self", "doctor", "--json"])

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out) == report.as_dict()


def test_main_doctor_uses_explicit_config(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm_model:\n  model: openai:configured-model\n")
    run = MagicMock(return_value=0)
    monkeypatch.setattr(doctor_mod, "run_doctor", run)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

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
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(path), "self", "doctor", "--json"])

    assert exit_info.value.code == 1
    assert "LEAKME" not in capsys.readouterr().out


def test_human_output_falls_back_to_json_when_rich_rendering_fails(
    monkeypatch, capsys
):
    report = CheckResult.group(
        "URSA Doctor",
        [CheckResult("optional", CheckStatus.WARN, "missing")],
    )
    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)
    monkeypatch.setattr(
        doctor_mod,
        "_print_report",
        MagicMock(side_effect=RuntimeError("renderer failed")),
    )

    return_code = doctor_mod.run_doctor()

    assert return_code == 0
    assert json.loads(capsys.readouterr().out) == report.as_dict()


def test_unavailable_config_skips_external_checks():
    context = DoctorContext(config=None, config_error=ValueError("bad config"))

    report = run_checks(context)

    assert report.status is CheckStatus.ERROR
    assert report.checks[0].message == "bad config"
    assert all(
        result.status is CheckStatus.WARN for result in report.checks[1:]
    )
