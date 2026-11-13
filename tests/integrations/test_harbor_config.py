import json
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from ursa.cli.config import InferenceProviderConfig
from ursa.integrations.harbor import UrsaHarborAgent
from ursa.integrations.harbor.config import (
    URSA_AGENT,
    AgentConfig,
    ModelConfig,
    load_agent_config,
    load_provider_config,
    resolve_run,
)


def provider():
    return InferenceProviderConfig.model_validate({
        "base_url": "https://example.invalid/v1",
        "api_key": {"env": "TEST_PROVIDER_TOKEN"},
        "model_provider": "openai",
    })


def test_resolution_is_lazy_and_uses_common_connection_flags(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("TEST_PROVIDER_TOKEN", raising=False)
    agent = AgentConfig(
        harbor_agent="codex",
        provider_kwargs={"lab": {"reasoning_effort": "high"}},
    )
    original = agent.model_dump()
    result = resolve_run(
        Path("task"),
        agent,
        ModelConfig(name="org/model:tag", provider="lab"),
        provider(),
        base_dir=tmp_path,
        base_url="http://127.0.0.1:8080/v1",
    )
    assert result.task == tmp_path / "task"
    assert result.agent.model_name == "openai/org/model:tag"
    assert result.agent.base_url == "http://127.0.0.1:8080/v1"
    assert result.agent.kwargs["reasoning_effort"] == "high"
    args = result.cli_args()
    assert args[args.index("--base-url") + 1] == result.agent.base_url
    assert args[args.index("--api-key-name") + 1] == "URSA_HARBOR_API_KEY"
    monkeypatch.setenv("TEST_PROVIDER_TOKEN", "not-for-cli")
    assert result.environment() == {"URSA_HARBOR_API_KEY": "not-for-cli"}
    assert "not-for-cli" not in repr(result) + json.dumps(args)
    assert agent.model_dump() == original


def test_ursa_native_model_and_endpoint_override(tmp_path, monkeypatch):
    monkeypatch.setenv("URSA_HARBOR_API_KEY", "test-key")
    result = resolve_run(
        Path("task"),
        AgentConfig(harbor_agent=URSA_AGENT, native_config={"use_web": False}),
        ModelConfig(name="test", provider="lab"),
        provider(),
        reasoning="high",
        base_url="http://relay.invalid/v1",
    )
    native = result.agent.kwargs["config"]
    assert native["llm_model"]["reasoning_effort"] == "high"
    assert native["inference_providers"]["lab"]["api_key"] == {
        "env": "URSA_HARBOR_API_KEY"
    }
    assert "base_url" not in native["inference_providers"]["lab"]
    harness = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name=result.agent.model_name,
        base_url=result.agent.base_url,
        model_api_key_name=result.agent.model_api_key_name,
        **result.agent.kwargs,
    )
    config, secrets = harness._materialize_native_config()
    assert config["llm_model"]["model"] == "test"
    assert config["llm_model"]["reasoning_effort"] == "high"
    assert (
        config["inference_providers"]["lab"]["base_url"]
        == "http://relay.invalid/v1"
    )
    assert "test-key" in secrets.values()


@pytest.mark.parametrize(
    "extension,contents",
    [
        ("json", '{"use_web": false}'),
        ("yaml", "use_web: false\n"),
        ("toml", "use_web = false\n"),
    ],
)
def test_native_file_is_relative_to_agent_file(tmp_path, extension, contents):
    native = tmp_path / f"native.{extension}"
    native.write_text(contents)
    file = tmp_path / "agent.yaml"
    file.write_text(
        f"harbor_agent: {URSA_AGENT}\nnative_config: {native.name}\n"
    )
    agent = load_agent_config(file)
    result = resolve_run(
        Path("task"),
        agent,
        ModelConfig(name="test", provider="lab"),
        provider(),
        base_dir=tmp_path / "unrelated",
    )
    assert result.agent.kwargs["config"]["use_web"] is False


def test_provider_keyring_alias_stays_a_reference(tmp_path):
    path = tmp_path / "provider.yaml"
    path.write_text("api_key: {keyring: true}\n")
    assert load_provider_config(path, "lab").api_key.keyring == "lab"


@pytest.mark.parametrize(
    "embedding,providers,hosts",
    [
        (
            {"base_url": "https://embeddings.invalid/v1"},
            {},
            ["127.0.0.1", "embeddings.invalid"],
        ),
        (
            {"inference_provider": "embedding"},
            {"embedding": {"base_url": "https://embeddings.invalid/v1"}},
            ["127.0.0.1", "embeddings.invalid"],
        ),
        ({"inference_provider": "lab"}, {}, ["127.0.0.1"]),
    ],
)
def test_ursa_chat_and_embedding_allowlist(embedding, providers, hosts):
    agent = AgentConfig(
        harbor_agent=URSA_AGENT,
        native_config={
            "emb_model": {
                "model": "text-embedding-3-small",
                "model_provider": "openai",
                "api_key": {"env": "UNSET_EMBEDDING_KEY"},
                **embedding,
            },
            "inference_providers": providers,
        },
    )
    original = agent.model_dump()
    result = resolve_run(
        Path("task"),
        agent,
        ModelConfig(name="test", provider="lab"),
        provider(),
        base_url="http://127.0.0.1:8080/v1",
    )
    assert result.agent.extra_allowed_hosts == hosts
    args = result.cli_args()
    assert [
        args[index + 1]
        for index, flag in enumerate(args)
        if flag == "--allow-agent-host"
    ] == hosts
    assert agent.model_dump() == original


def test_ursa_embedding_allowlist_keeps_default_chat_endpoint():
    result = resolve_run(
        Path("task"),
        AgentConfig(
            harbor_agent=URSA_AGENT,
            native_config={
                "emb_model": {
                    "model": "openai:text-embedding-3-small",
                    "base_url": "https://embeddings.invalid/v1",
                }
            },
        ),
        ModelConfig(name="test", provider="openai"),
        InferenceProviderConfig(),
    )
    assert result.agent.extra_allowed_hosts == [
        "api.openai.com",
        "embeddings.invalid",
    ]


@pytest.mark.parametrize(
    "settings,reasoning,match",
    [
        ({"harbor_agent": "oracle"}, "high", "does not support reasoning"),
        ({"harbor_agent": "oracle"}, None, "does not support base_url"),
        (
            {"harbor_agent": "opencode", "native_config": {"foo": True}},
            None,
            "does not support native",
        ),
        (
            {"harbor_agent": "codex", "kwargs": {"invented_option": 1}},
            None,
            "Unknown option",
        ),
        (
            {"harbor_agent": "codex", "kwargs": {"reasoning_effort": "low"}},
            "high",
            "Reasoning conflicts",
        ),
    ],
)
def test_capability_and_option_validation(settings, reasoning, match):
    with pytest.raises(ValueError, match=match):
        resolve_run(
            Path("task"),
            AgentConfig.model_validate(settings),
            ModelConfig(name="test", provider="lab"),
            provider(),
            reasoning=reasoning,
        )


def test_opencode_chat_format_uses_harbor_connection_resolution(
    tmp_path, monkeypatch
):
    from harbor.agents.installed.opencode import (
        OpenCode,
        build_opencode_provider_config,
    )

    monkeypatch.setenv("URSA_HARBOR_API_KEY", "test-key")
    result = resolve_run(
        Path("task"),
        AgentConfig(
            harbor_agent="opencode",
            api_format="openai_chat_completions",
            reasoning_kwarg="variant",
        ),
        ModelConfig(name="test", provider="lab"),
        provider(),
        reasoning="medium",
    )
    harness = OpenCode(
        logs_dir=tmp_path,
        model_name=result.agent.model_name,
        base_url=result.agent.base_url,
        model_api_key_name=result.agent.model_api_key_name,
        model_api_format=result.agent.model_api_format,
        **result.agent.kwargs,
    )
    config = build_opencode_provider_config(
        model_selector=result.agent.model_name,
        connection=harness.model_connection,
        native_config={},
        explicit_env={},
        base_url=result.agent.base_url,
        model_api_key_name=result.agent.model_api_key_name,
        model_api_format=result.agent.model_api_format,
    )
    assert config["openai"]["npm"] == "@ai-sdk/openai-compatible"
    assert (
        config["openai"]["options"]["baseURL"] == "https://example.invalid/v1"
    )


@pytest.mark.parametrize(
    "option", ["base_url_env", "native_config_format", "unknown"]
)
def test_removed_or_unknown_settings_are_rejected(option):
    with pytest.raises(ValueError):
        AgentConfig.model_validate({"harbor_agent": "codex", option: "value"})
