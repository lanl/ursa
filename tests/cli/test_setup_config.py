from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import SecretStr

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
)
from ursa.cli.setup_config import (
    SetupError,
    apply_setup_defaults,
    format_setup_config_yaml,
    load_setup_config,
    parse_setup_model_yaml,
    prepare_provider_probe,
    reject_literal_secrets,
    replace_inference_providers,
    replace_mcp_servers,
    replace_models,
    save_setup_config,
    validate_setup_name,
    validate_setup_url,
)
from ursa.util.mcp import parse_server_parameters
from ursa.util.secrets import SecretReference, SecretTemplate


def test_public_setup_value_validators_normalize_and_reject():
    assert validate_setup_name(" company-one ", "Provider") == "company-one"
    assert (
        validate_setup_url(" https://models.example/v1/ ", "Base URL")
        == "https://models.example/v1"
    )
    with pytest.raises(SetupError, match="must start with a letter"):
        validate_setup_name("1-company", "Provider")
    with pytest.raises(SetupError, match="complete HTTP URL"):
        validate_setup_url("models.example/v1", "Base URL")


def test_literal_secret_validation_preserves_installer_template_shorthand():
    value = {"headers": {"Authorization": {"template": "Bearer %s"}}}

    with pytest.raises(SetupError, match="Literal secrets cannot be saved"):
        reject_literal_secrets(value)

    reject_literal_secrets(value, allow_implicit_keyring_template=True)
    reject_literal_secrets({"env": {"API_TOKEN": "${TOOLS_TOKEN}"}})


def test_load_setup_config_preserves_placeholders_and_explicit_providers(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MODEL_URL", "https://resolved.example/v1")
    monkeypatch.setenv("MCP_SECRET", "resolved-secret")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    base_url: "${MODEL_URL}"
    ssl_verify: false
llm_model:
  model: gpt-test
  inference_provider: company
mcp_servers:
  tools:
    transport: stdio
    command: tool
    env: {API_KEY: "${MCP_SECRET}"}
"""
    )

    config, llm_configured = load_setup_config(path)

    assert set(config.inference_providers) == {"company"}
    assert config.inference_providers["company"].base_url == "${MODEL_URL}"
    assert config.inference_providers["company"].ssl_verify is True
    assert config.mcp_servers["tools"].env == {"API_KEY": "${MCP_SECRET}"}
    assert llm_configured


def test_apply_setup_defaults_uses_canonical_mcp_parser(tmp_path):
    config, llm_configured = load_setup_config(tmp_path / "missing.yaml")

    llm_configured = apply_setup_defaults(
        config,
        {
            "inference_providers": [
                {
                    "name": "company",
                    "base_url": "https://models.example/v1",
                }
            ],
            "llm_model": {
                "model": "gpt-test",
                "inference_provider": "company",
            },
            "mcp_servers": {
                "tools": {
                    "transport": "streamable_http",
                    "url": "https://tools.example/mcp",
                    "headers": {"Authorization": {"template": "Bearer %s"}},
                }
            },
        },
        llm_configured=llm_configured,
    )

    assert llm_configured
    assert config.inference_providers["company"].api_key == SecretReference(
        keyring=True
    )
    assert config.mcp_servers["tools"].headers == {
        "Authorization": SecretTemplate(keyring=True, template="Bearer %s")
    }


def test_save_setup_config_rolls_back_keyring_when_file_save_fails(
    monkeypatch, tmp_path
):
    config = UrsaConfig.model_validate({
        "inference_providers": {"company": {"api_key": {"keyring": True}}},
        "llm_model": {
            "model": "gpt-test",
            "inference_provider": "company",
        },
    })
    store = MagicMock()
    store.get_password.side_effect = lambda _service, username: {
        "company": "old-secret",
        "tools": None,
    }[username]
    monkeypatch.setattr("ursa.cli.setup_config.get_keyring", lambda: store)
    monkeypatch.setattr(
        "ursa.cli.setup_config.save_config_file",
        MagicMock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        save_setup_config(
            tmp_path / "config.yaml",
            config,
            {"company": "new-secret", "tools": "mcp-secret"},
        )

    store.set_password.assert_any_call("ursa", "company", "old-secret")
    store.delete_password.assert_called_once_with("ursa", "tools")


def test_replace_providers_retargets_renamed_and_removed_references(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  old: {base_url: https://old.example/v1}
  removed: {base_url: https://removed.example/v1}
llm_model:
  model: chat-test
  model_provider: openai
  inference_provider: old
emb_model:
  model: emb-test
  model_provider: openai
  inference_provider: removed
"""
    )
    config, _ = load_setup_config(path)

    updated = replace_inference_providers(
        config,
        {
            "fallback": InferenceProviderConfig(
                base_url="https://fallback.example/v1"
            ),
            "renamed": InferenceProviderConfig(
                base_url="https://renamed.example/v1"
            ),
        },
        renamed={"old": "renamed"},
    )

    assert set(updated.inference_providers) == {"fallback", "renamed"}
    assert updated.llm_model.inference_provider == "renamed"
    assert updated.emb_model is not None
    assert updated.emb_model.inference_provider == "fallback"
    assert config.llm_model.inference_provider == "old"
    assert config.emb_model is not None
    assert config.emb_model.inference_provider == "removed"


def test_replace_models_is_atomic_and_validates_provider_references(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company: {base_url: https://models.example/v1}
llm_model:
  model: old-chat
  model_provider: openai
  inference_provider: company
"""
    )
    config, _ = load_setup_config(path)
    chat = ChatModelConfig(
        model="new-chat",
        model_provider="openai",
        inference_provider="company",
    )
    embedding = EmbModelConfig(
        model="new-embedding",
        model_provider="openai",
        inference_provider="company",
    )

    updated = replace_models(config, chat, embedding)

    assert updated.llm_model.model == "new-chat"
    assert updated.emb_model == embedding
    assert config.llm_model.model == "old-chat"
    with pytest.raises(SetupError, match="unknown inference_provider"):
        replace_models(
            config,
            ChatModelConfig(model="bad", inference_provider="missing"),
            None,
        )
    assert config.llm_model.model == "old-chat"


def test_parse_setup_model_yaml_replaces_complete_block_and_deletes_fields():
    original = ChatModelConfig(
        model="old-chat",
        model_provider="anthropic",
        inference_provider="company",
        api_key=SecretStr("old-model-secret"),
        max_completion_tokens=2048,
        old_extra="remove-me",
    )

    parsed = parse_setup_model_yaml(
        "model: new-chat\ninference_provider: company\n",
        ChatModelConfig,
        original,
        original.model_dump(mode="json"),
        {"company"},
    )

    assert isinstance(parsed, ChatModelConfig)
    assert parsed.model == "new-chat"
    assert parsed.api_key is None
    assert parsed.max_completion_tokens is None
    assert parsed.model_extra == {}
    assert parsed.model_fields_set == {"model", "inference_provider"}
    assert original.max_completion_tokens == 2048
    assert original.model_extra == {"old_extra": "remove-me"}


def test_parse_setup_model_yaml_accepts_null_embedding():
    original = EmbModelConfig(
        model="old-embedding", inference_provider="company"
    )

    assert (
        parse_setup_model_yaml(
            "null\n",
            EmbModelConfig,
            original,
            original.model_dump(mode="json"),
            {"company"},
        )
        is None
    )


def test_replace_mcp_servers_is_atomic_and_formats_review_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company: {base_url: https://models.example/v1}
llm_model:
  model: chat-test
  model_provider: openai
  inference_provider: company
"""
    )
    config, _ = load_setup_config(path)
    server = parse_server_parameters({
        "transport": "streamable_http",
        "url": "https://tools.example/mcp",
        "headers": {"Authorization": {"env": "TOOLS_TOKEN"}},
    })

    updated = replace_mcp_servers(config, {"tools": server})

    assert config.mcp_servers == {}
    assert updated.mcp_servers["tools"].url == "https://tools.example/mcp"
    assert yaml.safe_load(format_setup_config_yaml(updated)) == (
        updated.editable_values()
    )
    with pytest.raises(SetupError, match="both provider and MCP"):
        replace_mcp_servers(config, {"company": server})


def test_prepare_provider_probe_interpolates_and_uses_staged_secret(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MODEL_URL", "https://models.example/v1")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    base_url: "${MODEL_URL}"
    api_key: {keyring: true}
llm_model:
  model: chat-test
  model_provider: anthropic
  inference_provider: company
"""
    )
    config, _ = load_setup_config(path)
    staged = SecretStr("pending-secret")

    runtime = prepare_provider_probe(config, "company", staged)

    assert runtime.base_url == "https://models.example/v1"
    assert isinstance(runtime.api_key, SecretStr)
    assert runtime.api_key.get_secret_value() == "pending-secret"
    assert runtime.model_provider == "anthropic"
    assert config.inference_providers["company"].api_key == SecretReference(
        keyring=True
    )
