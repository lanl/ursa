from pathlib import Path

import pytest

import ursa.cli.config as config_mod


def test_interpolate_env_replaces_existing_variable(monkeypatch):
    monkeypatch.setenv("URSA_TEST_VAR", "world")

    assert config_mod.interpolate_env("hello ${URSA_TEST_VAR}") == "hello world"


def test_interpolate_env_uses_default_when_missing(
    monkeypatch,
):
    monkeypatch.delenv("URSA_MISSING_VAR", raising=False)

    assert (
        config_mod.interpolate_env("value ${URSA_MISSING_VAR:fallback}")
        == "value fallback"
    )


def test_interpolate_env_allows_colon_in_default(monkeypatch):
    monkeypatch.delenv("URSA_URL_VAR", raising=False)

    assert (
        config_mod.interpolate_env(
            "url=${URSA_URL_VAR:mysql://localhost:5432/db}"
        )
        == "url=mysql://localhost:5432/db"
    )


def test_interpolate_env_missing_variable_without_default_is_empty(
    monkeypatch,
):
    monkeypatch.delenv("URSA_EMPTY_VAR", raising=False)

    assert (
        config_mod.interpolate_env("start ${URSA_EMPTY_VAR} end")
        == "start  end"
    )


def test_deep_interp_env_recurses_nested_dictionaries(
    monkeypatch,
):
    monkeypatch.setenv("URSA_DEEP_VALUE", "galaxy")
    monkeypatch.delenv("URSA_DEEP_FALLBACK", raising=False)

    data = {
        "layer1": {
            "with_env": "prefix ${URSA_DEEP_VALUE} suffix",
            "with_default": "${URSA_DEEP_FALLBACK:nebula}",
        },
        "unchanged": 42,
    }

    result = config_mod.deep_interp_env(data)

    assert result == {
        "layer1": {
            "with_env": "prefix galaxy suffix",
            "with_default": "nebula",
        },
        "unchanged": 42,
    }
    # Confirm original structure is untouched
    assert data["layer1"]["with_env"] == "prefix ${URSA_DEEP_VALUE} suffix"


@pytest.mark.parametrize(
    ("cls", "model", "expected_model", "expected_provider"),
    [
        (
            config_mod.ModelConfig,
            "openai:gpt-5.4",
            "openai:gpt-5.4",
            None,
        ),
        (
            config_mod.ChatModelConfig,
            "openai:gpt-5.4",
            "gpt-5.4",
            "openai",
        ),
        (
            config_mod.ModelConfig,
            "ollama:nomic-embed-text:latest",
            "ollama:nomic-embed-text:latest",
            None,
        ),
        (
            config_mod.EmbModelConfig,
            "ollama:nomic-embed-text:latest",
            "nomic-embed-text:latest",
            "ollama",
        ),
    ],
)
def test_model_config_model_parsing_for_known_provider_prefix(
    cls, model, expected_model, expected_provider
):
    cfg = cls(model=model)

    assert cfg.model == expected_model
    assert cfg.model_provider == expected_provider


@pytest.mark.parametrize(
    (
        "cls",
        "model",
        "model_provider",
        "expected_model",
        "expected_provider",
        "raises_match",
    ),
    [
        (
            config_mod.ChatModelConfig,
            "claude-4.8:0",
            "anthropic",
            "claude-4.8:0",
            "anthropic",
            None,
        ),
        (
            config_mod.EmbModelConfig,
            "gemma4:latest",
            "ollama",
            "gemma4:latest",
            "ollama",
            None,
        ),
        (
            config_mod.ChatModelConfig,
            "claude-4.8:0",
            None,
            "claude-4.8:0",
            "openai",
            None,
        ),
        (
            config_mod.EmbModelConfig,
            "gemma4:latest",
            None,
            None,
            None,
            "Unable to infer model provider",
        ),
        (
            config_mod.ChatModelConfig,
            "anthropic:claude-4.8:0",
            None,
            "anthropic:claude-4.8:0",
            "openai",
            None,
        ),
        (
            config_mod.EmbModelConfig,
            "ollama:gemma4:latest",
            None,
            "gemma4:latest",
            "ollama",
            None,
        ),
        (
            config_mod.ChatModelConfig,
            "anthropic:claude-4.8:0",
            "anthropic",
            "claude-4.8:0",
            "anthropic",
            None,
        ),
        (
            config_mod.EmbModelConfig,
            "ollama:gemma4:latest",
            "ollama",
            "gemma4:latest",
            "ollama",
            None,
        ),
    ],
)
def test_model_config_colon_model_name_handling(
    cls,
    model,
    model_provider,
    expected_model,
    expected_provider,
    raises_match,
):
    kwargs = {"model": model}
    if model_provider is not None:
        kwargs["model_provider"] = model_provider

    if raises_match is not None:
        with pytest.raises(ValueError, match=raises_match):
            cls(**kwargs)
        return

    cfg = cls(**kwargs)

    assert cfg.model == expected_model
    assert cfg.model_provider == expected_provider


@pytest.mark.parametrize(
    "higher_priority",
    [
        {
            "model": "ollama:nomic-embed-text:latest",
            "base_url": "http://localhost:11434",
        },
        config_mod.EmbModelConfig(
            model="ollama:nomic-embed-text:latest",
            base_url="http://localhost:11434",
        ),
    ],
)
def test_emb_model_merge_base_url_overrides_inference_provider(
    higher_priority,
):
    base = config_mod.EmbModelConfig(
        model="text-embedding-3-large",
        model_provider="openai",
        inference_provider="openai",
    )

    merged = base.model_merge(higher_priority)

    assert merged.model == "nomic-embed-text:latest"
    assert merged.model_provider == "ollama"
    assert merged.base_url == "http://localhost:11434"
    assert merged.inference_provider is None


def test_model_merge_keeps_provider_defaults_resolvable():
    config = config_mod.UrsaConfig().model_merge({
        "llm_model": {"model": "openai:gpt-5.4"}
    })

    resolved = config.resolve()

    assert resolved.llm_model.inference_provider == "openai"
    assert resolved.llm_model.base_url == "https://api.openai.com/v1"
    assert resolved.llm_model.api_key.env == "OPENAI_API_KEY"


@pytest.mark.parametrize(
    (
        "base_data",
        "override_data",
        "expected_model",
        "expected_provider",
        "expected_base_url",
        "expected_inference_provider",
        "resolve_result",
        "expected_api_key_env",
        "expect_nested_type",
    ),
    [
        (
            {
                "emb_model": {
                    "model": "text-embedding-3-large",
                    "model_provider": "openai",
                    "inference_provider": "openai",
                }
            },
            {
                "emb_model": {
                    "model": "ollama:nomic-embed-text:latest",
                    "base_url": "http://localhost:11434",
                }
            },
            "nomic-embed-text:latest",
            "ollama",
            "http://localhost:11434",
            None,
            False,
            None,
            False,
        ),
        (
            {
                "emb_model": {
                    "model": "ollama:nomic-embed-text:latest",
                    "base_url": "http://localhost:11434",
                }
            },
            {
                "emb_model": {
                    "model": "text-embedding-3-large",
                    "model_provider": "openai",
                    "inference_provider": "openai",
                }
            },
            "text-embedding-3-large",
            "openai",
            None,
            "openai",
            False,
            None,
            False,
        ),
        (
            {
                "emb_model": {
                    "model": "text-embedding-3-large",
                    "model_provider": "openai",
                }
            },
            {
                "emb_model": {
                    "model": "ollama:nomic-embed-text:latest",
                    "base_url": "http://localhost:11434",
                }
            },
            "nomic-embed-text:latest",
            "ollama",
            "http://localhost:11434",
            None,
            False,
            None,
            False,
        ),
        (
            {},
            {
                "emb_model": {
                    "model": "text-embedding-3-large",
                    "model_provider": "openai",
                    "inference_provider": "openai",
                }
            },
            "text-embedding-3-large",
            "openai",
            "https://api.openai.com/v1",
            "openai",
            True,
            "OPENAI_API_KEY",
            False,
        ),
        (
            {},
            {
                "emb_model": {
                    "model": "ollama:nomic-embed-text:latest",
                    "base_url": "http://localhost:11434",
                }
            },
            "nomic-embed-text:latest",
            "ollama",
            "http://localhost:11434",
            None,
            False,
            None,
            True,
        ),
    ],
)
def test_ursa_config_emb_model_layering(
    base_data,
    override_data,
    expected_model,
    expected_provider,
    expected_base_url,
    expected_inference_provider,
    resolve_result,
    expected_api_key_env,
    expect_nested_type,
):
    base = config_mod.UrsaConfig.model_validate(base_data)

    merged = base.model_merge(override_data)

    if resolve_result:
        merged = merged.resolve()

    assert merged.emb_model is not None
    if expect_nested_type:
        assert isinstance(merged.emb_model, config_mod.EmbModelConfig)
    assert merged.emb_model.model == expected_model
    assert merged.emb_model.model_provider == expected_provider
    assert merged.emb_model.base_url == expected_base_url
    assert merged.emb_model.inference_provider == expected_inference_provider
    if expected_api_key_env is not None:
        assert merged.emb_model.api_key.env == expected_api_key_env


@pytest.mark.parametrize(
    (
        "base_config",
        "merge_input",
        "expected_group",
        "expected_workspace",
        "expected_fields_set",
    ),
    [
        (
            config_mod.UrsaConfig(),
            {"group": "science"},
            "science",
            Path("."),
            {"group"},
        ),
        (
            config_mod.UrsaConfig(workspace=Path("/tmp/custom-workspace")),
            config_mod.UrsaConfig().model_merge({"group": "science"}),
            "science",
            Path("/tmp/custom-workspace"),
            None,
        ),
    ],
)
def test_ursa_config_merge_sparse_layer_behavior(
    base_config,
    merge_input,
    expected_group,
    expected_workspace,
    expected_fields_set,
):
    merged = base_config.model_merge(merge_input)

    assert merged.group == expected_group
    assert merged.workspace == expected_workspace
    if expected_fields_set is not None:
        assert merged.model_fields_set == expected_fields_set
