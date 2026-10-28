from copy import deepcopy

import pytest
from pydantic import SecretStr

from ursa.cli.config import UrsaConfig
from ursa.util.secrets import SecretReference

MERGE_CASES = [
    pytest.param(
        {
            "group": "restricted",
            "agent_config": {
                "research": {"temperature": 0.2, "nested": {"a": 1}}
            },
        },
        {
            "group": "science",
            "agent_config": {"research": {"max_steps": 4, "nested": {"b": 2}}},
        },
        {
            "group": "science",
            "agent_config": {
                "research": {
                    "temperature": 0.2,
                    "max_steps": 4,
                    "nested": {"a": 1, "b": 2},
                }
            },
        },
        id="mappings-recurse-and-scalars-replace",
    ),
    pytest.param(
        {
            "inference_providers": {
                "shared": {
                    "base_url": "https://models.example/v1",
                    "timeout": 10,
                    "client_options": {"a": 1},
                }
            }
        },
        {
            "inference_providers": {
                "shared": {
                    "ssl_verify": False,
                    "timeout": 20,
                    "client_options": {"b": 2},
                }
            }
        },
        {
            "inference_providers": {
                "shared": {
                    "base_url": "https://models.example/v1",
                    "ssl_verify": False,
                    "timeout": 20,
                    "client_options": {"a": 1, "b": 2},
                }
            }
        },
        id="provider-settings-merge-recursively",
    ),
    pytest.param(
        {
            "inference_providers": {
                "shared": {
                    "base_url": "https://models.example/v1",
                    "api_key": {"env": "SHARED_API_KEY"},
                }
            }
        },
        {"inference_providers": {"shared": {"api_key": {"keyring": True}}}},
        {
            "inference_providers": {
                "shared": {
                    "base_url": "https://models.example/v1",
                    "api_key": {"keyring": True},
                }
            }
        },
        id="secret-references-replace-atomically",
    ),
    pytest.param(
        {
            "mcp_servers": {
                "tools": {
                    "transport": "stdio",
                    "command": "old-tool-server",
                    "args": ["old"],
                },
                "unchanged": {
                    "transport": "stdio",
                    "command": "unchanged-server",
                },
            }
        },
        {
            "mcp_servers": {
                "tools": {
                    "transport": "streamable-http",
                    "url": "https://tools.example/mcp",
                }
            }
        },
        {
            "mcp_servers": {
                "tools": {
                    "transport": "streamable-http",
                    "url": "https://tools.example/mcp",
                },
                "unchanged": {
                    "transport": "stdio",
                    "command": "unchanged-server",
                },
            }
        },
        id="mcp-catalog-merges-but-server-definitions-replace",
    ),
]


def _dump(config: UrsaConfig):
    return config.model_dump(mode="python")


@pytest.mark.parametrize(("a", "b", "expected"), MERGE_CASES)
def test_config_a_plus_b_equals_expected(a, b, expected):
    a_before = deepcopy(a)
    b_before = deepcopy(b)

    batch = UrsaConfig().model_merge(a, b)
    sequential = UrsaConfig().model_merge(a).model_merge(b)
    expected_config = UrsaConfig.model_validate(expected)

    assert _dump(batch) == _dump(expected_config)
    assert _dump(sequential) == _dump(expected_config)
    assert a == a_before
    assert b == b_before


def test_later_layer_can_complete_optional_model():
    result = UrsaConfig().model_merge(
        {"emb_model": {"model_provider": "openai"}},
        {"emb_model": {"model": "text-embedding-3-large"}},
    )

    assert result.emb_model is not None
    assert result.emb_model.model == "text-embedding-3-large"
    assert result.emb_model.model_provider == "openai"


def test_incomplete_optional_model_fails_after_all_layers():
    with pytest.raises(ValueError, match="Field required"):
        UrsaConfig().model_merge({"emb_model": {"model_provider": "openai"}})


def test_legacy_provider_api_key_env_wins_over_lower_api_key():
    with pytest.warns(DeprecationWarning, match="api_key_env is deprecated"):
        result = UrsaConfig().model_merge(
            {
                "inference_providers": {
                    "shared": {"api_key": {"keyring": "old"}}
                }
            },
            {"inference_providers": {"shared": {"api_key_env": "NEW_API_KEY"}}},
        )

    assert result.inference_providers["shared"].api_key == SecretReference(
        env="NEW_API_KEY"
    )


def test_invalid_higher_priority_secret_does_not_inherit_lower_source():
    with pytest.raises(ValueError, match="exactly one source"):
        UrsaConfig().model_merge(
            {
                "inference_providers": {
                    "shared": {"api_key": {"env": "OLD_API_KEY"}}
                }
            },
            {"inference_providers": {"shared": {"api_key": {}}}},
        )


def test_validated_config_is_not_accepted_as_a_sparse_layer():
    layer = UrsaConfig().model_merge({"group": "science"})

    with pytest.raises(TypeError, match="raw configuration mappings"):
        UrsaConfig().model_merge(layer)  # type: ignore[arg-type]


def test_merge_result_does_not_alias_mutable_input():
    layer = {"agent_config": {"research": {"tags": ["original"]}}}

    result = UrsaConfig().model_merge(layer)
    result.agent_config["research"]["tags"].append("result-only")

    assert layer == {"agent_config": {"research": {"tags": ["original"]}}}


def test_merge_preserves_temporary_workspace_owner():
    base = UrsaConfig(workspace="tmp").resolve()

    result = base.model_merge({"group": "science"})

    assert result.workspace == base.workspace
    assert result._temp_workspace is base._temp_workspace


def test_partial_default_provider_update_is_inherited_by_default_model():
    merged = UrsaConfig().model_merge({
        "inference_providers": {"openai": {"ssl_verify": False}}
    })

    resolved = merged.resolve()

    assert resolved.llm_model.base_url == "https://api.openai.com/v1"
    assert resolved.llm_model.api_key is not None
    assert resolved.llm_model.ssl_verify is False


def test_ollama_provider_base_url_reaches_runtime_model():
    """Test for Issue #358, fixed in PR#352"""

    config = UrsaConfig().model_merge({
        "inference_providers": {
            "local_ollama": {"base_url": "http://127.0.0.1:11999"}
        },
        "llm_model": {
            "model": "ollama:gpt-oss:20b",
            "inference_provider": "local_ollama",
        },
    })

    resolved = config.resolve()

    assert resolved.llm_model.model == "gpt-oss:20b"
    assert resolved.llm_model.model_provider == "ollama"
    assert resolved.llm_model.base_url == "http://127.0.0.1:11999"
    assert resolved.llm_model.kwargs["base_url"] == "http://127.0.0.1:11999"


def test_literal_secret_replaces_secret_reference():
    result = UrsaConfig().model_merge(
        {
            "inference_providers": {
                "shared": {"api_key": {"env": "SHARED_API_KEY"}}
            }
        },
        {"inference_providers": {"shared": {"api_key": "literal"}}},
    )

    api_key = result.inference_providers["shared"].api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "literal"
