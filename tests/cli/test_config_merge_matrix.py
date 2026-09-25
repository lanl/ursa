from copy import deepcopy

import pytest
from pydantic import SecretStr

from ursa.cli.config import ChatModelConfig, EmbModelConfig, UrsaConfig
from ursa.util.mcp import transport

MODEL_MERGE_CASES = [
    pytest.param(
        {
            "model": "old-model",
            "model_provider": "openai",
            "base_url": "https://old.example/v1",
            "ssl_verify": False,
        },
        {"max_completion_tokens": 128},
        {
            "model": "old-model",
            "model_provider": "openai",
            "base_url": "https://old.example/v1",
            "ssl_verify": False,
            "max_completion_tokens": 128,
        },
        id="ordinary-fields-preserve-lower-values",
    ),
    pytest.param(
        {
            "model": "old-model",
            "model_provider": "openai",
            "base_url": "https://old.example/v1",
        },
        {"inference_provider": "shared"},
        {
            "model": "old-model",
            "model_provider": "openai",
            "inference_provider": "shared",
        },
        id="inference-provider-supersedes-base-url",
    ),
    pytest.param(
        {
            "model": "old-model",
            "model_provider": "openai",
            "inference_provider": "shared",
        },
        {"base_url": "https://new.example/v1"},
        {
            "model": "old-model",
            "model_provider": "openai",
            "base_url": "https://new.example/v1",
        },
        id="base-url-supersedes-inference-provider",
    ),
    pytest.param(
        {
            "model": "old-model",
            "model_provider": "openai",
            "inference_provider": "old-provider",
        },
        {
            "base_url": "https://new.example/v1",
            "inference_provider": "new-provider",
        },
        {
            "model": "old-model",
            "model_provider": "openai",
            "base_url": "https://new.example/v1",
        },
        id="base-url-wins-when-one-layer-specifies-both-endpoints",
    ),
    pytest.param(
        {
            "model": "old-model",
            "model_provider": "openai",
            "inference_provider": "shared",
        },
        {"base_url": None},
        {
            "model": "old-model",
            "model_provider": "openai",
            "base_url": None,
            "inference_provider": "shared",
        },
        id="explicit-null-remains-explicit",
    ),
    pytest.param(
        {
            "model": "old-model",
            "model_provider": "openai",
            "timeout": 30,
        },
        {"temperature": 0.2},
        {
            "model": "old-model",
            "model_provider": "openai",
            "timeout": 30,
            "temperature": 0.2,
        },
        id="extra-fields-layer-like-declared-fields",
    ),
]


@pytest.mark.parametrize(("a", "b", "expected"), MODEL_MERGE_CASES)
def test_model_config_a_plus_b_equals_expected(a, b, expected):
    a_before = deepcopy(a)
    b_before = deepcopy(b)
    left = ChatModelConfig.model_validate(a)

    result = left.model_merge(b)

    assert result.model_dump(mode="python", exclude_unset=True) == expected
    assert a == a_before
    assert b == b_before


@pytest.mark.parametrize(
    ("b", "expected_updates"),
    [
        pytest.param(
            {"ssl_verify": False},
            {"ssl_verify": False},
            id="declared-field",
        ),
        pytest.param(
            {"cache_dir": "/tmp/cache"},
            {"cache_dir": "/tmp/cache"},
            id="extra-field",
        ),
        pytest.param(
            {"model": "ollama:nomic-embed-text:latest"},
            {
                "model": "nomic-embed-text:latest",
                "model_provider": "ollama",
            },
            id="model-prefix-infers-new-provider",
        ),
    ],
)
def test_embedding_model_accepts_partial_b_without_losing_a(
    b, expected_updates
):
    a = {
        "model": "text-embedding-3-large",
        "model_provider": "openai",
    }

    result = EmbModelConfig.model_validate(a).model_merge(b)

    expected = {**a, **expected_updates}
    assert result.model_dump(mode="python", exclude_unset=True) == expected


URSA_MERGE_CASES = [
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
        id="scalars-replace-and-mappings-recurse",
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
        {},
        {"inference_providers": {"openai": {"ssl_verify": False}}},
        {
            "inference_providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": {"env": "OPENAI_API_KEY"},
                    "ssl_verify": False,
                }
            }
        },
        id="partial-default-provider-update-preserves-default-endpoint",
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
        {"inference_providers": {"shared": {"api_key": {"keyring": "shared"}}}},
        {
            "inference_providers": {
                "shared": {
                    "base_url": "https://models.example/v1",
                    "api_key": {"keyring": "shared"},
                }
            }
        },
        id="provider-secret-source-replaces-atomically",
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
                    "transport": "stdio",
                    "command": "new-tool-server",
                }
            }
        },
        {
            "mcp_servers": {
                "tools": {
                    "transport": "stdio",
                    "command": "new-tool-server",
                },
                "unchanged": {
                    "transport": "stdio",
                    "command": "unchanged-server",
                },
            }
        },
        id="mcp-catalog-preserves-names-and-replaces-entries",
    ),
    pytest.param(
        {
            "emb_model": {
                "model": "old-embedding",
                "model_provider": "openai",
                "base_url": "https://old.example/v1",
            }
        },
        {"emb_model": {"inference_provider": "openai"}},
        {
            "emb_model": {
                "model": "old-embedding",
                "model_provider": "openai",
                "inference_provider": "openai",
            }
        },
        id="nested-model-uses-model-specific-endpoint-precedence",
    ),
    pytest.param(
        {
            "emb_model": {
                "model": "old-embedding",
                "model_provider": "openai",
            }
        },
        {"emb_model": None},
        {"emb_model": None},
        id="null-replaces-optional-model",
    ),
    pytest.param(
        {"rag_tools": ["first", "second"]},
        {"rag_tools": ["replacement"]},
        {"rag_tools": ["replacement"]},
        id="lists-replace-rather-than-concatenate",
    ),
]


def _dump(config: UrsaConfig):
    return config.model_dump(mode="python")


@pytest.mark.parametrize(("a", "b", "expected"), URSA_MERGE_CASES)
def test_ursa_config_a_plus_b_equals_expected(a, b, expected):
    a_before = deepcopy(a)
    b_before = deepcopy(b)

    together = UrsaConfig().model_merge(a, b)
    sequential = UrsaConfig().model_merge(a).model_merge(b)
    expected_config = UrsaConfig.model_validate(expected)

    assert _dump(together) == _dump(expected_config)
    assert _dump(sequential) == _dump(expected_config)
    assert a == a_before
    assert b == b_before


def test_higher_layer_model_prefix_replaces_lower_provider():
    result = UrsaConfig().model_merge(
        {
            "emb_model": {
                "model": "text-embedding-3-large",
                "model_provider": "openai",
            }
        },
        {"emb_model": {"model": "ollama:nomic-embed-text:latest"}},
    )

    assert result.emb_model is not None
    assert result.emb_model.model == "nomic-embed-text:latest"
    assert result.emb_model.model_provider == "ollama"


def test_mcp_transport_switch_does_not_resurrect_old_fields():
    result = UrsaConfig().model_merge(
        {
            "mcp_servers": {
                "tools": {
                    "transport": "stdio",
                    "command": "old-command",
                    "args": ["old-argument"],
                }
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
                    "transport": "stdio",
                    "command": "new-command",
                }
            }
        },
    )

    server = result.mcp_servers["tools"]
    assert server.command == "new-command"
    assert server.args == []


def test_typed_mcp_server_can_replace_another_transport():
    replacement = UrsaConfig.model_validate({
        "mcp_servers": {
            "tools": {
                "transport": "sse",
                "url": "https://tools.example/sse",
            }
        }
    }).mcp_servers["tools"]
    result = UrsaConfig().model_merge(
        {
            "mcp_servers": {
                "tools": {
                    "transport": "stdio",
                    "command": "old-command",
                }
            }
        },
        {"mcp_servers": {"tools": replacement}},
    )

    assert transport(result.mcp_servers["tools"]) == "sse"
    assert result.mcp_servers["tools"].url == "https://tools.example/sse"


def test_incomplete_optional_model_still_fails_after_all_layers():
    with pytest.raises(ValueError, match="Field required"):
        UrsaConfig().model_merge({"emb_model": {"model_provider": "openai"}})


def test_merge_result_does_not_alias_mutable_layer_values():
    layer = {"agent_config": {"research": {"tags": ["original"]}}}

    result = UrsaConfig().model_merge(layer)
    result.agent_config["research"]["tags"].append("result-only")

    assert layer == {"agent_config": {"research": {"tags": ["original"]}}}


def test_merge_preserves_temporary_workspace_ownership():
    base = UrsaConfig(workspace="tmp").resolve()

    result = base.model_merge({"group": "science"})

    assert result.workspace == base.workspace
    assert result._temp_workspace is base._temp_workspace


def test_merge_adopts_temporary_workspace_owned_by_higher_config():
    higher_priority = UrsaConfig(workspace="tmp").resolve()

    result = UrsaConfig().model_merge(higher_priority)

    assert result.workspace == higher_priority.workspace
    assert result._temp_workspace is higher_priority._temp_workspace


def test_partial_default_provider_update_is_inherited_by_default_model():
    merged = UrsaConfig().model_merge({
        "inference_providers": {"openai": {"ssl_verify": False}}
    })

    resolved = merged.resolve()

    assert resolved.llm_model.base_url == "https://api.openai.com/v1"
    assert resolved.llm_model.api_key is not None
    assert resolved.llm_model.ssl_verify is False


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
