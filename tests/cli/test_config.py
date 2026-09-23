from pathlib import Path

import pytest
from pydantic import ValidationError

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
    "cls",
    [
        config_mod.ModelConfig,
        config_mod.ChatModelConfig,
        config_mod.EmbModelConfig,
    ],
)
def test_model_config_model_parsing(cls):
    cfg = cls(model="bar:gpt-5.4")
    assert cfg.model == "gpt-5.4"
    assert cfg.model_provider == "bar"


def test_model_merge_keeps_provider_defaults_resolvable():
    config = config_mod.UrsaConfig().model_merge({
        "llm_model": {"model": "openai:gpt-5.4"}
    })

    resolved = config.resolve()

    assert resolved.llm_model.inference_provider == "openai"
    assert resolved.llm_model.base_url == "https://api.openai.com/v1"
    assert resolved.llm_model.api_key.env == "OPENAI_API_KEY"


def test_ursa_config_merge_preserves_explicit_fields():
    merged = config_mod.UrsaConfig().model_merge({"group": "science"})

    assert merged.model_fields_set == {"group"}


def test_ursa_config_merge_can_be_reused_as_sparse_layer():
    sparse = config_mod.UrsaConfig().model_merge({"group": "science"})
    merged = config_mod.UrsaConfig(
        workspace=Path("/tmp/custom-workspace")
    ).model_merge(sparse)

    assert merged.group == "science"
    assert merged.workspace == Path("/tmp/custom-workspace")


@pytest.mark.parametrize(
    "override",
    [{"emb_model": None}, config_mod.UrsaConfig(emb_model=None)],
    ids=["mapping", "config"],
)
def test_ursa_config_merge_can_disable_embedding_model(override):
    original = config_mod.UrsaConfig(
        emb_model={"model": "openai:original-embedding", "dimensions": 128}
    )

    merged = original.model_merge(override)

    assert merged.emb_model is None
    assert "emb_model" in merged.model_fields_set
    assert merged.model_dump(exclude_unset=True)["emb_model"] is None
    assert original.emb_model.model == "original-embedding"
    assert original.emb_model.dimensions == 128


def test_ursa_config_merge_can_reenable_embedding_after_explicit_null():
    original = config_mod.UrsaConfig(
        emb_model={"model": "openai:original-embedding", "dimensions": 128}
    )

    merged = original.model_merge(
        {"emb_model": None},
        {"emb_model": {"model": "openai:replacement-embedding"}},
    )

    assert merged.emb_model.model == "replacement-embedding"
    assert "dimensions" not in merged.emb_model.model_extra


def test_ursa_config_merge_rejects_null_for_required_llm_model():
    with pytest.raises(ValidationError) as exc_info:
        config_mod.UrsaConfig().model_merge({"llm_model": None})

    assert exc_info.value.errors()[0]["loc"] == ("llm_model",)
