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
