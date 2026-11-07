import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ursa.cli import build_parser, main
from ursa.cli.auth import (
    config_keyring_usernames,
    configured_secret_lines,
    list_credentials,
    login,
)
from ursa.cli.config import UrsaConfig, load_config_file
from ursa.util.secrets import KEYRING_SERVICE, keyring_usernames


def _config_file(tmp_path, contents, name="config.yaml"):
    if not isinstance(contents, str):
        contents = json.dumps(contents)
    path = tmp_path / name
    path.write_text(contents)
    return path


def _secret(reference, model=None):
    return {"api_key": reference, **({"model": model} if model else {})}


def _secret_config(
    tmp_path, reference, section="llm_model", name=None, file_name="config.yaml"
):
    entry = _secret(reference, "openai:test" if name is None else None)
    value = entry if name is None else {name: entry}
    return _config_file(tmp_path, {section: value}, file_name)


def _mcp_data(headers):
    server = {
        "transport": "streamable-http",
        "url": "https://example.test/mcp",
        "headers": headers,
    }
    return {"mcp_servers": {"tools": server}}


def _use_config_layers(monkeypatch, *paths):
    monkeypatch.setattr(
        "ursa.cli.auth.config_layers",
        lambda namespace, level: [load_config_file(path) for path in paths],
    )


def _list_output(monkeypatch, capsys, config, **kwargs):
    _use_config_layers(monkeypatch, config)
    list_credentials(config, **kwargs)
    return capsys.readouterr().out


@pytest.fixture
def fake_keyring(monkeypatch):
    values = {}
    backend = SimpleNamespace(
        get_password=lambda *key: values.get(key),
        set_password=lambda *entry: values.__setitem__(entry[:2], entry[2]),
    )
    monkeypatch.setattr("ursa.cli.auth.get_keyring", lambda: backend)
    monkeypatch.setattr("ursa.util.secrets.get_keyring", lambda: backend)
    return values


def test_login_stores_secret(monkeypatch, fake_keyring, capsys):
    monkeypatch.setattr("ursa.cli.auth.getpass.getpass", lambda prompt: "token")

    login("openai")

    assert fake_keyring[(KEYRING_SERVICE, "openai")] == "token"
    assert capsys.readouterr().out == "Stored credential: openai\n"


def test_login_rejects_empty_secret(monkeypatch):
    monkeypatch.setattr("ursa.cli.auth.getpass.getpass", lambda prompt: "")

    with pytest.raises(ValueError, match="cannot be empty"):
        login("openai")


def test_login_reads_secret_from_environment(monkeypatch, fake_keyring):
    monkeypatch.setenv("PROVIDER_TOKEN", "from-environment")
    monkeypatch.setattr(
        "ursa.cli.auth.getpass.getpass",
        lambda prompt: pytest.fail("password prompt should not be used"),
    )

    login("openai", from_env="PROVIDER_TOKEN")

    assert fake_keyring[(KEYRING_SERVICE, "openai")] == "from-environment"


def test_login_rejects_missing_environment_secret(monkeypatch):
    monkeypatch.delenv("MISSING_PROVIDER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="is not set or is empty"):
        login("openai", from_env="MISSING_PROVIDER_TOKEN")


def test_config_login_finds_provider_model_and_mcp_keyring_names(tmp_path):
    path = _config_file(
        tmp_path,
        {
            "inference_providers": {"hosted": _secret({"keyring": True})},
            "llm_model": _secret(
                {"keyring": "direct-model"}, "anthropic:model"
            ),
            **_mcp_data({
                "Authorization": {
                    "keyring": True,
                    "template": "Bearer %s",
                },
                "X-API-Key": {"keyring": "shared-mcp"},
            }),
        },
    )

    assert config_keyring_usernames(path) == [
        "direct-model",
        "hosted",
        "shared-mcp",
        "tools",
    ]


def test_keyring_discovery_accepts_typed_config() -> None:
    config = UrsaConfig.model_validate({
        "llm_model": _secret({"keyring": "model-account"}, "openai:test"),
        **_mcp_data({
            "Authorization": {
                "keyring": True,
                "template": "Bearer %s",
            }
        }),
    })

    assert keyring_usernames(config) == ["model-account", "tools"]


def test_config_login_prompts_for_each_discovered_secret(
    tmp_path, monkeypatch, fake_keyring
):
    path = _secret_config(
        tmp_path, {"keyring": True}, "inference_providers", "hosted"
    )
    prompts = []

    def prompt(message):
        prompts.append(message)
        return "token"

    monkeypatch.setattr("ursa.cli.auth.getpass.getpass", prompt)

    login(config=path)

    assert prompts == ["Secret for hosted: "]
    assert fake_keyring[(KEYRING_SERVICE, "hosted")] == "token"


def test_config_login_supports_environment_configs(tmp_path):
    path = _config_file(
        tmp_path,
        {
            "name": "reviewers",
            "inference_providers": {"anthropic": _secret({"keyring": True})},
            "members": [
                {
                    "name": "reviewer",
                    "model": _secret(
                        {"keyring": "model-override"}, "openai:gpt-test"
                    ),
                }
            ],
        },
        "team.yaml",
    )

    assert config_keyring_usernames(path) == ["anthropic", "model-override"]


def test_from_env_rejects_config_with_multiple_secrets(tmp_path, monkeypatch):
    path = _config_file(
        tmp_path,
        {
            "inference_providers": {
                name: _secret({"keyring": True}) for name in ("first", "second")
            }
        },
    )
    monkeypatch.setenv("SHARED_TOKEN", "token")

    with pytest.raises(ValueError, match="exactly one"):
        login(config=path, from_env="SHARED_TOKEN")


def test_auth_main_dispatches_login(monkeypatch):
    called = Mock()
    monkeypatch.setattr("ursa.cli.auth.login", called)

    main(["auth", "login", "openai", "--from-env", "OPENAI_API_KEY"])

    called.assert_called_once_with(
        username="openai", config=None, from_env="OPENAI_API_KEY"
    )


def test_auth_main_dispatches_list(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    called = Mock()
    monkeypatch.setattr("ursa.cli.auth.list_credentials", called)

    main(["auth", "list", "--config", str(config)])

    called.assert_called_once_with(config=config, show_secrets=False)


def test_auth_list_reports_merged_configured_secrets(
    tmp_path, monkeypatch, fake_keyring, capsys
):
    system = _config_file(
        tmp_path,
        {
            "inference_providers": {"openai": _secret({"keyring": True})},
            **_mcp_data({
                "Authorization": {
                    "keyring": "mcp-account",
                    "template": "Bearer %s",
                }
            }),
        },
        "system.yaml",
    )
    user = _secret_config(
        tmp_path,
        {"env": "ANTHROPIC_API_KEY"},
        "inference_providers",
        "anthropic",
        "user.yaml",
    )
    explicit = _config_file(
        tmp_path,
        {
            "emb_model": _secret({"env": "SIGNING_KEY"}, "openai:embedding"),
            "llm_model": _secret({"keyring": "model-account"}, "openai:test"),
        },
        "explicit.yaml",
    )
    _use_config_layers(monkeypatch, system, user, explicit)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    fake_keyring[(KEYRING_SERVICE, "mcp-account")] = "present"

    list_credentials(explicit)

    assert capsys.readouterr().out == (
        "Inference Providers\n"
        "  anthropic: env ok\n"
        "  openai: keyring missing\n"
        "\nMCP Servers:\n"
        "  tools (mcp-account): keyring ok\n"
        "\nOther\n"
        "  emb_model: env missing\n"
        "  llm_model (model-account): keyring missing\n"
    )


def test_auth_list_uses_shared_config_search_primitive(tmp_path, monkeypatch):
    system = _secret_config(
        tmp_path,
        {"env": "SYSTEM_TOKEN"},
        file_name="system.yaml",
    )
    explicit = _secret_config(
        tmp_path,
        {"env": "FILE_TOKEN"},
        "emb_model",
        file_name="explicit.yaml",
    )
    monkeypatch.setattr("ursa.cli.config.system_config_paths", lambda: [system])
    monkeypatch.setattr("ursa.cli.config.user_config_paths", lambda: [])

    lines = configured_secret_lines(explicit)["Other"]

    assert lines == ["  emb_model: env missing", "  llm_model: env missing"]


def test_auth_list_discovers_multiple_mcp_secrets(
    tmp_path, monkeypatch, fake_keyring
):
    config = _config_file(
        tmp_path,
        _mcp_data({
            "Authorization": {"env": "TOOLS_BEARER"},
            "X-API-Key": {"keyring": True},
        }),
    )
    _use_config_layers(monkeypatch, config)

    lines = configured_secret_lines(config)["MCP Servers:"]

    assert lines == [
        "  tools.Authorization: env missing",
        "  tools.X-API-Key: keyring missing",
    ]


def test_auth_list_uses_secret_resolution(tmp_path, monkeypatch, capsys):
    config = _secret_config(tmp_path, {"env": "MODEL_TOKEN"})
    calls = []

    def resolve(reference, default_username=None):
        calls.append((reference.env, default_username))
        return None

    monkeypatch.setattr("ursa.util.secrets.SecretReference.resolve", resolve)

    output = _list_output(monkeypatch, capsys, config)

    assert calls == [("MODEL_TOKEN", "openai")]
    assert "llm_model: env missing" in output


def test_auth_list_skips_empty_sections(tmp_path, monkeypatch, capsys):
    config = _secret_config(tmp_path, {"env": "SIGNING_KEY"})
    assert _list_output(monkeypatch, capsys, config) == (
        "Other\n  llm_model: env missing\n"
    )


def test_auth_list_can_show_secret_values(
    tmp_path, monkeypatch, fake_keyring, capsys
):
    config = _secret_config(
        tmp_path, {"keyring": True}, "inference_providers", "hosted"
    )
    fake_keyring[(KEYRING_SERVICE, "hosted")] = "visible-token"

    assert _list_output(monkeypatch, capsys, config, show_secrets=True) == (
        "Inference Providers\n  hosted: keyring ok = visible-token\n"
    )


def test_auth_list_prints_nothing_when_no_secrets_exist(
    tmp_path, monkeypatch, capsys
):
    config = _config_file(tmp_path, "workspace: .\n")
    assert _list_output(monkeypatch, capsys, config) == ""


def test_auth_parser_ignores_configuration_environment_variables(monkeypatch):
    monkeypatch.setenv("URSA_AUTH__LOGIN__USERNAME", "from-environment")
    monkeypatch.setenv("URSA_AUTH__LOGIN__FROM_ENV", "SECRET_VARIABLE")
    monkeypatch.setenv("URSA_AUTH__LIST__CONFIG", "/ignored/config.yaml")

    with pytest.raises(SystemExit):
        build_parser().parse_args(["auth", "login"])

    parsed = build_parser().parse_args(["auth", "login", "explicit"])
    assert parsed.auth.login.username == "explicit"
    assert parsed.auth.login.from_env is None

    parsed = build_parser().parse_args(["auth", "list"])
    assert parsed.auth.list.config is None
    assert parsed.auth.list.show_secrets is False
