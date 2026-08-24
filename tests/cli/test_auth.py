import pytest

from ursa.cli import build_parser, main
from ursa.cli.auth import (
    KEYRING_SERVICE,
    config_keyring_usernames,
    configured_secret_lines,
    list_credentials,
    login,
)


@pytest.fixture
def fake_keyring(monkeypatch):
    values = {}

    def get_password(service, username):
        return values.get((service, username))

    def set_password(service, username, password):
        values[(service, username)] = password

    monkeypatch.setattr("ursa.cli.auth.keyring.get_password", get_password)
    monkeypatch.setattr("ursa.cli.auth.keyring.set_password", set_password)
    return values


def test_login_stores_secret(monkeypatch, fake_keyring, capsys):
    monkeypatch.setattr("ursa.cli.auth.getpass.getpass", lambda prompt: "token")

    login("openai")

    assert fake_keyring[(KEYRING_SERVICE, "openai")] == "token"
    assert capsys.readouterr().out == "Stored credential: openai\n"


def test_login_rejects_empty_secret(monkeypatch, fake_keyring):
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


def test_login_rejects_missing_environment_secret(monkeypatch, fake_keyring):
    monkeypatch.delenv("MISSING_PROVIDER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="is not set or is empty"):
        login("openai", from_env="MISSING_PROVIDER_TOKEN")


def test_config_login_finds_provider_model_and_mcp_keyring_names(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  hosted:
    api_key:
      keyring: true
llm_model:
  model: anthropic:model
  api_key:
    keyring: direct-model
mcp_servers:
  tools:
    transport: streamable-http
    url: https://example.test/mcp
    headers:
      Authorization:
        keyring: true
        template: "Bearer %s"
      X-API-Key:
        keyring: shared-mcp
"""
    )

    assert config_keyring_usernames(path) == [
        "direct-model",
        "hosted",
        "shared-mcp",
        "tools",
    ]


def test_config_login_prompts_for_each_discovered_secret(
    tmp_path, monkeypatch, fake_keyring
):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  hosted:
    api_key:
      keyring: true
llm_model:
  model: openai:model
  inference_provider: hosted
"""
    )
    prompts = []

    def prompt(message):
        prompts.append(message)
        return "token"

    monkeypatch.setattr("ursa.cli.auth.getpass.getpass", prompt)

    login(config=path)

    assert prompts == ["Secret for hosted: "]
    assert fake_keyring[(KEYRING_SERVICE, "hosted")] == "token"


def test_config_login_supports_environment_configs(tmp_path, fake_keyring):
    path = tmp_path / "team.yaml"
    path.write_text(
        """
name: reviewers
inference_providers:
  anthropic:
    api_key:
      keyring: true
members:
  - name: reviewer
    model:
      model: openai:gpt-test
      api_key:
        keyring: model-override
"""
    )

    assert config_keyring_usernames(path) == [
        "anthropic",
        "model-override",
    ]


def test_from_env_rejects_config_with_multiple_secrets(
    tmp_path, monkeypatch, fake_keyring
):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  first:
    api_key:
      keyring: true
  second:
    api_key:
      keyring: true
"""
    )
    monkeypatch.setenv("SHARED_TOKEN", "token")

    with pytest.raises(ValueError, match="exactly one"):
        login(config=path, from_env="SHARED_TOKEN")


def test_auth_main_dispatches_login(monkeypatch):
    called = []
    monkeypatch.setattr(
        "ursa.cli.auth.login",
        lambda username, config, from_env: called.append((
            username,
            config,
            from_env,
        )),
    )

    main(["auth", "login", "openai", "--from-env", "OPENAI_API_KEY"])

    assert called == [("openai", None, "OPENAI_API_KEY")]


def test_auth_main_dispatches_list(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    called = []
    monkeypatch.setattr(
        "ursa.cli.auth.list_credentials",
        lambda config, show_secrets: called.append((config, show_secrets)),
    )

    main(["auth", "list", "--config", str(config)])

    assert called == [(config, False)]


def test_auth_list_reports_merged_configured_secrets(
    tmp_path, monkeypatch, fake_keyring, capsys
):
    system = tmp_path / "system.yaml"
    user = tmp_path / "user.yaml"
    explicit = tmp_path / "explicit.yaml"
    system.write_text(
        """
inference_providers:
  openai:
    api_key:
      keyring: true
mcp_servers:
  tools:
    transport: streamable-http
    url: https://example.test/mcp
    headers:
      Authorization:
        keyring: mcp-account
        template: "Bearer %s"
"""
    )
    user.write_text(
        """
inference_providers:
  anthropic:
    api_key:
      env: ANTHROPIC_API_KEY
"""
    )
    explicit.write_text(
        """
emb_model:
  model: openai:embedding
  api_key:
    env: SIGNING_KEY
llm_model:
  model: openai:test
  api_key:
    keyring: model-account
"""
    )
    monkeypatch.setattr(
        "ursa.cli.auth.config_search_paths",
        lambda namespace: [system, user, explicit],
    )
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
    system = tmp_path / "system.yaml"
    explicit = tmp_path / "explicit.yaml"
    system.write_text(
        """
llm_model:
  model: openai:test
  api_key:
    env: SYSTEM_TOKEN
"""
    )
    explicit.write_text(
        """
emb_model:
  model: openai:embedding
  api_key:
    env: FILE_TOKEN
"""
    )
    monkeypatch.setattr("ursa.cli.config.system_config_paths", lambda: [system])
    monkeypatch.setattr("ursa.cli.config.user_config_paths", lambda: [])

    lines = configured_secret_lines(explicit)["Other"]

    assert lines == [
        "  emb_model: env missing",
        "  llm_model: env missing",
    ]


def test_auth_list_discovers_multiple_mcp_secrets(
    tmp_path, monkeypatch, fake_keyring
):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
mcp_servers:
  tools:
    transport: streamable-http
    url: https://example.test/mcp
    headers:
      Authorization:
        env: TOOLS_BEARER
      X-API-Key:
        keyring: true
"""
    )
    monkeypatch.setattr(
        "ursa.cli.auth.config_search_paths", lambda namespace: [config]
    )

    lines = configured_secret_lines(config)["MCP Servers:"]

    assert lines == [
        "  tools.Authorization: env missing",
        "  tools.X-API-Key: keyring missing",
    ]


def test_auth_list_skips_empty_sections(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm_model:
  model: openai:test
  api_key:
    env: SIGNING_KEY
"""
    )
    monkeypatch.setattr(
        "ursa.cli.auth.config_search_paths", lambda namespace: [config]
    )

    list_credentials(config)

    assert capsys.readouterr().out == "Other\n  llm_model: env missing\n"


def test_auth_list_uses_secret_resolution(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm_model:
  model: openai:test
  api_key:
    env: MODEL_TOKEN
"""
    )
    monkeypatch.setattr(
        "ursa.cli.auth.config_search_paths", lambda namespace: [config]
    )
    calls = []

    def resolve(reference, default_username=None):
        calls.append((reference.env, default_username))
        return None

    monkeypatch.setattr("ursa.cli.auth.SecretReference.resolve", resolve)

    list_credentials(config)

    assert calls == [("MODEL_TOKEN", "openai")]
    assert "llm_model: env missing" in capsys.readouterr().out


def test_auth_list_can_show_secret_values(
    tmp_path, monkeypatch, fake_keyring, capsys
):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
inference_providers:
  hosted:
    api_key:
      keyring: true
"""
    )
    monkeypatch.setattr(
        "ursa.cli.auth.config_search_paths", lambda namespace: [config]
    )
    fake_keyring[(KEYRING_SERVICE, "hosted")] = "visible-token"

    list_credentials(config, show_secrets=True)

    assert capsys.readouterr().out == (
        "Inference Providers\n  hosted: keyring ok = visible-token\n"
    )


def test_auth_list_prints_nothing_when_no_secrets_exist(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "config.yaml"
    config.write_text("workspace: .\n")
    monkeypatch.setattr(
        "ursa.cli.auth.config_search_paths", lambda namespace: [config]
    )

    list_credentials(config)

    assert capsys.readouterr().out == ""


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
