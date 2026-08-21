from pathlib import Path

import pytest
from jsonargparse import Namespace
from pydantic import SecretStr

from ursa.cli import build_parser
from ursa.cli.config import (
    APIKeyConfig,
    ChatModelConfig,
    UrsaConfig,
    merge_ursa_config,
)
from ursa.cli.print_config import parse_print_config_spec
from ursa.util import crossplatform
from ursa.util.secrets import SecretReference, SecretTemplate


def test_config_precedence_all_six_layers(tmp_path, monkeypatch):
    system = tmp_path / "system.yaml"
    user = tmp_path / "user.yaml"
    explicit = tmp_path / "explicit.yaml"
    system.write_text("llm_model:\n  model: system\n")
    user.write_text("llm_model:\n  model: user\n")
    explicit.write_text("llm_model:\n  model: explicit\n")
    monkeypatch.setattr(
        "ursa.cli.config.config_search_paths",
        lambda cfg, level="final": [system, user, explicit],
    )

    assert UrsaConfig().llm_model.model == "openai:gpt-5.4"
    namespace = Namespace(config=explicit)
    config_flag = {"config": explicit}
    assert (
        merge_ursa_config(
            namespace, overrides={}, cli_overrides=config_flag
        ).llm_model.model
        == "explicit"
    )
    assert (
        merge_ursa_config(
            namespace,
            overrides={"llm_model": {"model": "env"}},
            cli_overrides=config_flag,
        ).llm_model.model
        == "explicit"
    )
    assert (
        merge_ursa_config(
            namespace,
            overrides={"llm_model": {"model": "env"}},
            cli_overrides={
                "config": explicit,
                "llm_model": {"model": "cli"},
            },
        ).llm_model.model
        == "cli"
    )


def test_api_key_direct_value_is_secret():
    config = ChatModelConfig(model="provider:model", api_key="secret")

    assert isinstance(config.api_key, SecretStr)
    assert config.kwargs["api_key"] == "secret"
    assert "secret" not in repr(config)


def test_secret_reference_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one source"):
        SecretReference(env="TOKEN", keyring="account")


def test_secret_reference_get_secret_value(monkeypatch):
    monkeypatch.setenv("MODEL_TOKEN", "secret")

    assert SecretReference(env="MODEL_TOKEN").get_secret_value() == "secret"
    assert (
        SecretTemplate(
            env="MODEL_TOKEN", template="Bearer %s"
        ).get_secret_value()
        == "Bearer secret"
    )


@pytest.mark.parametrize("template", ["Bearer", "%s:%s"])
def test_secret_template_requires_one_placeholder(template):
    with pytest.raises(ValueError, match="exactly one"):
        SecretTemplate(env="MODEL_TOKEN", template=template)


def test_api_key_env_reference_resolves(monkeypatch):
    monkeypatch.setenv("MODEL_TOKEN", "secret")
    config = ChatModelConfig(
        model="provider:model", api_key={"env": "MODEL_TOKEN"}
    )

    resolved = config.resolve_api_key("provider")

    assert isinstance(resolved.api_key, SecretStr)
    assert resolved.kwargs["api_key"] == "secret"


@pytest.mark.parametrize(
    ("setting", "username"), [(True, "acme"), ("account", "account")]
)
def test_api_key_keyring_reference(monkeypatch, setting, username):
    calls = []
    monkeypatch.setattr(
        "keyring.get_password",
        lambda system, user: calls.append((system, user)) or "secret",
    )
    config = ChatModelConfig(
        model="provider:model", api_key={"keyring": setting}
    )

    resolved = config.resolve_api_key("acme")

    assert resolved.kwargs["api_key"] == "secret"
    assert calls == [("ursa", username)]


def test_legacy_api_key_env_is_migrated_with_warning(monkeypatch):
    monkeypatch.setenv("OLD_TOKEN", "secret")
    with pytest.warns(DeprecationWarning, match="api_key_env is deprecated"):
        config = ChatModelConfig(
            model="provider:model", api_key_env="OLD_TOKEN"
        )

    resolved = config.resolve_api_key("provider")

    assert resolved.kwargs["api_key"] == "secret"
    assert "api_key_env" not in config.model_dump()


def test_default_openai_provider_is_explicit():
    config = UrsaConfig()

    assert config.llm_model.inference_provider == "openai"
    assert config.inference_providers["openai"].api_key == APIKeyConfig(
        env="OPENAI_API_KEY"
    )


def test_print_config_level_gets_default_stage():
    assert parse_print_config_spec("user") == ("user", "resolved")
    assert parse_print_config_spec("system") == ("system", "resolved")
    assert parse_print_config_spec("file+") == ("file+", "resolved")


def test_invalid_print_config_shows_concise_error_without_function_repr(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--print-config=invalid"])

    error = capsys.readouterr().err
    assert "URSA: The Universal Research and Scientific Agent" not in error
    assert "usage: ursa" in error
    assert "_validate_print_config_spec" not in error
    assert "Unknown print-config level or stage 'invalid'" in error


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("linux", Path("/etc/ursa/config.yaml")),
        ("darwin", Path("/Library/Application Support/ursa/config.yaml")),
        ("win32", Path("C:/ProgramData/ursa/config.yaml")),
    ],
)
def test_system_config_path_is_platform_specific(
    monkeypatch, platform, expected
):
    monkeypatch.setattr(crossplatform.sys, "platform", platform)
    monkeypatch.delenv("PROGRAMDATA", raising=False)

    assert crossplatform.system_config_path() == expected


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_portable_and_xdg_user_paths_override_platform_default(
    tmp_path, monkeypatch, platform
):
    home = tmp_path / "home"
    xdg_home = tmp_path / "xdg"
    monkeypatch.setattr(crossplatform.sys, "platform", platform)
    monkeypatch.setattr(crossplatform.Path, "home", lambda: home)
    monkeypatch.setenv("APPDATA", str(home / "AppData/Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))

    paths = crossplatform.user_config_paths()

    assert paths[-2:] == [
        home / ".config/ursa/config.yaml",
        xdg_home / "ursa/config.yaml",
    ]
    if platform in {"darwin", "win32"}:
        assert len(paths) == 3


def test_user_config_precedence_is_platform_then_dot_config_then_xdg(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    xdg_home = tmp_path / "xdg"
    monkeypatch.setattr(crossplatform.sys, "platform", "darwin")
    monkeypatch.setattr(crossplatform.Path, "home", lambda: home)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    monkeypatch.delenv("XDG_CONFIG_DIRS", raising=False)

    native, portable, xdg = crossplatform.user_config_paths()
    for path, group in (
        (native, "native"),
        (portable, "portable"),
        (xdg, "xdg"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"group: {group}\n", encoding="utf-8")

    config = merge_ursa_config(Namespace(), overrides={})

    assert config.group == "xdg"
