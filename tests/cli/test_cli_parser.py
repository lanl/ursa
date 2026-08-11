from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from openai import OpenAIError

from ursa.cli import (
    _xdg_config_search_paths,
    build_parser,
    main,
    resolve_config,
)
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    ModelConfig,
    UrsaConfig,
    resolve_ursa_config,
)


def _stub_mcp_server(monkeypatch):
    mcp = MagicMock()
    hitl = MagicMock()
    hitl.as_mcp_server.return_value = mcp
    hitl_class = MagicMock(return_value=hitl)
    monkeypatch.setattr("ursa.cli.hitl.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    return hitl, mcp


def _stub_cli_repl(monkeypatch):
    monkeypatch.setattr("ursa.cli.hitl.HITL", MagicMock())
    monkeypatch.setattr("ursa.cli.hitl.UrsaRepl", MagicMock())
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)


def test_cli_warns_about_legacy_unnamed_checkpoint(
    monkeypatch, tmp_path, capsys
):
    _stub_cli_repl(monkeypatch)
    checkpoint = tmp_path / "db" / "checkpointer.db"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    main(["--workspace", str(tmp_path)])

    warning = capsys.readouterr().err
    assert "URSA no longer restarts unnamed CLI sessions" in warning
    assert "only persisted when --name is used" in warning
    assert (
        "ursa import-agent db/checkpointer.db --name <new agent name>"
        in warning
    )
    assert "--name <new agent name>" in warning


def test_cli_does_not_warn_about_legacy_checkpoint_with_name(
    monkeypatch, tmp_path, capsys
):
    _stub_cli_repl(monkeypatch)
    checkpoint = tmp_path / "db" / "checkpointer.db"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    main(["--workspace", str(tmp_path), "--name", "continued-agent"])

    assert capsys.readouterr().err == ""


def test_cli_does_not_warn_without_legacy_checkpoint(
    monkeypatch, tmp_path, capsys
):
    _stub_cli_repl(monkeypatch)

    main(["--workspace", str(tmp_path)])

    assert capsys.readouterr().err == ""


def test_cli_reports_model_initialization_error_without_traceback(
    monkeypatch, capsys
):
    error = OpenAIError(
        "The api_key client option must be set by setting the "
        "OPENAI_API_KEY environment variable"
    )
    monkeypatch.setattr("ursa.cli.hitl.HITL", MagicMock(side_effect=error))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    with pytest.raises(SystemExit, match="2"):
        main([])

    stderr = capsys.readouterr().err
    assert stderr.startswith("Error: unable to initialize the language model.")
    assert "OPENAI_API_KEY" in stderr
    assert "Traceback" not in stderr


def test_mcp_server_passes_only_stdio_run_options(monkeypatch):
    hitl, mcp = _stub_mcp_server(monkeypatch)

    main(["mcp-server"])

    hitl.as_mcp_server.assert_called_once_with()
    mcp.run.assert_called_once_with(transport="stdio", log_level="INFO")


def test_mcp_server_config_flag_sets_hosted_llm(monkeypatch, tmp_path):
    """`ursa mcp-server --config FILE` must configure the hosted URSA instance.

    Regression: previously the mcp-server subparser had no --config option, so
    `ursa mcp-server --config foo.yaml` failed with "Unrecognized arguments".
    The subcommand-local --config should feed the LLM model/endpoint (and other
    UrsaConfig fields) into the HITL instance backing the MCP server.
    """
    hitl_class = MagicMock()
    hitl_class.return_value = MagicMock()
    hitl_class.return_value.as_mcp_server.return_value = MagicMock()
    monkeypatch.setattr("ursa.cli.hitl.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    config_file = tmp_path / "mcp.yaml"
    config_file.write_text(
        yaml.safe_dump({
            "llm_model": {
                "model": "ollama:llama3.1",
                "base_url": "http://localhost:11434",
            }
        })
    )

    main(["mcp-server", "--config", str(config_file)])

    # HITL should be constructed with the config loaded from --config.
    (ursa_config,), _ = hitl_class.call_args
    assert ursa_config.llm_model.model == "ollama:llama3.1"
    assert ursa_config.llm_model.base_url == "http://localhost:11434"


def test_mcp_server_config_flag_parses_alongside_transport(
    monkeypatch, tmp_path
):
    """--config coexists with MCP transport options on the subcommand."""
    hitl_class = MagicMock()
    hitl_class.return_value = MagicMock()
    mcp = MagicMock()
    hitl_class.return_value.as_mcp_server.return_value = mcp
    monkeypatch.setattr("ursa.cli.hitl.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    config_file = tmp_path / "mcp.yaml"
    config_file.write_text(
        yaml.safe_dump({"llm_model": {"model": "ollama:llama3.1"}})
    )

    main([
        "mcp-server",
        "--config",
        str(config_file),
        "--transport",
        "streamable-http",
        "--port",
        "9001",
    ])

    (ursa_config,), _ = hitl_class.call_args
    assert ursa_config.llm_model.model == "ollama:llama3.1"
    mcp.run.assert_called_once_with(
        transport="streamable-http",
        log_level="INFO",
        host="localhost",
        port=9001,
    )


def test_mcp_server_passes_http_options_when_running(monkeypatch):
    hitl, mcp = _stub_mcp_server(monkeypatch)

    main([
        "mcp-server",
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        "9001",
        "--log_level",
        "warning",
    ])

    hitl.as_mcp_server.assert_called_once_with()
    mcp.run.assert_called_once_with(
        transport="streamable-http",
        log_level="WARNING",
        host="127.0.0.1",
        port=9001,
    )


def test_cli_parses_typed_flags(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--workspace",
        str(tmp_path / "workspace"),
        "--llm_model.model",
        "openai:gpt-5-nano",
        "--llm_model.max_completion_tokens",
        "2048",
    ])

    config = UrsaConfig.from_namespace(args)
    assert config.workspace == tmp_path / "workspace"
    assert config.llm_model.model == "openai:gpt-5-nano"
    assert config.llm_model.max_completion_tokens == 2048


def test_print_config_flag_sets_bool_and_preserves_defaults(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-missing"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg-dirs-missing"))
    monkeypatch.chdir(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["--print-config"])

    assert args["print_config"] is True

    config = resolve_config(args)
    assert config.model_dump() == UrsaConfig().model_dump()


def test_resolve_config_preserves_cli_tmp_workspace_owner():
    parser = build_parser()
    args = parser.parse_args(["--workspace", "tmp"])

    config = resolve_config(args)

    assert config.workspace.exists()
    assert config._temp_workspace is not None
    assert config._temp_workspace.name == str(config.workspace)


def test_resolve_config_preserves_file_tmp_workspace_owner(tmp_path):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text("workspace: tmp\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg_path)])

    config = resolve_config(args)

    assert config.workspace.exists()
    assert config._temp_workspace is not None
    assert config._temp_workspace.name == str(config.workspace)


def test_cli_applies_chat_only_openai_defaults_to_llm_model():
    parser = build_parser()
    args = parser.parse_args([])

    config = resolve_config(args)

    assert isinstance(config.llm_model, ChatModelConfig)
    assert config.llm_model.kwargs["use_responses_api"] is True


def test_cli_does_not_apply_chat_only_openai_defaults_to_emb_model():
    parser = build_parser()
    args = parser.parse_args([
        "--emb_model.model",
        "openai:text-embedding-3-large",
    ])

    config = resolve_config(args)

    assert isinstance(config.emb_model, EmbModelConfig)
    assert not isinstance(config.emb_model, ChatModelConfig)
    assert "use_responses_api" not in config.emb_model.kwargs


def test_print_config_yaml_round_trip(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--workspace",
        str(tmp_path / "original"),
        "--llm_model.model",
        "openai:gpt-5-nano",
    ])

    original_config = resolve_config(args)
    yaml_text = yaml.safe_dump(original_config.model_dump())

    cfg_path = tmp_path / "round-trip.yml"
    cfg_path.write_text(yaml_text)

    parser = build_parser()
    loaded_args = parser.parse_args(["--config", str(cfg_path)])
    loaded_config = resolve_config(loaded_args)

    assert loaded_config.model_dump() == original_config.model_dump()


def test_config_env_cli_precedence(tmp_path, monkeypatch):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text(
        "\n".join([
            "workspace: config_workspace",
            "llm_model:",
            "  model: config-model",
        ])
    )

    env_workspace = tmp_path / "env-workspace"
    env_workspace.mkdir()
    monkeypatch.setenv("URSA_WORKSPACE", str(env_workspace))
    monkeypatch.setenv("URSA_LLM_MODEL__MODEL", "env-model")

    parser = build_parser()

    args_env = parser.parse_args(["--config", str(cfg_path)])
    config_env = resolve_config(args_env)
    assert config_env.workspace == env_workspace
    assert config_env.llm_model.model == "env-model"

    cli_workspace = tmp_path / "cli-workspace"
    cli_workspace.mkdir()
    args_cli = parser.parse_args([
        "--config",
        str(cfg_path),
        "--emb_model.model",
        "openai:text-embedding-3-large",
        "--workspace",
        str(cli_workspace),
        "--llm_model.model",
        "cli-model",
    ])
    config_cli = resolve_config(args_cli)
    assert config_cli.workspace == cli_workspace
    assert config_cli.llm_model.model == "cli-model"
    assert config_cli.emb_model.model == "openai:text-embedding-3-large"


def test_config_file_env_interpolation(tmp_path, monkeypatch):
    env_workspace = tmp_path / "env-workspace"
    env_workspace.mkdir()
    monkeypatch.setenv("URSA_CFG_WORKSPACE", str(env_workspace))
    monkeypatch.setenv("URSA_CFG_LLM_MODEL", "openai:gpt-env")
    monkeypatch.delenv("URSA_CFG_EMB_MODEL", raising=False)

    cfg_path = tmp_path / "ursa-env.yml"
    cfg_path.write_text(
        "\n".join([
            "workspace: ${URSA_CFG_WORKSPACE}",
            "llm_model:",
            "  model: ${URSA_CFG_LLM_MODEL}",
            "emb_model:",
            "  model: ${URSA_CFG_EMB_MODEL:openai:gpt-5}",
        ])
    )

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg_path)])
    config = resolve_config(args)

    assert config.workspace == env_workspace
    assert config.llm_model.model == "openai:gpt-env"
    assert config.emb_model.model == "openai:gpt-5"


def test_config_file_with_extra_keys(tmp_path):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-5-small",
            "  temperature: 0.4",
            "  seed: 123",
            "emb_model:",
            "  model: openai:text-embedding-3-large",
            "  cache_dir: /tmp/cache",
        ])
    )

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg_path)])
    config = resolve_config(args)

    assert config.llm_model.model == "openai:gpt-5-small"
    assert config.llm_model.model_extra["seed"] == 123
    assert config.emb_model.model_extra["cache_dir"] == "/tmp/cache"


def test_config_file_and_cli_are_merged(tmp_path):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text(
        "\n".join([
            "workspace: config_workspace",
            "llm_model:",
            "  model: openai:gpt-5-small",
            "  temperature: 0.4",
            "emb_model:",
            "  model: openai:text-embedding-3-large",
            "  cache_dir: /tmp/cache",
        ])
    )

    cli_workspace = tmp_path / "cli-workspace"
    parser = build_parser()
    args = parser.parse_args([
        "--config",
        str(cfg_path),
        "--emb_model.model",
        "openai:text-embedding-3-large",
        "--workspace",
        str(cli_workspace),
        "--llm_model.model",
        "openai:gpt-5-nano",
    ])

    config = resolve_config(args)

    assert config.workspace == cli_workspace
    assert config.llm_model.model == "openai:gpt-5-nano"
    assert config.llm_model.model_extra["temperature"] == 0.4
    assert config.emb_model.model == "openai:text-embedding-3-large"
    assert config.emb_model.model_extra["cache_dir"] == "/tmp/cache"


def test_xdg_config_search_paths_honor_env_overrides(tmp_path, monkeypatch):
    xdg_home = tmp_path / "xdg-home"
    xdg_dir_1 = tmp_path / "xdg-dir-1"
    xdg_dir_2 = tmp_path / "xdg-dir-2"

    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    monkeypatch.setenv("XDG_CONFIG_DIRS", f"{xdg_dir_1}:{xdg_dir_2}")

    assert _xdg_config_search_paths() == [
        xdg_dir_1 / "ursa" / "config.yaml",
        xdg_dir_2 / "ursa" / "config.yaml",
        xdg_home / "ursa" / "config.yaml",
    ]


def test_resolve_config_merges_xdg_local_file_and_cli_layers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.config.enforce_group_base_url_policy",
        lambda base_url, group: None,
    )
    xdg_dir = tmp_path / "xdg-home"
    xdg_dir_1 = tmp_path / "xdg-dir-1"
    project_dir = tmp_path / "project"
    explicit_cfg = tmp_path / "explicit.yaml"
    cli_workspace = tmp_path / "cli-workspace"
    cli_workspace.mkdir()

    xdg_dir_config = xdg_dir_1 / "ursa" / "config.yaml"
    xdg_dir_config.parent.mkdir(parents=True)
    xdg_dir_config.write_text(
        yaml.safe_dump({
            "workspace": "xdg-dir-workspace",
            "llm_model": {
                "model": "openai:gpt-5-mini",
                "max_completion_tokens": 1024,
            },
        })
    )

    xdg_config = xdg_dir / "ursa" / "config.yaml"
    xdg_config.parent.mkdir(parents=True)
    xdg_config.write_text(
        yaml.safe_dump({
            "workspace": "xdg-workspace",
            "group": "xdg-group",
            "llm_model": {
                "model": "openai:gpt-5-mini",
                "base_url": "https://xdg.example/v1",
            },
            "agent_config": {"chat": {"temperature": 0.1}},
        })
    )

    local_config = project_dir / ".ursa" / "config.yaml"
    local_config.parent.mkdir(parents=True)
    local_config.write_text(
        yaml.safe_dump({
            "group": "local-group",
            "llm_model": {
                "model": "openai:gpt-5-mini",
                "base_url": "https://local.example/v1",
            },
            "emb_model": {"model": "openai:text-embedding-3-small"},
            "agent_config": {"chat": {"top_p": 0.9}},
        })
    )

    explicit_cfg.write_text(
        yaml.safe_dump({
            "group": "file-group",
            "llm_model": {
                "model": "openai:gpt-5-mini",
                "api_key_env": "EXPLICIT_KEY",
            },
            "agent_config": {"chat": {"seed": 7}},
        })
    )

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(xdg_dir_1))

    parser = build_parser()
    args = parser.parse_args([
        "--config",
        str(explicit_cfg),
        "--workspace",
        str(cli_workspace),
        "--group",
        "cli-group",
        "--llm_model.model",
        "openai:gpt-5-nano",
    ])

    config = resolve_config(args)

    assert config.workspace == cli_workspace
    assert config.group == "cli-group"
    assert config.llm_model.model == "openai:gpt-5-nano"
    assert config.llm_model.base_url == "https://local.example/v1"
    assert config.llm_model.api_key_env == "EXPLICIT_KEY"
    assert config.llm_model.max_completion_tokens == 1024
    assert config.emb_model is not None
    assert config.emb_model.model == "openai:text-embedding-3-small"
    assert config.agent_config["chat"] == {
        "temperature": 0.1,
        "top_p": 0.9,
        "seed": 7,
    }


def test_resolve_config_uses_local_and_xdg_defaults_without_explicit_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.config.enforce_group_base_url_policy",
        lambda base_url, group: None,
    )
    xdg_config_dir = tmp_path / "xdg-home"
    xdg_dir_config_dir = tmp_path / "xdg-dir"
    xdg_config = xdg_config_dir / "ursa" / "config.yaml"
    local_config = tmp_path / "local-config.yaml"

    xdg_dir_config = xdg_dir_config_dir / "ursa" / "config.yaml"
    xdg_dir_config.parent.mkdir(parents=True)
    xdg_dir_config.write_text(
        yaml.safe_dump({
            "workspace": "xdg-dir-workspace",
            "llm_model": {"model": "openai:gpt-5-mini"},
        })
    )

    xdg_config.parent.mkdir(parents=True)
    xdg_config.write_text(
        yaml.safe_dump({
            "group": "xdg-group",
            "llm_model": {"model": "openai:gpt-5-mini"},
        })
    )
    local_config.write_text(
        yaml.safe_dump({
            "group": "local-group",
            "llm_model": {
                "model": "openai:gpt-5-mini",
                "base_url": "https://local.example/v1",
            },
        })
    )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_dir))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(xdg_dir_config_dir))
    monkeypatch.chdir(tmp_path)
    local_path = tmp_path / ".ursa" / "config.yaml"
    local_path.parent.mkdir(parents=True)
    local_path.write_text(local_config.read_text())

    parser = build_parser()
    args = parser.parse_args([])
    config = resolve_config(args)

    assert config.workspace == Path("xdg-dir-workspace")
    assert config.group == "local-group"
    assert config.llm_model.model == "openai:gpt-5-mini"
    assert config.llm_model.base_url == "https://local.example/v1"


def test_model_config_kwargs_includes_extra():
    cfg = ModelConfig(
        model="openai:gpt-5",
        ssl_verify=False,
    )
    cfg.model_extra["timeout"] = 30

    kwargs = cfg.kwargs
    assert kwargs["model"] == "openai:gpt-5"
    assert "http_client" in kwargs  # ssl_verify False triggers custom client
    assert "http_async_client" in kwargs
    assert kwargs["timeout"] == 30


def test_chat_model_config_kwargs_includes_max_completion_tokens():
    cfg = ChatModelConfig(model="openai:gpt-5", max_completion_tokens=1024)

    kwargs = cfg.kwargs

    assert kwargs["model"] == "openai:gpt-5"
    assert kwargs["max_completion_tokens"] == 1024


def test_chat_model_config_initializes_chat_model(monkeypatch):
    captured_kwargs = {}

    def fake_init_chat_model(**kwargs):
        captured_kwargs.update(kwargs)
        return "chat-model"

    monkeypatch.setattr(
        "ursa.cli.config.init_chat_model",
        fake_init_chat_model,
    )
    cfg = ChatModelConfig(model="openai:gpt-5", max_completion_tokens=1024)

    result = cfg.init_chat_model()

    assert result == "chat-model"
    assert captured_kwargs["model"] == "openai:gpt-5"
    assert captured_kwargs["max_completion_tokens"] == 1024
    assert captured_kwargs["use_responses_api"] is True


def test_emb_model_config_initializes_embedding_model(monkeypatch):
    captured_kwargs = {}

    def fake_init_embeddings(**kwargs):
        captured_kwargs.update(kwargs)
        return "embedding-model"

    monkeypatch.setattr(
        "ursa.cli.config.init_embeddings",
        fake_init_embeddings,
    )
    cfg = EmbModelConfig(model="openai:text-embedding-3-large")

    result = cfg.init_embedding()

    assert result == "embedding-model"
    assert captured_kwargs["model"] == "openai:text-embedding-3-large"
    assert "use_responses_api" not in captured_kwargs


def test_model_config_openai_uses_truststore_client():
    cfg = ModelConfig(model="openai:text-embedding-3-large")

    kwargs = cfg.kwargs

    assert kwargs["model"] == "openai:text-embedding-3-large"
    assert "http_client" in kwargs
    assert "http_async_client" in kwargs


def test_model_config_ollama_uses_client_kwargs():
    cfg = ModelConfig(model="ollama:nomic-embed-text:latest")

    kwargs = cfg.kwargs

    assert kwargs["model"] == "ollama:nomic-embed-text:latest"
    assert "http_client" not in kwargs
    assert "http_async_client" not in kwargs
    assert kwargs["client_kwargs"]["verify"] is not False


def test_api_key_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_ENV_API_KEY", "super-secret-key")
    parser = build_parser()
    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--llm_model.api_key_env",
        "TEST_ENV_API_KEY",
    ])

    config = UrsaConfig.from_namespace(args)
    assert config.llm_model.api_key_env == "TEST_ENV_API_KEY"
    assert config.llm_model.kwargs["api_key"] == "super-secret-key"
    assert "api_key_env" not in config.llm_model.kwargs.keys()


def test_model_config_omits_unset_or_blank_base_url_for_provider_default():
    for value in (None, "", "   "):
        cfg = ModelConfig(model="openai:gpt-5", base_url=value)

        kwargs = cfg.kwargs

        assert cfg.base_url is None
        assert "base_url" not in kwargs


def test_model_config_strips_configured_base_url():
    cfg = ModelConfig(
        model="openai:gpt-5",
        base_url=" https://models.example.org/v1 ",
    )

    kwargs = cfg.kwargs

    assert cfg.base_url == "https://models.example.org/v1"
    assert kwargs["base_url"] == "https://models.example.org/v1"


def test_model_config_omits_blank_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = ModelConfig(model="openai:gpt-5", api_key_env="   ")

    kwargs = cfg.kwargs

    assert cfg.api_key_env is None
    assert "api_key" not in kwargs
    assert "api_key_env" not in kwargs


def test_inference_provider_applies_to_llm_model():
    config = UrsaConfig(
        inference_providers={
            "local-openai": {
                "base_url": " https://models.example.org/v1 ",
                "ssl_verify": False,
                "api_key_env": "PROVIDER_API_KEY",
                "timeout": 45,
            }
        },
        llm_model={
            "model": "openai:gpt-5",
            "inference_provider": "local-openai",
        },
    )

    resolved = config.llm_model.resolve_inference_provider(
        config.inference_providers
    )

    assert config.llm_model.inference_provider == "local-openai"
    assert config.llm_model.base_url is None
    assert resolved.base_url == "https://models.example.org/v1"
    assert resolved.ssl_verify is False
    assert resolved.api_key_env == "PROVIDER_API_KEY"
    assert resolved.model_extra["timeout"] == 45


def test_inference_provider_applies_to_embedding_model():
    config = UrsaConfig(
        inference_providers={
            "local-embeddings": {
                "base_url": "https://embeddings.example.org/v1",
                "ssl_verify": False,
                "cache_dir": "/tmp/provider-cache",
            }
        },
        emb_model={
            "model": "openai:text-embedding-3-large",
            "inference_provider": "local-embeddings",
        },
    )

    assert config.emb_model is not None
    resolved = config.emb_model.resolve_inference_provider(
        config.inference_providers
    )
    assert config.emb_model.inference_provider == "local-embeddings"
    assert config.emb_model.base_url is None
    assert resolved.base_url == "https://embeddings.example.org/v1"
    assert resolved.ssl_verify is False
    assert resolved.model_extra["cache_dir"] == "/tmp/provider-cache"


def test_model_config_explicit_values_override_inference_provider():
    config = UrsaConfig(
        inference_providers={
            "shared": {
                "base_url": "https://provider.example.org/v1",
                "ssl_verify": False,
                "api_key_env": "PROVIDER_API_KEY",
                "timeout": 30,
                "seed": 111,
            }
        },
        llm_model={
            "model": "openai:gpt-5",
            "inference_provider": "shared",
            "base_url": "https://model.example.org/v1",
            "ssl_verify": True,
            "api_key_env": "MODEL_API_KEY",
            "timeout": 60,
        },
    )

    resolved = config.llm_model.resolve_inference_provider(
        config.inference_providers
    )

    assert resolved.base_url == "https://model.example.org/v1"
    assert resolved.ssl_verify is True
    assert resolved.api_key_env == "MODEL_API_KEY"
    assert resolved.model_extra["timeout"] == 60
    assert resolved.model_extra["seed"] == 111


def test_unknown_inference_provider_is_validation_error():
    with pytest.raises(
        ValueError, match="Unknown inference_provider 'missing'"
    ):
        UrsaConfig(
            llm_model={
                "model": "openai:gpt-5",
                "inference_provider": "missing",
            }
        )


def test_inference_provider_is_not_forwarded_to_langchain_kwargs():
    cfg = ChatModelConfig(
        model="openai:gpt-5",
        inference_provider="shared-provider",
    )

    kwargs = cfg.kwargs

    assert "inference_provider" not in kwargs


def test_model_config_resolve_provider_returns_self_when_unset():
    cfg = ChatModelConfig(model="openai:gpt-5")

    resolved = cfg.resolve_inference_provider({})

    assert resolved is cfg


def test_resolve_config_preserves_unresolved_inference_provider(tmp_path):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text(
        "\n".join([
            "inference_providers:",
            "  openai_public:",
            "    base_url: https://api.openai.com/v1",
            "    api_key_env: OPENAI_API_KEY",
            "llm_model:",
            "  model: openai:gpt-5.4",
            "  inference_provider: openai_public",
        ])
    )

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg_path)])
    config = resolve_config(args)

    assert config.llm_model.inference_provider == "openai_public"
    assert config.llm_model.base_url is None
    assert config.inference_providers["openai_public"].base_url == (
        "https://api.openai.com/v1"
    )


def test_resolve_ursa_config_promotes_use_web_to_agent_config():
    config = UrsaConfig(use_web=True)

    resolved = resolve_ursa_config(config)

    for agent_name in ["chat", "execute", "deep_review", "prompt"]:
        assert resolved.agent_config[agent_name]["use_web"] is True


def test_resolve_ursa_config_creates_tmp_workspace():
    config = UrsaConfig(workspace="tmp")

    resolved = resolve_ursa_config(config)

    assert resolved.workspace.exists()
    assert resolved._temp_workspace is not None
    assert resolved._temp_workspace.name == str(resolved.workspace)


def test_resolve_ursa_config_enforces_group_base_url_policy(tmp_path, monkeypatch):
    from ursa import security
    from ursa.security import GroupBaseURLPolicyError

    root = tmp_path / "ursa"
    group_root = root / "science"
    group_root.mkdir(parents=True)
    (group_root / "group.yaml").write_text(
        "allowed_base_urls:\n  - https://models.example.test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(security, "URSA_CACHE_DIR", root)

    config = UrsaConfig(
        group="science",
        llm_model={
            "model": "openai:gpt-test",
            "base_url": "https://disallowed.example.test/v1",
        },
    )

    with pytest.raises(GroupBaseURLPolicyError, match="not allowed"):
        resolve_ursa_config(config)
