import importlib
from unittest.mock import MagicMock

import pytest
import yaml
from jsonargparse import Namespace
from openai import OpenAIError

from ursa.cli import (
    build_parser,
    main,
    resolve_config,
)
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    ModelConfig,
    UrsaConfig,
    config_search_paths,
    merge_ursa_config,
    resolve_ursa_config,
    xdg_config_search_paths,
)
from ursa.cli.print_config import (
    parse_print_config_spec,
    print_config,
)


@pytest.fixture(autouse=True)
def _isolate_xdg_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-home"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg-dirs"))


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


def _parse_config_args(parser, args):
    """Parse the runtime namespace and its sparse config overrides."""
    return (
        parser.parse_args(args),
        parser.parse_args(args, defaults=False),
    )


def _resolve_args(parser, args):
    cfg, overrides = _parse_config_args(parser, args)
    return resolve_config(cfg, overrides)


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

    assert args.workspace == tmp_path / "workspace"
    assert args.llm_model.model == "openai:gpt-5-nano"
    assert args.llm_model.max_completion_tokens == 2048


def test_print_config_flag_defaults_to_resolved_string():
    parser = build_parser()
    args = parser.parse_args(["--print-config"])

    assert args["print_config"] == "resolved"


def test_parse_print_config_spec_accepts_stage_only_and_level_stage():
    assert parse_print_config_spec("resolved") == ("final", "resolved")
    assert parse_print_config_spec("merged") == ("final", "merged")
    assert parse_print_config_spec("file,resolved") == (
        "file",
        "resolved",
    )
    assert parse_print_config_spec("project+,resolved") == (
        "project+",
        "resolved",
    )


@pytest.mark.parametrize("spec", ["bogus", "final,bogus", "bogus,resolved"])
def test_parse_print_config_spec_rejects_invalid_values(spec):
    with pytest.raises(ValueError):
        parse_print_config_spec(spec)


def test_resolve_config_preserves_cli_tmp_workspace_owner():
    parser = build_parser()
    config = _resolve_args(parser, ["--workspace", "tmp"])

    assert config.workspace.exists()
    assert config._temp_workspace is not None
    assert config._temp_workspace.name == str(config.workspace)


def test_resolve_config_preserves_file_tmp_workspace_owner(tmp_path):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text("workspace: tmp\n")
    parser = build_parser()
    config = _resolve_args(parser, ["--config", str(cfg_path)])

    assert config.workspace.exists()
    assert config._temp_workspace is not None
    assert config._temp_workspace.name == str(config.workspace)


def test_cli_applies_chat_only_openai_defaults_to_llm_model():
    parser = build_parser()
    config = _resolve_args(parser, [])

    assert isinstance(config.llm_model, ChatModelConfig)
    assert config.llm_model.kwargs["use_responses_api"] is True


def test_cli_does_not_apply_chat_only_openai_defaults_to_emb_model():
    parser = build_parser()
    config = _resolve_args(
        parser,
        [
            "--emb_model.model",
            "openai:text-embedding-3-large",
        ],
    )

    assert isinstance(config.emb_model, EmbModelConfig)
    assert not isinstance(config.emb_model, ChatModelConfig)
    assert "use_responses_api" not in config.emb_model.kwargs


def test_print_config_yaml_round_trip(tmp_path):
    parser = build_parser()
    original_config = _resolve_args(
        parser,
        [
            "--workspace",
            str(tmp_path / "original"),
            "--llm_model.model",
            "openai:gpt-5-nano",
        ],
    )
    yaml_text = yaml.safe_dump(original_config.model_dump())

    cfg_path = tmp_path / "round-trip.yml"
    cfg_path.write_text(yaml_text)

    parser = build_parser()
    loaded_config = _resolve_args(parser, ["--config", str(cfg_path)])

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

    config_env = _resolve_args(parser, ["--config", str(cfg_path)])
    assert config_env.workspace == env_workspace
    assert config_env.llm_model.model == "env-model"

    cli_workspace = tmp_path / "cli-workspace"
    cli_workspace.mkdir()
    config_cli = _resolve_args(
        parser,
        [
            "--config",
            str(cfg_path),
            "--emb_model.model",
            "openai:text-embedding-3-large",
            "--workspace",
            str(cli_workspace),
            "--llm_model.model",
            "cli-model",
        ],
    )
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
    config = _resolve_args(parser, ["--config", str(cfg_path)])

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
    config = _resolve_args(parser, ["--config", str(cfg_path)])

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
    config = _resolve_args(
        parser,
        [
            "--config",
            str(cfg_path),
            "--emb_model.model",
            "openai:text-embedding-3-large",
            "--workspace",
            str(cli_workspace),
            "--llm_model.model",
            "openai:gpt-5-nano",
        ],
    )

    assert config.workspace == cli_workspace
    assert config.llm_model.model == "openai:gpt-5-nano"
    assert config.llm_model.model_extra["temperature"] == 0.4
    assert config.emb_model.model == "openai:text-embedding-3-large"
    assert config.emb_model.model_extra["cache_dir"] == "/tmp/cache"


def test_sparse_parser_omits_config_defaults():
    parser = build_parser()

    assert parser.parse_args([], defaults=False).as_dict() == {}


@pytest.mark.parametrize(
    ("value", "expected"), [("true", True), ("false", False)]
)
def test_cli_parser_preserves_explicit_ssl_verify(value, expected):
    parser = build_parser()

    assert parser.parse_args(
        ["--group", "default", "--llm_model.ssl_verify", value],
        defaults=False,
    ).as_dict() == {
        "group": "default",
        "llm_model": {"ssl_verify": expected},
    }


def test_sparse_parser_reads_environment(monkeypatch):
    monkeypatch.setenv("URSA_GROUP", "default")
    monkeypatch.setenv("URSA_LLM_MODEL__SSL_VERIFY", "true")

    overrides = build_parser().parse_args([], defaults=False)

    assert overrides.as_dict() == {
        "group": "default",
        "llm_model": {"ssl_verify": True},
    }


def test_merge_ursa_config_applies_sparse_overrides(tmp_path, monkeypatch):
    cfg_path = tmp_path / "ursa.yml"
    cfg_path.write_text(
        yaml.safe_dump({
            "group": "restricted",
            "llm_model": {
                "model": "openai:file-model",
                "ssl_verify": False,
            },
        })
    )
    monkeypatch.setattr(
        "ursa.cli.config.config_search_paths",
        lambda cfg, level="final": [cfg_path],
    )

    config = merge_ursa_config(
        Namespace(),
        overrides={
            "group": "default",
            "llm_model": {
                "model": "openai:gpt-5.4",
                "ssl_verify": True,
            },
        },
    )

    assert config.group == "default"
    assert config.llm_model.model == "openai:gpt-5.4"
    assert config.llm_model.ssl_verify is True


def test_xdg_config_search_paths_honor_env_overrides(tmp_path, monkeypatch):
    xdg_home = tmp_path / "xdg-home"
    xdg_dir_1 = tmp_path / "xdg-dir-1"
    xdg_dir_2 = tmp_path / "xdg-dir-2"

    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    monkeypatch.setenv("XDG_CONFIG_DIRS", f"{xdg_dir_1}:{xdg_dir_2}")

    assert xdg_config_search_paths() == [
        xdg_dir_1 / "ursa" / "config.yaml",
        xdg_dir_2 / "ursa" / "config.yaml",
        xdg_home / "ursa" / "config.yaml",
    ]


def test_config_search_paths_returns_existing_sources_in_precedence_order(
    tmp_path, monkeypatch
):
    xdg_dir = tmp_path / "xdg"
    project_dir = tmp_path / "project"
    explicit_cfg = tmp_path / "explicit.yaml"
    user_cfg = xdg_dir / "ursa" / "config.yaml"
    project_cfg = project_dir / ".ursa" / "config.yaml"
    user_cfg.parent.mkdir(parents=True)
    project_cfg.parent.mkdir(parents=True)
    for path in (user_cfg, project_cfg, explicit_cfg):
        path.write_text("{}\n")

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "missing"))

    assert config_search_paths(Namespace(config=explicit_cfg)) == [
        user_cfg,
        project_cfg,
        explicit_cfg,
    ]


def test_merge_ursa_config_validates_provider_after_merging_layers(
    tmp_path, monkeypatch
):
    user_config = tmp_path / "user.yaml"
    project_config = tmp_path / "project.yaml"
    user_config.write_text(
        yaml.safe_dump({
            "inference_providers": {
                "openai_project": {
                    "base_url": "https://project.example/v1",
                    "ssl_verify": False,
                }
            }
        })
    )
    project_config.write_text(
        yaml.safe_dump({
            "llm_model": {
                "model": "openai:gpt-5.4-mini",
                "inference_provider": "openai_project",
            }
        })
    )
    monkeypatch.setattr(
        "ursa.cli.config.config_search_paths",
        lambda cfg, level="final": [user_config, project_config],
    )

    config = merge_ursa_config(Namespace(), overrides={})

    assert "openai_project" in config.inference_providers
    assert config.llm_model.model == "openai:gpt-5.4-mini"
    assert config.llm_model.inference_provider == "openai_project"


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


def test_chat_model_config_rejects_unresolved_provider(monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.config.init_chat_model",
        lambda **kwargs: "chat-model",
    )
    cfg = ChatModelConfig(
        model="openai:gpt-5",
        inference_provider="shared",
    )

    with pytest.raises(
        ValueError,
        match="references unresolved inference provider 'shared'",
    ):
        cfg.init_chat_model()


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
    config = _resolve_args(
        parser,
        [
            "--workspace",
            str(tmp_path),
            "--llm_model.api_key_env",
            "TEST_ENV_API_KEY",
        ],
    )

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
    assert resolved.inference_provider is None
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
    assert resolved.inference_provider is None
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


def test_model_config_kwargs_rejects_unresolved_inference_provider():
    cfg = ChatModelConfig(
        model="openai:gpt-5",
        inference_provider="shared-provider",
    )

    with pytest.raises(
        ValueError,
        match="references unresolved inference provider 'shared-provider'",
    ):
        _ = cfg.kwargs


def test_model_config_ssl_verify_defaults_to_true():
    cfg = ChatModelConfig(model="openai:gpt-5")

    assert cfg.ssl_verify is True


def test_model_config_explicit_null_extra_clears_provider_default():
    config = UrsaConfig(
        inference_providers={"shared": {"timeout": 30}},
        llm_model={
            "model": "openai:gpt-5",
            "inference_provider": "shared",
            "timeout": None,
        },
    )

    resolved = config.llm_model.resolve_inference_provider(
        config.inference_providers
    )

    assert resolved.model_extra["timeout"] is None
    assert "timeout" not in resolved.kwargs


def test_resolve_ursa_config_applies_inference_provider():
    config = UrsaConfig(
        inference_providers={
            "openai_public": {"base_url": "https://api.openai.com/v1"}
        },
        llm_model={"inference_provider": "openai_public"},
    )

    resolved = resolve_ursa_config(config)

    assert resolved.llm_model.inference_provider is None
    assert resolved.llm_model.base_url == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    ("stage", "expected_model"),
    [
        ("merged", {"inference_provider": "shared"}),
        (
            "resolved",
            {
                "base_url": "https://provider.example/v1",
                "ssl_verify": False,
            },
        ),
    ],
)
def test_print_config_omits_defaults_and_nulls(
    monkeypatch, capsys, stage, expected_model
):
    merged = UrsaConfig(
        inference_providers={
            "shared": {
                "base_url": "https://provider.example/v1",
                "ssl_verify": False,
            }
        },
        llm_model={"inference_provider": "shared"},
        mcp_servers={
            "example": {
                "transport": "stdio",
                "command": "example-server",
            }
        },
    )
    resolved = UrsaConfig(
        inference_providers=merged.inference_providers,
        llm_model={
            "base_url": "https://provider.example/v1",
            "ssl_verify": False,
        },
        mcp_servers=merged.mcp_servers,
    )
    print_config_module = importlib.import_module("ursa.cli.print_config")
    monkeypatch.setattr(
        print_config_module,
        "merge_ursa_config",
        lambda cfg, level, overrides: merged,
    )
    monkeypatch.setattr(
        print_config_module,
        "resolve_ursa_config",
        lambda config: resolved if config is merged else config,
    )

    assert print_config(Namespace(print_config=stage), {}) is True

    output = yaml.safe_load(capsys.readouterr().out)
    assert "workspace" not in output
    assert output["llm_model"] == expected_model
    assert output["inference_providers"] == {
        "shared": {
            "base_url": "https://provider.example/v1",
            "ssl_verify": False,
        }
    }
    assert output["mcp_servers"] == {
        "example": {
            "transport": "stdio",
            "command": "example-server",
        }
    }


def test_resolve_ursa_config_promotes_use_web_to_agent_config():
    config = UrsaConfig(use_web=True)

    resolved = resolve_ursa_config(config)

    for agent_name in ["chat", "execute", "deep_review", "prompt"]:
        assert resolved.agent_config[agent_name]["use_web"] is True


def test_resolve_ursa_config_use_web_only_fills_missing_values():
    config = UrsaConfig(
        use_web=True,
        agent_config={
            "chat": {"use_web": False},
            "execute": {},
            "prompt": {"temperature": 0.2},
        },
    )

    resolved = resolve_ursa_config(config)

    assert resolved.agent_config["chat"]["use_web"] is False
    assert resolved.agent_config["execute"]["use_web"] is True
    assert resolved.agent_config["deep_review"]["use_web"] is True
    assert resolved.agent_config["prompt"]["use_web"] is True
    assert resolved.agent_config["prompt"]["temperature"] == 0.2


def test_resolve_ursa_config_creates_tmp_workspace():
    config = UrsaConfig(workspace="tmp")

    resolved = resolve_ursa_config(config)

    assert resolved.workspace.exists()
    assert resolved._temp_workspace is not None
    assert resolved._temp_workspace.name == str(resolved.workspace)


def test_resolve_ursa_config_checks_group_base_url_policy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ursa.cli.config.enforce_group_base_url_policy",
        lambda base_url, group: calls.append((base_url, group)),
    )
    config = UrsaConfig(
        group="science",
        llm_model={
            "model": "openai:gpt-test",
            "base_url": "https://models.example.test/v1",
        },
        emb_model={
            "model": "openai:text-embedding-test",
            "base_url": "https://embeddings.example.test/v1",
        },
    )

    resolve_ursa_config(config)

    assert calls == [
        ("https://models.example.test/v1", "science"),
        ("https://embeddings.example.test/v1", "science"),
    ]
