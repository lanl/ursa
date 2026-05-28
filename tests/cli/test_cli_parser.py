import yaml

from ursa.cli import build_parser, resolve_config
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    ModelConfig,
    UrsaConfig,
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


def test_print_config_flag_sets_bool_and_preserves_defaults():
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
