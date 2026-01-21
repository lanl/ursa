import yaml

from ursa.cli import build_parser, resolve_config
from ursa.cli.config import ModelConfig, UrsaConfig


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
        "--emb_model.max_completion_tokens",
        "1024",
    ])
    config_cli = resolve_config(args_cli)
    assert config_cli.workspace == cli_workspace
    assert config_cli.llm_model.model == "cli-model"
    assert config_cli.emb_model.max_completion_tokens == 1024


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
        "--emb_model.max_completion_tokens",
        "1024",
    ])

    config = resolve_config(args)

    assert config.workspace == cli_workspace
    assert config.llm_model.model == "openai:gpt-5-nano"
    assert config.llm_model.model_extra["temperature"] == 0.4
    assert config.emb_model.model == "openai:text-embedding-3-large"
    assert config.emb_model.max_completion_tokens == 1024
    assert config.emb_model.model_extra["cache_dir"] == "/tmp/cache"


def test_model_config_kwargs_includes_extra():
    cfg = ModelConfig(
        model="openai:gpt-5",
        max_completion_tokens=1024,
        ssl_verify=False,
    )
    cfg.model_extra["timeout"] = 30

    kwargs = cfg.kwargs
    assert kwargs["model"] == "openai:gpt-5"
    assert kwargs["max_completion_tokens"] == 1024
    assert "http_client" in kwargs  # ssl_verify False triggers custom client
    assert kwargs["timeout"] == 30
