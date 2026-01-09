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
