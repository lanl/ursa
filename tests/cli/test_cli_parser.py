import gc
import importlib
from pathlib import Path
from unittest.mock import MagicMock
from warnings import catch_warnings, simplefilter

import pytest
import yaml
from jsonargparse import Namespace
from pydantic import ValidationError

from ursa.cli import (
    build_parser,
    main,
)
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    ModelConfig,
    UrsaConfig,
    config_search_paths,
    load_config_file,
    merge_ursa_config,
    resolve_ursa_config,
    xdg_config_search_paths,
)
from ursa.cli.print_config import (
    parse_print_config_spec,
    print_config,
)
from ursa.util.crossplatform import system_config_path, user_config_paths
from ursa.util.secrets import SecretReference


@pytest.fixture(autouse=True)
def _isolate_xdg_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-home"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg-dirs"))


def _stub_mcp_server(monkeypatch):
    mcp = MagicMock()
    hitl = MagicMock()
    hitl.as_mcp_server.return_value = mcp
    hitl_class = MagicMock(return_value=hitl)
    monkeypatch.setattr("ursa.cli.runtime.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    return hitl, mcp


def _stub_textual(monkeypatch):
    hitl_class = MagicMock()
    run_textual = MagicMock()
    monkeypatch.setattr("ursa.cli.runtime.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.app.run_textual", run_textual)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    return hitl_class, run_textual


def _parse_config_args(parser, args):
    """Parse the runtime namespace and its sparse config overrides."""
    return (
        parser.parse_args(args),
        parser.parse_args(args, defaults=False),
    )


def _resolve_args(parser, args):
    cfg, env_overrides = _parse_config_args(parser, args)
    return merge_ursa_config(cfg, env_overrides=env_overrides).resolve()


def test_cli_warns_about_legacy_unnamed_checkpoint(
    monkeypatch, tmp_path, capsys
):
    _stub_textual(monkeypatch)
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
    _stub_textual(monkeypatch)
    checkpoint = tmp_path / "db" / "checkpointer.db"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    main(["--workspace", str(tmp_path), "--name", "continued-agent"])

    assert capsys.readouterr().err == ""


def test_cli_does_not_warn_without_legacy_checkpoint(
    monkeypatch, tmp_path, capsys
):
    _stub_textual(monkeypatch)

    main(["--workspace", str(tmp_path)])

    assert capsys.readouterr().err == ""


def test_exec_uses_textual_one_shot_renderer(monkeypatch):
    hitl = MagicMock()
    run_once = MagicMock()
    monkeypatch.setattr("ursa.cli.runtime.HITL", MagicMock(return_value=hitl))
    monkeypatch.setattr("ursa.cli.app.run_textual_once", run_once)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    main(["exec", "#plan inspect this"])

    run_once.assert_called_once_with(hitl, "#plan inspect this")


@pytest.mark.parametrize("mode", ["interactive", "exec"])
def test_named_agent_reaches_textual_runtime(monkeypatch, mode):
    hitl_class = MagicMock()
    run_textual = MagicMock()
    run_once = MagicMock()
    monkeypatch.setattr("ursa.cli.runtime.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.app.run_textual", run_textual)
    monkeypatch.setattr("ursa.cli.app.run_textual_once", run_once)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    args = ["--name", "lab-assistant"]
    if mode == "exec":
        args.extend(["exec", "#plan inspect this"])

    main(args)

    config = hitl_class.call_args.args[0]
    assert config.agent_name == "lab-assistant"
    if mode == "interactive":
        run_textual.assert_called_once_with(hitl_class.return_value)
    else:
        run_once.assert_called_once_with(
            hitl_class.return_value, "#plan inspect this"
        )


@pytest.mark.parametrize("args", [[], ["mcp-server"], ["exec", "hello"]])
def test_cli_reports_runtime_initialization_error_without_traceback(
    monkeypatch, capsys, args
):
    monkeypatch.setattr(
        "ursa.cli.runtime.HITL",
        MagicMock(side_effect=ValueError("API credentials are invalid")),
    )
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    with pytest.raises(SystemExit, match="2"):
        main(args)

    stderr = capsys.readouterr().err
    assert stderr == "Error: API credentials are invalid\n"
    assert "Traceback" not in stderr


def test_exec_runs_prompt_with_textual_runtime(monkeypatch):
    hitl_class = MagicMock()
    run_once = MagicMock()
    monkeypatch.setattr("ursa.cli.runtime.HITL", hitl_class)
    monkeypatch.setattr("ursa.cli.app.run_textual_once", run_once)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    main(["exec", "summarize this"])

    hitl = hitl_class.return_value
    run_once.assert_called_once_with(hitl, "summarize this")


@pytest.mark.parametrize(
    ("modern_env", "cli_args", "expected"),
    [
        (None, [], "legacy-agent"),
        ("modern-agent", [], "modern-agent"),
        (None, ["--name", "cli-agent"], "cli-agent"),
    ],
)
def test_legacy_ursa_name_is_supported_with_deprecation_warning(
    monkeypatch, modern_env, cli_args, expected
):
    hitl_class, _ = _stub_textual(monkeypatch)
    monkeypatch.setenv("URSA_NAME", "legacy-agent")
    if modern_env is not None:
        monkeypatch.setenv("URSA_AGENT_NAME", modern_env)

    with pytest.warns(FutureWarning, match="URSA_NAME is deprecated"):
        main(cli_args)

    (config,), _ = hitl_class.call_args
    assert config.agent_name == expected


@pytest.mark.parametrize(
    "args",
    [
        ["list-rag-agents"],
        ["show-rag-agent", "docs"],
        ["delete-rag-agent", "docs"],
        ["save-rag-agent", "docs"],
    ],
)
def test_rag_metadata_commands_dispatch_before_config_resolution(
    monkeypatch, args
):
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    handle = MagicMock(return_value=True)
    monkeypatch.setattr("ursa.cli.handle_rag_command", handle)
    monkeypatch.setattr(
        "ursa.cli.merge_ursa_config",
        MagicMock(side_effect=AssertionError("must not resolve config")),
    )

    main(args)

    handle.assert_called_once()
    assert len(handle.call_args.args) == 1


def test_rag_model_commands_resolve_with_subcommand_group(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    config_file = tmp_path / "rag.yaml"
    config_file.write_text(
        yaml.safe_dump({
            "group": "file-group",
            "llm_model": {"base_url": "https://models.example/v1"},
        }),
        encoding="utf-8",
    )
    policy_calls = []
    monkeypatch.setattr(
        "ursa.cli.config.enforce_group_base_url_policy",
        lambda base_url, group: policy_calls.append((base_url, group)),
    )
    monkeypatch.setattr(
        "ursa.cli.handle_rag_command", MagicMock(return_value=True)
    )

    main([
        "rag-query",
        "--name",
        "docs",
        "--group",
        "rag-group",
        "--config",
        str(config_file),
        "question",
    ])

    assert policy_calls == [("https://models.example/v1", "rag-group")]


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
    monkeypatch.setattr("ursa.cli.runtime.HITL", hitl_class)
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
    assert ursa_config.llm_model.model == "llama3.1"
    assert ursa_config.llm_model.model_provider == "ollama"
    assert ursa_config.llm_model.base_url == "http://localhost:11434"


def test_mcp_server_config_flag_parses_alongside_transport(
    monkeypatch, tmp_path
):
    """--config coexists with MCP transport options on the subcommand."""
    hitl_class = MagicMock()
    hitl_class.return_value = MagicMock()
    mcp = MagicMock()
    hitl_class.return_value.as_mcp_server.return_value = mcp
    monkeypatch.setattr("ursa.cli.runtime.HITL", hitl_class)
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
    assert ursa_config.llm_model.model == "llama3.1"
    assert ursa_config.llm_model.model_provider == "ollama"
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
    assert parse_print_config_spec("file+,resolved") == (
        "file+",
        "resolved",
    )


@pytest.mark.parametrize("spec", ["bogus", "final,bogus", "bogus,resolved"])
def test_parse_print_config_spec_rejects_invalid_values(spec):
    with pytest.raises(ValueError):
        parse_print_config_spec(spec)


@pytest.mark.parametrize("spec", ["bogus", "final,bogus", "project,resolved"])
def test_print_config_parser_rejects_invalid_values(spec):
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args([f"--print-config={spec}"])


def test_cli_config_resolution_preserves_tmp_workspace_owner():
    parser = build_parser()
    config = _resolve_args(parser, ["--workspace", "tmp"])

    assert config.workspace.exists()
    assert config._temp_workspace is not None
    assert config._temp_workspace.name == str(config.workspace)


def test_file_config_resolution_preserves_tmp_workspace_owner(tmp_path):
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
    serialized = original_config.model_dump()
    serialized["llm_model"]["inference_provider"] = None
    yaml_text = yaml.safe_dump(serialized)

    cfg_path = tmp_path / "round-trip.yml"
    cfg_path.write_text(yaml_text)

    parser = build_parser()
    loaded_config = _resolve_args(parser, ["--config", str(cfg_path)])

    assert loaded_config.model_dump() == serialized


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
    assert config_cli.emb_model.model == "text-embedding-3-large"
    assert config_cli.emb_model.model_provider == "openai"


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
    assert config.llm_model.model == "gpt-env"
    assert config.llm_model.model_provider == "openai"
    assert config.emb_model.model == "gpt-5"
    assert config.emb_model.model_provider == "openai"


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

    assert config.llm_model.model == "gpt-5-small"
    assert config.llm_model.model_provider == "openai"
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
    assert config.llm_model.model == "gpt-5-nano"
    assert config.llm_model.model_provider == "openai"
    assert config.llm_model.model_extra["temperature"] == 0.4
    assert config.emb_model.model == "text-embedding-3-large"
    assert config.emb_model.model_provider == "openai"
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
        env_overrides={
            "group": "default",
            "llm_model": {
                "model": "openai:gpt-5.4",
                "ssl_verify": True,
            },
        },
    )

    assert config.group == "default"
    assert config.llm_model.model == "gpt-5.4"
    assert config.llm_model.model_provider == "openai"
    assert config.llm_model.ssl_verify is True


def test_merge_ursa_config_normalizes_empty_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "ursa.cli.config.config_search_paths",
        lambda cfg, level="final": [cfg_path],
    )

    config = merge_ursa_config(Namespace(), env_overrides={})

    assert config == UrsaConfig()
    assert load_config_file(cfg_path) == {}


def test_load_config_file_rejects_non_mapping_yaml(tmp_path):
    cfg_path = tmp_path / "list.yaml"
    cfg_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config_file(cfg_path)


def test_agent_config_null_is_normalized_with_deprecation_warning(caplog):
    caplog.set_level("WARNING", logger="ursa.cli.config")

    config = UrsaConfig(agent_config=None)

    assert config.agent_config == {}
    assert "agent_config to null is deprecated" in caplog.text


def test_xdg_config_search_paths_honor_env_overrides(tmp_path, monkeypatch):
    home = tmp_path / "home"
    xdg_home = tmp_path / "xdg-home"
    xdg_dir_1 = tmp_path / "xdg-dir-1"
    xdg_dir_2 = tmp_path / "xdg-dir-2"

    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    monkeypatch.setattr(Path, "home", lambda: home)
    path_list_separator = ";" if tmp_path.drive else ":"
    monkeypatch.setenv(
        "XDG_CONFIG_DIRS",
        path_list_separator.join((str(xdg_dir_1), str(xdg_dir_2))),
    )

    assert xdg_config_search_paths() == [
        system_config_path(),
        *user_config_paths(),
    ]


def test_xdg_config_dirs_do_not_replace_native_system_config(
    tmp_path, monkeypatch
):
    high = tmp_path / "high" / "ursa" / "config.yaml"
    low = tmp_path / "low" / "ursa" / "config.yaml"
    high.parent.mkdir(parents=True)
    low.parent.mkdir(parents=True)
    high.write_text("group: high\n", encoding="utf-8")
    low.write_text("group: low\n", encoding="utf-8")
    path_list_separator = ";" if tmp_path.drive else ":"
    monkeypatch.setenv(
        "XDG_CONFIG_DIRS",
        path_list_separator.join((str(high.parents[1]), str(low.parents[1]))),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "missing"))
    monkeypatch.setattr("ursa.cli.config.system_config_path", lambda: high)

    config = merge_ursa_config(Namespace(), env_overrides={})

    assert config.group == "high"


def test_config_search_paths_ignores_project_config(tmp_path, monkeypatch):
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
        explicit_cfg,
    ]


def test_config_search_paths_cumulative_levels_include_lower_precedence(
    tmp_path, monkeypatch
):
    system = tmp_path / "system.yaml"
    user = tmp_path / "user.yaml"
    explicit = tmp_path / "explicit.yaml"
    for path in (system, user, explicit):
        path.write_text("{}\n")
    monkeypatch.setattr("ursa.cli.config.system_config_paths", lambda: [system])
    monkeypatch.setattr("ursa.cli.config.user_config_paths", lambda: [user])
    cfg = Namespace(config=explicit)

    assert config_search_paths(cfg, "user") == [user]
    assert config_search_paths(cfg, "user+") == [system, user]
    assert config_search_paths(cfg, "file") == [explicit]
    assert config_search_paths(cfg, "file+") == [system, user, explicit]


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

    config = merge_ursa_config(Namespace(), env_overrides={})

    assert "openai_project" in config.inference_providers
    assert config.llm_model.model == "gpt-5.4-mini"
    assert config.llm_model.model_provider == "openai"
    assert config.llm_model.inference_provider == "openai_project"


def test_model_config_kwargs_includes_extra():
    cfg = ModelConfig(
        model="openai:gpt-5",
        ssl_verify=False,
    )
    cfg.model_extra["timeout"] = 30

    kwargs = cfg.kwargs
    assert kwargs["model"] == "gpt-5"
    assert kwargs["model_provider"] == "openai"
    assert "http_client" in kwargs  # ssl_verify False triggers custom client
    assert "http_async_client" in kwargs
    assert kwargs["timeout"] == 30


def test_chat_model_config_kwargs_includes_max_completion_tokens():
    cfg = ChatModelConfig(model="openai:gpt-5", max_completion_tokens=1024)

    kwargs = cfg.kwargs

    assert kwargs["model"] == "gpt-5"
    assert kwargs["model_provider"] == "openai"
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
    assert captured_kwargs["model"] == "gpt-5"
    assert captured_kwargs["model_provider"] == "openai"
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
    assert captured_kwargs["model"] == "text-embedding-3-large"
    assert captured_kwargs["provider"] == "openai"
    assert "model_provider" not in captured_kwargs
    assert "use_responses_api" not in captured_kwargs


def test_emb_model_kwargs_use_embedding_provider_argument():
    config = EmbModelConfig(
        model="text-embedding-3-small",
        model_provider="openai",
    )

    assert config.kwargs["provider"] == "openai"
    assert "model_provider" not in config.kwargs


def test_model_config_openai_uses_truststore_client():
    cfg = ModelConfig(model="openai:text-embedding-3-large")

    kwargs = cfg.kwargs

    assert kwargs["model"] == "text-embedding-3-large"
    assert kwargs["model_provider"] == "openai"
    assert "http_client" in kwargs
    assert "http_async_client" in kwargs


def test_model_config_ollama_uses_client_kwargs():
    cfg = ModelConfig(model="ollama:nomic-embed-text:latest")

    kwargs = cfg.kwargs

    assert kwargs["model"] == "nomic-embed-text:latest"
    assert kwargs["model_provider"] == "ollama"
    assert "http_client" not in kwargs
    assert "http_async_client" not in kwargs
    assert kwargs["client_kwargs"]["verify"] is not False


def test_api_key_env_cli_option(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_ENV_API_KEY", "super-secret-key")
    parser = build_parser()
    with catch_warnings(record=True) as warnings:
        simplefilter("always")
        config = _resolve_args(
            parser,
            [
                "--workspace",
                str(tmp_path),
                "--llm_model.api_key_env",
                "TEST_ENV_API_KEY",
            ],
        )

    assert not any("api_key_env" in str(item.message) for item in warnings)
    assert config.llm_model.api_key == SecretReference(env="TEST_ENV_API_KEY")
    assert config.llm_model.kwargs["api_key"] == "super-secret-key"
    assert "api_key_env" not in config.llm_model.model_dump()


def test_api_key_env_cli_options_are_visible_in_help():
    parser = build_parser()
    help_text = parser.format_help()

    assert "--llm_model.api_key_env" in help_text
    assert "--emb_model.api_key_env" in help_text
    assert (
        parser._option_string_actions["--llm_model.api_key_env"].container
        is parser._option_string_actions["--llm_model"].container
    )
    assert (
        parser._option_string_actions["--emb_model.api_key_env"].container
        is parser._option_string_actions["--emb_model"].container
    )


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

    assert "api_key" not in kwargs
    assert "api_key_env" not in cfg.model_dump()


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
    assert resolved.inference_provider == "local-openai"
    assert resolved.base_url == "https://models.example.org/v1"
    assert resolved.ssl_verify is False
    assert resolved.api_key == SecretReference(env="PROVIDER_API_KEY")
    assert resolved.model_extra["timeout"] == 45
    assert "inference_provider" not in resolved.kwargs


def test_model_config_model_provider_overrides_inference_provider():
    config = UrsaConfig(
        inference_providers={"gateway": {"model_provider": "anthropic"}},
        llm_model={
            "model": "gpt-test",
            "model_provider": "openai",
            "inference_provider": "gateway",
        },
    )

    resolved = config.resolve()

    assert resolved.llm_model.model_provider == "openai"
    assert resolved.inference_providers["gateway"].model_extra == {
        "model_provider": "anthropic"
    }


def test_inference_provider_sets_unset_model_provider():
    config = UrsaConfig(
        inference_providers={"gateway": {"model_provider": "anthropic"}},
        llm_model={
            "model": "gpt-test",
            "inference_provider": "gateway",
        },
    )

    resolved = config.resolve()

    assert "model_provider" not in config.llm_model.model_fields_set
    assert resolved.llm_model.model_provider == "anthropic"


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
    assert resolved.inference_provider == "local-embeddings"
    assert resolved.base_url == "https://embeddings.example.org/v1"
    assert resolved.ssl_verify is False
    assert resolved.model_extra["cache_dir"] == "/tmp/provider-cache"
    assert "inference_provider" not in resolved.kwargs


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
            "ssl_verify": True,
            "api_key_env": "MODEL_API_KEY",
            "timeout": 60,
        },
    )

    resolved = config.llm_model.resolve_inference_provider(
        config.inference_providers
    )

    assert resolved.base_url == "https://provider.example.org/v1"
    assert resolved.ssl_verify is True
    assert resolved.api_key == SecretReference(env="MODEL_API_KEY")
    assert resolved.model_extra["timeout"] == 60
    assert resolved.model_extra["seed"] == 111
    assert resolved.model_fields_set == config.llm_model.model_fields_set
    assert "seed" not in resolved.model_fields_set


def test_unknown_inference_provider_is_validation_error():
    with pytest.raises(
        ValueError, match="unknown inference_provider 'missing'"
    ):
        UrsaConfig(
            llm_model={
                "model": "openai:gpt-5",
                "inference_provider": "missing",
            }
        )


def test_invalid_provider_catalog_preserves_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        UrsaConfig.model_validate({
            "inference_providers": {"shared": {"ssl_verify": "not-a-boolean"}},
            "llm_model": {"inference_provider": "shared"},
        })

    assert exc_info.value.errors()[0]["loc"] == (
        "inference_providers",
        "shared",
        "ssl_verify",
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

    assert resolved.llm_model.inference_provider == "openai_public"
    assert resolved.llm_model.base_url == "https://api.openai.com/v1"
    assert "inference_provider" not in resolved.llm_model.kwargs


def test_resolved_config_without_inference_provider_can_be_reloaded():
    resolved = UrsaConfig(
        llm_model={
            "model": "openai:gpt-5",
            "base_url": "https://models.example/v1",
        }
    ).resolve()

    reloaded = UrsaConfig.model_validate(resolved.model_dump()).resolve()

    assert reloaded.llm_model.inference_provider is None
    assert reloaded.llm_model.base_url == "https://models.example/v1"
    assert reloaded.llm_model.model == "gpt-5"
    assert reloaded.llm_model.model_provider == "openai"


@pytest.mark.parametrize(
    ("stage", "expected_model"),
    [
        ("merged", {"inference_provider": "shared"}),
        (
            "resolved",
            {
                "base_url": "https://provider.example/v1",
                "inference_provider": "shared",
                "ssl_verify": False,
            },
        ),
    ],
)
def test_print_config_includes_defaults_and_nulls(
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
    resolved = resolve_ursa_config(merged)
    print_config_module = importlib.import_module("ursa.cli.print_config")
    monkeypatch.setattr(
        print_config_module,
        "merge_ursa_config",
        lambda cfg, level, env_overrides, cli_overrides: merged,
    )
    monkeypatch.setattr(
        print_config_module,
        "resolve_ursa_config",
        lambda config: resolved if config is merged else config,
    )

    assert print_config(Namespace(print_config=stage), {}) is True

    output = yaml.safe_load(capsys.readouterr().out)
    assert output["workspace"] == "."
    assert output["agent_name"] is None
    assert output["emb_model"] is None
    assert output["agent_config"] == {}
    for key, value in expected_model.items():
        assert output["llm_model"][key] == value
    assert output["inference_providers"]["shared"] == {
        "base_url": "https://provider.example/v1",
        "api_key": None,
        "ssl_verify": False,
    }
    assert output["inference_providers"]["openai"]["api_key"] == {
        "env": "OPENAI_API_KEY",
        "keyring": None,
    }
    assert output["mcp_servers"] == {
        "example": {
            "transport": "stdio",
            "command": "example-server",
            "args": [],
            "env": None,
            "cwd": None,
            "encoding": "utf-8",
            "encoding_error_handler": "strict",
        }
    }


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


def test_resolve_ursa_config_uses_existing_literal_tmp_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tmp").mkdir()

    resolved = resolve_ursa_config(UrsaConfig(workspace="tmp"))

    assert resolved.workspace == Path("tmp")
    assert resolved._temp_workspace is None


def test_resolve_ursa_config_reuses_tmp_workspace_owner():
    first = resolve_ursa_config(UrsaConfig(workspace="tmp"))
    second = resolve_ursa_config(first)
    workspace = second.workspace

    assert second._temp_workspace is first._temp_workspace
    del first
    gc.collect()
    assert workspace.exists()


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
