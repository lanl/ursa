import asyncio
import base64
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

harbor = pytest.importorskip("harbor")

from harbor.models.task.config import (  # noqa: E402
    EnvironmentConfig,
    MCPServerConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from ursa.agents import BaseAgent  # noqa: E402
from ursa.agents.base import AgentWithTools  # noqa: E402
from ursa.cli import config as config_module  # noqa: E402
from ursa.cli.config import UrsaConfig  # noqa: E402
from ursa.integrations.harbor import (  # noqa: E402
    UrsaHarborAgent,
    make_harbor_agent,
)
from ursa.integrations.harbor_runner import (  # noqa: E402
    _agent_config,
    _attach_mcp_tools,
    _capture_output,
    _close_checkpoint,
    _usage,
)
from ursa.integrations.harbor_runner import _run as _runner_run  # noqa: E402
from ursa.integrations.harbor_runner import (  # noqa: E402
    main as _runner_main,
)
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
    docker_compose_to_singularity_compose,
)
from ursa.integrations.harbor_validation import (  # noqa: E402
    discover_harbor_tasks,
    validate_harbor_task,
)


def _config(path: Path) -> Path:
    path.write_text("llm_model:\n  model: gpt-4.1-nano\n")
    return path


def _harbor_task(tmp_path: Path, *, sidecar: bool = False) -> Path:
    task = tmp_path / "example-task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Complete the task.\n")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "Dockerfile").write_text("FROM scratch\n")
    services = "  main:\n    build: .\n"
    if sidecar:
        services += "  database:\n    image: postgres:17\n"
    (task / "tests" / "docker-compose.yaml").write_text(
        f"services:\n{services}"
    )
    (task / "task.toml").write_text(
        """schema_version = "1.4"

[task]
name = "example/task"
version = "1.0.0"
description = "Example task"

[verifier]
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"

[environment]
network_mode = "public"
"""
    )
    return task


def test_validate_harbor_task_accepts_supported_definitions(tmp_path):
    task = _harbor_task(tmp_path)

    validate_harbor_task(task)

    assert discover_harbor_tasks([tmp_path]) == [task]
    assert discover_harbor_tasks([task / "tests" / "Dockerfile"]) == [task]


def test_validate_harbor_task_rejects_no_network_sidecar(tmp_path):
    task = _harbor_task(tmp_path, sidecar=True)

    with pytest.raises(ValueError, match="sidecar networking"):
        validate_harbor_task(task)


@pytest.fixture(autouse=True)
def _host_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-openai-key")
    monkeypatch.setattr(
        "ursa.integrations.harbor.config_search_paths",
        lambda namespace, _level: [Path(namespace.config)],
    )


def test_usage_reads_current_metrics_schema(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        """{
          "totals": {"llm_total_s": 1.5},
          "costs": {"total_usd": 0.012},
          "llm_events": [
            {"metrics": {"usage_rollup": {
              "input_tokens": 10, "output_tokens": 2
            }}},
            {"metrics": {"usage_rollup": {
              "input_tokens": 20, "output_tokens": 3
            }}}
          ]
        }"""
    )

    assert _usage(metrics) == {
        "n_input_tokens": 30,
        "n_output_tokens": 5,
        "cost_usd": 0.012,
    }


def test_usage_leaves_missing_event_usage_unknown(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        '{"totals": {"llm_total_s": 1.5}, '
        '"llm_events": [{"metrics": {"error": "failed"}}]}'
    )

    assert _usage(metrics) == {
        "n_input_tokens": None,
        "n_output_tokens": None,
        "cost_usd": None,
    }


def test_runner_output_is_teeed_to_harbor_log(tmp_path, capsys):
    log_path = tmp_path / "agent" / "ursa.log"

    with _capture_output(log_path):
        print("agent stdout")
        print("agent stderr", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.out == "agent stdout\n"
    assert captured.err == "agent stderr\n"
    assert log_path.read_text().splitlines() == [
        "agent stdout",
        "agent stderr",
    ]


def test_runner_failure_is_written_to_harbor_log(tmp_path, monkeypatch):
    log_path = tmp_path / "agent" / "ursa.log"
    encoded = base64.urlsafe_b64encode(
        json.dumps({"log_path": str(log_path)}).encode()
    ).decode()

    def fail(_config):
        raise RuntimeError("agent failed")

    monkeypatch.setattr("ursa.integrations.harbor_runner._run", fail)

    with pytest.raises(RuntimeError, match="agent failed"):
        _runner_main(encoded)

    text = log_path.read_text()
    assert "Traceback" in text
    assert "RuntimeError: agent failed" in text


def test_runner_console_entrypoint_reads_encoded_argument(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "agent" / "ursa.log"
    payload = {"log_path": str(log_path), "instruction": "task"}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    received = []
    monkeypatch.setattr("ursa.integrations.harbor_runner._run", received.append)
    monkeypatch.setattr(sys, "argv", ["ursa-harbor-runner", encoded])

    _runner_main()

    assert received == [payload]


def _singularity_env(
    tmp_path,
    monkeypatch,
    builders=("docker",),
    fail=None,
    runtime="singularity",
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    environment.environment_dir = environment_dir
    environment._image_cache_dir = tmp_path / "cache"
    environment.session_id = "trial"
    commands = []

    async def fake_run(*command):
        commands.append(command)
        if fail and fail(command):
            raise RuntimeError("command failed")
        if command[0:3] == (f"/usr/bin/{runtime}", "sif", "list"):
            if Path(command[3]).read_text() == "broken":
                raise RuntimeError("invalid SIF")
        if command[:2] == (f"/usr/bin/{runtime}", "build"):
            Path(command[2]).write_text("sif")

    monkeypatch.setattr(
        "shutil.which",
        lambda command: (
            f"/usr/bin/{command}" if command in (*builders, runtime) else None
        ),
    )
    monkeypatch.setattr(environment, "_run", fake_run)
    return environment, commands


def test_singularity_preflight_checks_only_runtime_presence(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/apptainer" if name == "apptainer" else None,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("preflight must not probe tools"),
    )

    DockerfileSingularityEnvironment.preflight()


def test_singularity_preflight_rejects_missing_runtime(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(SystemExit, match="Apptainer or Singularity"):
        DockerfileSingularityEnvironment.preflight()


def test_singularity_uses_a_shared_default_cache(tmp_path, monkeypatch):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")

    def create(**kwargs):
        return DockerfileSingularityEnvironment(
            environment_dir=environment_dir,
            environment_name="test",
            session_id="trial__env",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=EnvironmentConfig(),
            **kwargs,
        )

    monkeypatch.delenv("URSA_HARBOR_SIF_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))

    default = create()
    configured = tmp_path / "configured-cache"
    monkeypatch.setenv("URSA_HARBOR_SIF_CACHE", str(configured))
    from_environment = create()
    explicit = tmp_path / "explicit-cache"
    overridden = create(singularity_image_cache_dir=explicit)

    assert [
        default._image_cache_dir,
        from_environment._image_cache_dir,
        overridden._image_cache_dir,
    ] == [
        tmp_path / "xdg-cache" / "ursa" / "harbor" / "sif",
        configured,
        explicit,
    ]


def test_singularity_uses_docker_workdir_semantics(tmp_path):
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    environment.environment_dir = tmp_path
    environment.task_env_config = SimpleNamespace(workdir=None)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")

    assert environment._resolve_workdir() == "/"

    dockerfile.write_text("FROM scratch\nWORKDIR /workspace\nWORKDIR project\n")
    assert environment._resolve_workdir() == "/workspace/project"

    dockerfile.write_text(
        "FROM scratch AS builder\nWORKDIR /build\nFROM scratch\nWORKDIR app\n"
    )
    assert environment._resolve_workdir() == "/app"

    environment.task_env_config.workdir = "/task-override"
    assert environment._resolve_workdir() == "/task-override"

    environment.task_env_config.workdir = None
    dockerfile.write_text("FROM scratch\nWORKDIR $APP_DIR\n")
    with pytest.raises(ValueError, match="cannot resolve variables"):
        environment._resolve_workdir()


def test_factory_rejects_unrelated_class(tmp_path):
    with pytest.raises(TypeError, match="BaseAgent"):
        make_harbor_agent(  # type: ignore[arg-type]
            str, _config(tmp_path / "ursa.yaml")
        )


def test_factory_accepts_arbitrary_ursa_subclass(tmp_path):
    class MinimalAgent(BaseAgent):
        def _invoke(self, inputs, **config):
            return inputs

    bound = make_harbor_agent(MinimalAgent, _config(tmp_path / "ursa.yaml"))
    assert bound.__name__ == "MinimalAgentHarborAgent"
    assert bound.name() == "ursa"


def test_runtime_config_uses_default_stack_with_harbor_last(
    tmp_path, monkeypatch
):
    system = tmp_path / "system.yaml"
    user = tmp_path / "user.yaml"
    config_file = tmp_path / "ursa.yaml"
    system.write_text(
        "use_web: true\n"
        "agent_name: system\n"
        "mcp_servers:\n"
        "  tools:\n"
        "    transport: stdio\n"
        "    command: old-command\n"
    )
    user.write_text("agent_name: user\n")
    config_file.write_text(
        "agent_name: file\n"
        "llm_model:\n"
        "  model: file-model\n"
        "  max_completion_tokens: 123\n"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.config_search_paths",
        lambda _namespace, _level: [system, user, config_file],
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-nano",
        config_file=config_file,
        config_only=False,
        mcp_servers=[
            MCPServerConfig(
                name="tools",
                transport="streamable-http",
                url="http://tools:8000/mcp",
            )
        ],
    )

    runtime_config, _ = agent._runtime_config()

    assert runtime_config["use_web"] is True
    assert runtime_config["agent_name"] == "file"
    assert runtime_config["llm_model"]["model"] == "gpt-5.4-nano"
    assert runtime_config["llm_model"]["inference_provider"] == "openai"
    assert runtime_config["llm_model"]["max_completion_tokens"] == 123
    assert runtime_config["mcp_servers"]["tools"]["url"] == (
        "http://tools:8000/mcp"
    )
    assert "command" not in runtime_config["mcp_servers"]["tools"]


def test_runtime_config_uses_ursa_config_path_discovery(tmp_path, monkeypatch):
    system = tmp_path / "system.yaml"
    user = tmp_path / "user.yaml"
    config_file = tmp_path / "ursa.yaml"
    system.write_text("use_web: true\n")
    user.write_text("agent_name: user\n")
    config_file.write_text("agent_name: supplied\n")
    monkeypatch.setattr(
        "ursa.integrations.harbor.config_search_paths",
        config_module.config_search_paths,
    )
    monkeypatch.setattr("ursa.cli.config.system_config_paths", lambda: [system])
    monkeypatch.setattr("ursa.cli.config.user_config_paths", lambda: [user])
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-nano",
        config_file=config_file,
        config_only=False,
    )

    runtime_config, _ = agent._runtime_config()

    assert runtime_config["use_web"] is True
    assert runtime_config["agent_name"] == "supplied"


def test_runtime_config_only_uses_supplied_file_before_harbor(
    tmp_path, monkeypatch
):
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "agent_name: supplied\n"
        "llm_model:\n"
        "  model: supplied-model\n"
        "  max_completion_tokens: 321\n"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.config_search_paths",
        lambda *_args: pytest.fail("config search must be skipped"),
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-nano",
        config_file=config_file,
    )

    runtime_config, _ = agent._runtime_config()

    assert runtime_config["agent_name"] == "supplied"
    assert runtime_config["llm_model"]["model"] == "gpt-5.4-nano"
    assert runtime_config["llm_model"]["max_completion_tokens"] == 321


@pytest.mark.parametrize("use_web", [True, False])
def test_runtime_config_gives_harbor_network_policy_highest_priority(
    tmp_path, use_web
):
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(f"use_web: {str(not use_web).lower()}\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        config_only=True,
    )

    runtime_config, _ = agent._runtime_config(use_web=use_web)

    assert runtime_config["use_web"] is use_web


@pytest.mark.parametrize(
    ("network_mode", "expected"),
    [
        (NetworkMode.PUBLIC, True),
        (NetworkMode.ALLOWLIST, True),
        (NetworkMode.NO_NETWORK, False),
    ],
)
def test_network_policy_controls_web_access(network_mode, expected):
    environment = SimpleNamespace(
        network_policy=NetworkPolicy(network_mode=network_mode)
    )

    assert UrsaHarborAgent._network_use_web(environment) is expected


def test_runner_web_policy_overrides_nested_agent_config():
    class WebAgent:
        def __init__(self, *, use_web=False):
            pass

    class AgentWithoutWeb:
        def __init__(self):
            pass

    class ForwardingWebAgent(WebAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    config = SimpleNamespace(
        agent_config={
            "web": {"use_web": True},
            "agent_without_web": {"custom": "value"},
        }
    )

    assert _agent_config(config, WebAgent, use_web=False)["use_web"] is False
    assert _agent_config(config, ForwardingWebAgent, use_web=False) == {
        "use_web": False
    }
    assert _agent_config(config, AgentWithoutWeb, use_web=False) == {
        "custom": "value"
    }


def test_runtime_config_externalizes_secrets_from_all_layers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SYSTEM_TOKEN", "system-secret")
    monkeypatch.setenv("EMBEDDING_TOKEN", "embedding-secret")
    monkeypatch.setattr(
        "keyring.get_password",
        lambda service, username: (
            "unused-secret"
            if (service, username) == ("ursa", "unused")
            else None
        ),
    )
    system = tmp_path / "system.yaml"
    config_file = tmp_path / "ursa.yaml"
    system.write_text(
        "mcp_servers:\n"
        "  system:\n"
        "    transport: streamable-http\n"
        "    url: http://system.test/mcp\n"
        "    headers:\n"
        "      Authorization:\n"
        "        env: SYSTEM_TOKEN\n"
        "        template: Bearer %s\n"
    )
    config_file.write_text(
        "inference_providers:\n"
        "  unused:\n"
        "    api_key:\n"
        "      keyring: true\n"
        "emb_model:\n"
        "  model: openai:text-embedding-3-small\n"
        "  api_key:\n"
        "    env: EMBEDDING_TOKEN\n"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.config_search_paths",
        lambda _namespace, _level: [system, config_file],
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-nano",
        config_file=config_file,
        config_only=False,
    )

    runtime_config, secret_env = agent._runtime_config()

    assert set(secret_env.values()) == {
        "host-openai-key",
        "system-secret",
        "embedding-secret",
        "unused-secret",
    }
    serialized = json.dumps(runtime_config)
    for original in (
        "OPENAI_API_KEY",
        "SYSTEM_TOKEN",
        "EMBEDDING_TOKEN",
        '"keyring": true',
    ):
        assert original not in serialized
    assert "URSA_HARBOR_SECRET_" in serialized


def test_harbor_model_connection_is_an_ursa_provider_layer(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.test/v1")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
    )

    runtime_config, secret_env = agent._runtime_config()

    assert runtime_config["inference_providers"]["openai"]["base_url"] == (
        "https://proxy.test/v1"
    )
    assert set(secret_env.values()) == {"host-openai-key"}


def test_harbor_model_switch_drops_old_provider_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("OLD_AZURE_KEY", raising=False)
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  ollama:\n"
        "    model_provider: ollama\n"
        "    base_url: http://localhost:11434\n"
        "llm_model:\n"
        "  model: old-model\n"
        "  model_provider: azure_openai\n"
        "  api_key:\n"
        "    env: OLD_AZURE_KEY\n"
        "  ssl_verify: false\n"
        "  azure_deployment: old-deployment\n"
        "  max_completion_tokens: 456\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="ollama/gemma4:latest",
        config_file=config_file,
    )

    runtime_config, secret_env = agent._runtime_config()

    assert "OLD_AZURE_KEY" not in json.dumps(runtime_config)
    assert "api_key" not in runtime_config["llm_model"]
    assert "azure_deployment" not in runtime_config["llm_model"]
    assert runtime_config["llm_model"]["model_provider"] == "ollama"
    assert runtime_config["llm_model"]["max_completion_tokens"] == 456
    monkeypatch.setenv(next(iter(secret_env)), next(iter(secret_env.values())))
    resolved = UrsaConfig.model_validate(runtime_config).resolve()
    assert resolved.llm_model.model_provider == "ollama"


def test_harbor_model_accepts_an_explicit_model_provider(tmp_path):
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  ollama:\n"
        "    base_url: http://localhost:11434\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="ollama/ollama:gemma4:latest",
        config_file=config_file,
    )

    runtime_config, _ = agent._runtime_config()

    assert runtime_config["llm_model"]["model"] == "ollama:gemma4:latest"
    assert runtime_config["llm_model"]["model_provider"] == "ollama"
    assert runtime_config["llm_model"]["inference_provider"] == "ollama"


def test_harbor_model_requires_a_configured_inference_provider(tmp_path):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-test",
        config_file=_config(tmp_path / "ursa.yaml"),
    )

    with pytest.raises(ValueError, match="anthropic.*defined.*merged"):
        agent._runtime_config()


def test_harbor_model_requires_a_backend_for_non_openai_provider(tmp_path):
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  ollama:\n"
        "    base_url: http://localhost:11434\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="ollama/gemma4",
        config_file=config_file,
    )

    with pytest.raises(
        ValueError, match="model_provider:model.*model_provider"
    ):
        agent._runtime_config()


def test_runtime_config_rejects_generic_environment_interpolation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TOKEN", "must-not-enter-runtime-json")
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text("agent_name: ${TOKEN}\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
    )

    with pytest.raises(ValueError, match="agent_name.*explicit.*env"):
        agent._runtime_config()


@pytest.mark.asyncio
async def test_install_uses_uv_and_uploads_one_config(tmp_path, monkeypatch):
    config_file = _config(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        ursa_install_spec="ursa-ai==1.2",
        ursa_extras="image",
        extra_packages=["numpy", "scipy"],
    )
    commands = []
    timeouts = []

    async def fake_exec_as_root(environment, command, **kwargs):
        commands.append(command)
        timeouts.append(kwargs["timeout_sec"])

    async def fake_exec_as_agent(environment, command, **kwargs):
        assert command == "pwd"
        return SimpleNamespace(stdout="/app\n")

    class FakeEnvironment:
        async def upload_file(self, source, destination):
            uploads.append((
                json.loads(source.read_text()),
                destination,
                source.stat().st_mode & 0o777,
            ))

    uploads = []

    monkeypatch.setattr(agent, "exec_as_root", fake_exec_as_root)
    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    await agent.install(FakeEnvironment())

    assert "command -v tar" in commands[0]
    assert 'missing_packages="$missing_packages tar"' in commands[0]
    assert "ca-certificates $missing_packages" in commands[0]
    assert "unknown-linux-musl.tar.gz" in commands[0]
    assert "sha256sum -c" in commands[0]
    assert "case $(uname -m)" in commands[0]
    assert "/opt/uv/uv python install 3.13" in commands[0]
    install_command = commands[1]
    assert "uv tool install --force --python 3.13" in install_command
    assert "UV_TOOL_BIN_DIR=/usr/local/bin" in install_command
    assert "ursa-ai[image]==1.2" in install_command
    assert "--with numpy" in install_command
    assert "--with scipy" in install_command
    assert 'command -v ursa)" = /usr/local/bin/ursa' in install_command
    assert "test -x /usr/local/bin/ursa-harbor-runner" in install_command
    assert timeouts == [600, 900]
    assert len(uploads) == 1
    runtime_config, destination, mode = uploads[0]
    assert destination == "/tmp/ursa-config.json"
    assert mode == 0o600
    assert runtime_config["inference_providers"]["openai"]["api_key"] == {
        "env": "URSA_HARBOR_SECRET_0"
    }
    assert agent._secret_env == {"URSA_HARBOR_SECRET_0": "host-openai-key"}
    assert agent._workspace == "/app"


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout", ["", "relative\n", "/one\n/two\n"])
async def test_install_rejects_invalid_task_working_directory(
    tmp_path, monkeypatch, stdout
):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        ursa_install_spec="ursa-ai",
    )

    async def fake_exec_as_root(*_args, **_kwargs):
        pass

    async def fake_exec_as_agent(*_args, **_kwargs):
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(agent, "exec_as_root", fake_exec_as_root)
    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    with pytest.raises(RuntimeError, match="Invalid task working directory"):
        await agent.install(SimpleNamespace())


@pytest.mark.asyncio
async def test_source_install_does_not_upload_secrets(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        "[project]\nname='test'\nversion='0'\n"
    )
    (source / "module.py").write_text("VALUE = 1\n")
    for name in (".env", ".env.local", "client.key", "credentials.json"):
        (source / name).write_text("secret")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        ursa_install_spec=str(source),
        ursa_extras="image",
    )
    uploaded = []
    commands = []

    async def fake_exec_as_root(environment, command, **kwargs):
        commands.append(command)

    async def fake_exec_as_agent(*args, **kwargs):
        return SimpleNamespace(stdout="/app\n")

    class FakeEnvironment:
        async def upload_dir(self, staged, destination):
            uploaded.extend(
                path.relative_to(staged).as_posix()
                for path in staged.rglob("*")
            )

        async def upload_file(self, source_file, destination):
            pass

    monkeypatch.setattr(agent, "exec_as_root", fake_exec_as_root)
    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)
    await agent.install(FakeEnvironment())

    assert "module.py" in uploaded
    secrets = {".env", ".env.local", "client.key", "credentials.json"}
    assert not secrets & set(uploaded)
    assert any("/tmp/ursa-source[image]" in command for command in commands)


def test_install_spec_path_must_be_a_python_project(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="ursa_install_spec.*Python project"):
        UrsaHarborAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4.1-nano",
            config_file=_config(tmp_path / "ursa.yaml"),
            ursa_install_spec=source,
        )


def test_install_spec_defaults_to_current_checkout(tmp_path):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
    )

    assert agent.ursa_install_spec == Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        (
            {
                "url": "https://example.com/ursa.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            },
            "git+https://example.com/ursa.git@abc123",
        ),
        (
            {"url": "https://example.com/ursa.whl"},
            "https://example.com/ursa.whl",
        ),
    ],
)
def test_default_install_spec_preserves_installed_provenance(
    tmp_path, monkeypatch, provenance, expected
):
    installed_module = (
        tmp_path / "installed" / "site-packages" / "ursa" / "integrations"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.__file__", str(installed_module / "harbor.py")
    )
    distribution = SimpleNamespace(
        version="0.0",
        read_text=lambda name: json.dumps(provenance)
        if name == "direct_url.json"
        else None,
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.importlib.metadata.distribution",
        lambda _name: distribution,
    )

    assert UrsaHarborAgent._default_install_spec() == expected


def test_default_install_spec_uses_installed_local_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='ursa-ai'\n")
    installed_module = (
        tmp_path / "installed" / "site-packages" / "ursa" / "integrations"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.__file__", str(installed_module / "harbor.py")
    )
    distribution = SimpleNamespace(
        version="0.0",
        read_text=lambda _name: json.dumps({"url": source.as_uri()}),
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.importlib.metadata.distribution",
        lambda _name: distribution,
    )

    assert UrsaHarborAgent._default_install_spec() == source


@pytest.mark.parametrize(
    "install_spec",
    [
        "ursa-ai",
        "ursa-ai==1.2",
        "git+https://github.com/harbor-framework/ursa.git@main",
        "https://example.com/ursa.whl",
        "ursa-ai @ git+https://github.com/harbor-framework/ursa.git@main",
    ],
)
def test_install_requirement_is_not_treated_as_a_path(tmp_path, install_spec):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        ursa_install_spec=install_spec,
    )

    assert agent.ursa_install_spec == install_spec


@pytest.mark.parametrize(
    "install_spec",
    [
        "git+https://github.com/harbor-framework/ursa.git@main",
        "https://example.com/ursa.whl",
    ],
)
def test_install_extras_use_a_named_direct_reference(tmp_path, install_spec):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        ursa_install_spec=install_spec,
        ursa_extras="image,harbor",
    )

    assert agent._install_target(install_spec) == (
        f"ursa-ai[image,harbor] @ {install_spec}"
    )


def test_install_extras_extend_an_existing_named_direct_reference(tmp_path):
    install_spec = "ursa-ai @ git+https://example.com/ursa.git@revision"
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        ursa_install_spec=install_spec,
        ursa_extras="image",
        extra_packages="numpy>=1.26,<3",
    )

    assert agent._install_target(install_spec) == (
        "ursa-ai[image] @ git+https://example.com/ursa.git@revision"
    )
    assert agent.extra_packages == ("numpy>=1.26,<3",)


def test_extra_packages_accepts_a_json_array_from_the_cli(tmp_path):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        extra_packages='  ["numpy","scipy"]',
    )

    assert agent.extra_packages == ("numpy", "scipy")


@pytest.mark.parametrize("extra_packages", ['["numpy", 1]', "[invalid"])
def test_extra_packages_rejects_invalid_json_arrays(tmp_path, extra_packages):
    with pytest.raises(ValueError, match="extra_packages"):
        UrsaHarborAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4.1-nano",
            config_file=_config(tmp_path / "ursa.yaml"),
            extra_packages=extra_packages,
        )


@pytest.mark.parametrize(
    "archive_name",
    ["ursa_ai-0.0-py3-none-any.whl", "ursa_ai-0.0.tar.gz", "ursa_ai-0.0.zip"],
)
def test_default_install_spec_uses_installed_local_archive(
    tmp_path, monkeypatch, archive_name
):
    archive = tmp_path / archive_name
    archive.write_bytes(b"wheel")
    installed_module = (
        tmp_path / "installed" / "site-packages" / "ursa" / "integrations"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.__file__", str(installed_module / "harbor.py")
    )
    distribution = SimpleNamespace(
        version="0.0",
        read_text=lambda _name: json.dumps({"url": archive.as_uri()}),
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.importlib.metadata.distribution",
        lambda _name: distribution,
    )

    assert UrsaHarborAgent._default_install_spec() == archive


def test_default_install_spec_rejects_invalid_local_provenance(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "ursa.txt"
    artifact.write_text("not a package")
    installed_module = (
        tmp_path / "installed" / "site-packages" / "ursa" / "integrations"
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.__file__", str(installed_module / "harbor.py")
    )
    distribution = SimpleNamespace(
        version="0.0",
        read_text=lambda _name: json.dumps({"url": artifact.as_uri()}),
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor.importlib.metadata.distribution",
        lambda _name: distribution,
    )

    with pytest.raises(ValueError, match="pass ursa_install_spec"):
        UrsaHarborAgent._default_install_spec()


def test_install_spec_rejects_a_non_package_file(tmp_path):
    artifact = tmp_path / "ursa.txt"
    artifact.write_text("not a package")

    with pytest.raises(ValueError, match="Python project or wheel/sdist"):
        UrsaHarborAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4.1-nano",
            config_file=_config(tmp_path / "ursa.yaml"),
            ursa_install_spec=artifact,
        )


@pytest.mark.asyncio
async def test_install_uploads_a_local_archive(tmp_path, monkeypatch):
    archive = tmp_path / "ursa_ai-0.0-py3-none-any.whl"
    archive.write_bytes(b"wheel")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        ursa_install_spec=archive,
        ursa_extras="harbor",
    )
    commands = []
    uploads = []

    async def fake_exec_as_root(_environment, command, **_kwargs):
        commands.append(command)

    async def fake_exec_as_agent(*_args, **_kwargs):
        return SimpleNamespace(stdout="/app\n")

    class FakeEnvironment:
        async def upload_file(self, source, destination):
            uploads.append((source, destination))

    monkeypatch.setattr(agent, "exec_as_root", fake_exec_as_root)
    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    await agent.install(FakeEnvironment())

    assert uploads[0] == (archive, f"/tmp/{archive.name}")
    assert f"ursa-ai[harbor] @ file:///tmp/{archive.name}" in commands[1]


@pytest.mark.parametrize(
    "install_spec",
    [
        Path("missing"),
        "/missing/ursa-project",
        "./missing-ursa-project",
        "../missing-ursa-project",
        "~/missing-ursa-project",
    ],
)
def test_missing_install_path_fails_before_install(
    tmp_path, monkeypatch, install_spec
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(
        ValueError, match="ursa_install_spec path does not exist"
    ):
        UrsaHarborAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4.1-nano",
            config_file=_config(tmp_path / "ursa.yaml"),
            ursa_install_spec=install_spec,
        )


def test_git_source_staging_respects_ignored_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n")
    (source / "module.py").write_text("tracked\n")
    (source / "credentials.py").write_text("legitimate module\n")
    (source / "new.py").write_text("untracked\n")
    (source / ".env").write_text("accidentally unignored\n")
    (source / "tracked.key").write_text("tracked secret\n")
    (source / ".gitignore").write_text("private-data\n")
    (source / "private-data").write_text("secret\n")
    subprocess.run(["git", "init", "-q", source], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            source,
            "add",
            "pyproject.toml",
            "module.py",
            "credentials.py",
            "tracked.key",
        ],
        check=True,
    )
    staged = tmp_path / "staged"

    UrsaHarborAgent._stage_source(source, staged)

    assert (staged / "module.py").is_file()
    assert (staged / "credentials.py").is_file()
    assert (staged / "new.py").is_file()
    assert not (staged / "private-data").exists()
    assert not (staged / ".env").exists()
    assert not (staged / "tracked.key").exists()


@pytest.mark.parametrize("git_source", [False, True])
def test_source_staging_rejects_symlinks(tmp_path, git_source):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n")
    secret = tmp_path / "host-secret"
    secret.write_text("secret")
    (source / "data").symlink_to(secret)
    if git_source:
        subprocess.run(["git", "init", "-q", source], check=True)
        subprocess.run(
            ["git", "-C", source, "add", "pyproject.toml", "data"],
            check=True,
        )

    with pytest.raises(ValueError, match="must not contain symlinks.*data"):
        UrsaHarborAgent._stage_source(source, tmp_path / "staged")


@pytest.mark.asyncio
async def test_cancelled_run_terminates_container_runner(tmp_path, monkeypatch):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
    )
    agent._remote_config_file = "/tmp/ursa-config.yaml"
    runner_started = asyncio.Event()
    cleanup_commands = []

    async def fake_exec_as_agent(*args, **kwargs):
        runner_started.set()
        await asyncio.Event().wait()

    async def fake_exec_as_root(environment, command, **kwargs):
        cleanup_commands.append(command)

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)
    monkeypatch.setattr(agent, "exec_as_root", fake_exec_as_root)

    task = asyncio.create_task(agent.run("task", object(), object()))
    await runner_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(cleanup_commands) == 1
    assert "kill -TERM" in cleanup_commands[0]
    assert "kill -KILL" in cleanup_commands[0]
    assert "/proc/$1/task/$1/children" in cleanup_commands[0]


@pytest.mark.asyncio
async def test_runner_cleanup_terminates_descendants(tmp_path):
    pid_file = tmp_path / "runner.pid"
    child_file = tmp_path / "child.pid"
    runner = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        f"echo $$ > {pid_file}; sleep 30 & echo $! > {child_file}; wait",
    )
    for _ in range(100):
        if pid_file.is_file() and child_file.is_file():
            break
        await asyncio.sleep(0.01)
    child_pid = int(child_file.read_text())

    cleanup = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        UrsaHarborAgent._terminate_runner_command(str(pid_file)),
    )
    assert await cleanup.wait() == 0
    await asyncio.wait_for(runner.wait(), timeout=3)
    for _ in range(100):
        if not Path(f"/proc/{child_pid}").exists():
            break
        await asyncio.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.asyncio
async def test_run_leaves_trial_timeout_to_harbor(tmp_path, monkeypatch):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
    )
    agent._remote_config_file = "/tmp/ursa-config.yaml"
    agent._secret_env = {"URSA_HARBOR_SECRET_0": "resolved-on-host"}
    observed_timeout = object()
    observed_env = None
    observed_command = None

    async def fake_exec_as_agent(*args, **kwargs):
        nonlocal observed_command, observed_env, observed_timeout
        observed_command = kwargs["command"]
        observed_timeout = kwargs["timeout_sec"]
        observed_env = kwargs["env"]
        return SimpleNamespace(
            return_code=0,
            stdout=(
                'URSA_HARBOR_RESULT={"result": {"answer": "done"}, '
                '"n_input_tokens": 11, "n_output_tokens": 7, '
                '"cost_usd": 0.25}\n'
            ),
        )

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    context = SimpleNamespace()
    await agent.run("task", object(), context)

    assert observed_timeout is None
    assert observed_env["URSA_HARBOR_SECRET_0"] == "resolved-on-host"
    assert observed_command is not None
    assert "exec /usr/local/bin/ursa-harbor-runner" in observed_command
    encoded = shlex.split(observed_command)[-1]
    payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
    assert payload["log_path"] == "/logs/agent/ursa.log"
    assert context.metadata == {"ursa_result": {"answer": "done"}}
    assert context.n_input_tokens == 11
    assert context.n_output_tokens == 7
    assert context.cost_usd == 0.25


@pytest.mark.asyncio
async def test_run_reuploads_config_when_phase_network_policy_changes(
    tmp_path, monkeypatch
):
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "use_web: true\nagent_config:\n  execute:\n    use_web: true\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        config_only=True,
    )
    agent._remote_config_file = "/tmp/ursa-config.json"
    agent._runtime_use_web = True
    uploaded = []
    payloads = []

    class FakeEnvironment:
        network_policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)

        async def upload_file(self, source, destination):
            assert destination == "/tmp/ursa-config.json"
            uploaded.append(json.loads(source.read_text()))

    environment = FakeEnvironment()

    async def fake_exec_as_agent(*args, **kwargs):
        encoded = shlex.split(kwargs["command"])[-1]
        payloads.append(json.loads(base64.urlsafe_b64decode(encoded).decode()))
        return SimpleNamespace(
            return_code=0,
            stdout='URSA_HARBOR_RESULT={"result": null}\n',
        )

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    await agent.run("task", environment, SimpleNamespace())
    environment.network_policy = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["example.com"],
    )
    await agent.run("task", environment, SimpleNamespace())

    assert [config["use_web"] for config in uploaded] == [False, True]
    assert [payload["use_web"] for payload in payloads] == [False, True]


@pytest.mark.asyncio
async def test_run_passes_provider_extra_env_only_to_runner(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session-token")
    monkeypatch.setenv("AWS_REGION", "us-test-1")
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  amazon-bedrock:\n"
        "    model_provider: bedrock_converse\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="amazon-bedrock/test-model",
        config_file=config_file,
    )
    runtime_config, agent._secret_env = agent._runtime_config()
    agent._remote_config_file = "/tmp/ursa-config.json"
    observed_env = None

    async def fake_exec_as_agent(*args, **kwargs):
        nonlocal observed_env
        observed_env = kwargs["env"]
        return SimpleNamespace(
            return_code=0,
            stdout='URSA_HARBOR_RESULT={"result": null}\n',
        )

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    await agent.run("task", object(), SimpleNamespace())

    assert observed_env is not None
    assert (
        "api_key" not in runtime_config["inference_providers"]["amazon-bedrock"]
    )
    assert observed_env["AWS_ACCESS_KEY_ID"] == "access-key"
    assert observed_env["AWS_SECRET_ACCESS_KEY"] == "secret-key"
    assert observed_env["AWS_SESSION_TOKEN"] == "session-token"
    assert observed_env["AWS_REGION"] == "us-test-1"
    assert "OPENAI_API_KEY" not in observed_env
    for name, value in agent._secret_env.items():
        monkeypatch.setenv(name, value)
    resolved = UrsaConfig.model_validate(runtime_config).resolve()
    assert resolved.llm_model.model_provider == "bedrock_converse"


def test_harbor_preserves_a_configured_custom_model_provider(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUSTOM_TOKEN", "credential")
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  custom:\n"
        "    model_provider: openai\n"
        "    base_url: https://models.test/v1\n"
        "    api_key:\n"
        "      env: CUSTOM_TOKEN\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="custom/test-model",
        config_file=config_file,
    )

    runtime_config, secret_env = agent._runtime_config()

    for name, value in secret_env.items():
        monkeypatch.setenv(name, value)
    resolved = UrsaConfig.model_validate(runtime_config).resolve()
    assert resolved.llm_model.model_provider == "openai"
    assert resolved.llm_model.base_url == "https://models.test/v1"


@pytest.mark.asyncio
async def test_run_reports_stderr_when_runner_has_no_stdout(
    tmp_path, monkeypatch
):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
    )
    agent._remote_config_file = "/tmp/ursa-config.yaml"

    async def fake_exec_as_agent(*args, **kwargs):
        return SimpleNamespace(stdout=None, stderr="runner failed")

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    with pytest.raises(RuntimeError, match="runner failed"):
        await agent.run("task", object(), SimpleNamespace())


def test_harbor_mcp_servers_convert_to_ursa_mapping(tmp_path):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
        mcp_servers=[
            MCPServerConfig(
                name="tools",
                transport="streamable-http",
                url="http://tools:8000/mcp",
            )
        ],
    )

    assert agent._mcp_config() == {
        "tools": {
            "transport": "streamable-http",
            "url": "http://tools:8000/mcp",
            "args": [],
        }
    }


@pytest.mark.asyncio
async def test_mcp_servers_fail_loudly_for_agent_without_tools():
    class AgentWithoutTools:
        pass

    with pytest.raises(TypeError, match="cannot use"):
        await _attach_mcp_tools(
            AgentWithoutTools(),
            {"tools": {"transport": "stdio", "command": "server"}},
        )


@pytest.mark.asyncio
async def test_mcp_servers_attach_to_tool_capable_agent(monkeypatch):
    expected_client = object()
    received = []

    class ToolAgent(AgentWithTools):
        async def add_mcp_tools(self, client, tool_name=None):
            received.append(client)
            return {}

    monkeypatch.setattr(
        "ursa.util.mcp.start_mcp_client", lambda servers: expected_client
    )

    await _attach_mcp_tools(
        ToolAgent.__new__(ToolAgent),
        {"tools": {"transport": "stdio", "command": "server"}},
    )

    assert received == [expected_client]


def test_checkpoint_close_flushes_an_integral_database(tmp_path):
    destination = tmp_path / "checkpointer.db"
    connection = sqlite3.connect(destination)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE checkpoints (value TEXT)")
    connection.execute("INSERT INTO checkpoints VALUES ('saved')")

    _close_checkpoint(SimpleNamespace(conn=connection))

    with sqlite3.connect(destination) as database:
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert database.execute("SELECT value FROM checkpoints").fetchone() == (
            "saved",
        )


def test_checkpoint_close_is_attempted_when_flush_fails():
    class BrokenConnection:
        closed = False

        def commit(self):
            raise sqlite3.Error("flush failed")

        def close(self):
            self.closed = True

    connection = BrokenConnection()

    with pytest.raises(sqlite3.Error, match="flush failed"):
        _close_checkpoint(SimpleNamespace(conn=connection))

    assert connection.closed


def test_runner_orchestrates_agent_and_artifacts(tmp_path, monkeypatch, capsys):
    artifacts = tmp_path / "artifacts"
    checkpoint_path = artifacts / "ursa" / "checkpointer.db"
    checkpoint_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(checkpoint_path)
    connection.execute("CREATE TABLE checkpoints (value TEXT)")
    checkpointer = SimpleNamespace(conn=connection)
    constructed = {}
    events = []

    class FakeAgent(BaseAgent):
        def __init__(self, *, use_web=True, **kwargs):
            constructed.update(kwargs)
            constructed["use_web"] = use_web
            self.checkpointer = kwargs["checkpointer"]

        def _build_graph(self):
            pass

        def format_result(self, output):
            return {"answer": output}

    def invoke(_self, instruction, **kwargs):
        assert instruction == "solve it"
        assert kwargs["save_json"] is True
        Path(kwargs["metrics_path"]).write_text(
            '{"totals": {"input_tokens": 3, "output_tokens": 2}}'
        )
        _self.checkpointer.conn.execute(
            "INSERT INTO checkpoints VALUES ('persisted')"
        )
        return "agent output"

    monkeypatch.setattr(FakeAgent, "invoke", invoke)

    runtime_config = SimpleNamespace(
        workspace=None,
        resolve=lambda: runtime_config,
        llm_model=SimpleNamespace(init_chat_model=lambda: "llm"),
        agent_name="runner",
        group="benchmark",
        thread_id="thread",
        rag_tools=None,
        emb_model=SimpleNamespace(init_embedding=lambda: "embedding"),
        agent_config={"fake": {"use_web": True}},
        mcp_servers={},
    )
    monkeypatch.setattr(
        "ursa.integrations.harbor_runner._import_symbol",
        lambda path: FakeAgent
        if path == "example:FakeAgent"
        else pytest.fail(f"unexpected import path: {path}"),
    )

    def load_config(path):
        assert path == config_file
        return {"config": "loaded"}

    def validate_config(_cls, data):
        assert data == {"config": "loaded"}
        return runtime_config

    def make_checkpointer(path, *, db_dir):
        assert path == artifacts
        assert db_dir == "ursa"
        return checkpointer

    monkeypatch.setattr("ursa.cli.config.load_config_file", load_config)
    monkeypatch.setattr(
        UrsaConfig,
        "model_validate",
        classmethod(validate_config),
    )
    monkeypatch.setattr(
        "ursa.util.Checkpointer.from_workspace",
        make_checkpointer,
    )
    monkeypatch.setattr(
        "ursa.util.events.configure_event_logging",
        lambda *, rich: events.append(rich),
    )
    config_file = tmp_path / "ursa.json"
    config_file.write_text("{}")
    metrics = tmp_path / "logs" / "metrics.json"

    _runner_run({
        "agent_import_path": "example:FakeAgent",
        "config_file": str(config_file),
        "workspace": str(tmp_path / "workspace"),
        "metrics_path": str(metrics),
        "artifacts_dir": str(artifacts),
        "instruction": "solve it",
        "use_web": False,
    })

    payload = json.loads(
        capsys.readouterr().out.removeprefix("URSA_HARBOR_RESULT=")
    )
    assert payload == {
        "result": {"answer": "agent output"},
        "n_input_tokens": 3,
        "n_output_tokens": 2,
        "cost_usd": None,
    }
    assert constructed["llm"] == "llm"
    assert constructed["workspace"] == tmp_path / "workspace"
    assert constructed["checkpointer"] is checkpointer
    assert constructed["use_web"] is False
    assert constructed["agent_name"] == "runner"
    assert constructed["group"] == "benchmark"
    assert constructed["thread_id"] == "thread"
    assert constructed["rag_tools"] is None
    assert constructed["rag_tool_embedding"] == "embedding"
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    with sqlite3.connect(checkpoint_path) as persisted:
        assert persisted.execute(
            "SELECT value FROM checkpoints"
        ).fetchall() == [("persisted",)]
    assert events == [False]


def test_runner_preserves_agent_failure_when_checkpoint_close_fails(
    tmp_path, monkeypatch, capsys
):
    class BrokenConnection:
        closed = False

        def commit(self):
            raise sqlite3.Error("checkpoint close failed")

        def close(self):
            self.closed = True

    connection = BrokenConnection()
    checkpointer = SimpleNamespace(conn=connection)

    class FailingAgent(BaseAgent):
        def __init__(self, **kwargs):
            self.checkpointer = kwargs["checkpointer"]

        def _build_graph(self):
            pass

        def format_result(self, output):
            return output

    def invoke(_self, _instruction, **_kwargs):
        raise RuntimeError("agent failed")

    monkeypatch.setattr(FailingAgent, "invoke", invoke)

    runtime_config = SimpleNamespace(
        workspace=None,
        resolve=lambda: runtime_config,
        llm_model=SimpleNamespace(init_chat_model=lambda: "llm"),
        agent_name=None,
        group="benchmark",
        thread_id="thread",
        rag_tools=None,
        emb_model=None,
        agent_config={},
        mcp_servers={},
    )
    config_file = tmp_path / "ursa.json"
    config_file.write_text("{}")
    monkeypatch.setattr(
        "ursa.integrations.harbor_runner._import_symbol",
        lambda _path: FailingAgent,
    )
    monkeypatch.setattr(
        "ursa.cli.config.load_config_file", lambda _path: {"config": "loaded"}
    )
    monkeypatch.setattr(
        UrsaConfig,
        "model_validate",
        classmethod(lambda _cls, _data: runtime_config),
    )
    monkeypatch.setattr(
        "ursa.util.Checkpointer.from_workspace",
        lambda _path, *, db_dir: checkpointer,
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        _runner_run({
            "agent_import_path": "example:FailingAgent",
            "config_file": str(config_file),
            "workspace": str(tmp_path / "workspace"),
            "metrics_path": str(tmp_path / "logs" / "metrics.json"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "instruction": "solve it",
        })

    assert connection.closed
    assert "checkpoint close failed" in capsys.readouterr().err


@pytest.mark.parametrize("imported", [object(), object])
def test_runner_rejects_non_ursa_agent(imported, monkeypatch):
    monkeypatch.setattr(
        "ursa.integrations.harbor_runner._import_symbol",
        lambda _path: imported,
    )

    with pytest.raises(TypeError, match="URSA BaseAgent subclass"):
        _runner_run({"agent_import_path": "example:Invalid"})


@pytest.mark.asyncio
async def test_singularity_builds_dockerfile_on_demand(tmp_path, monkeypatch):
    environment, commands = _singularity_env(tmp_path, monkeypatch)

    result = await environment._build_dockerfile_sif(force_build=False)

    assert result.is_file()
    assert commands[0][1] == "build"
    assert commands[1][1] == "save"
    assert commands[2][0:2] == ("/usr/bin/singularity", "build")
    assert commands[2][3].startswith("docker-archive://")

    build_count = sum(
        command[0:2] == ("/usr/bin/docker", "build") for command in commands
    )
    assert await environment._build_dockerfile_sif(False) == result
    assert (
        sum(
            command[0:2] == ("/usr/bin/docker", "build") for command in commands
        )
        == build_count
    )
    assert await environment._build_dockerfile_sif(True) == result
    assert (
        sum(
            command[0:2] == ("/usr/bin/docker", "build") for command in commands
        )
        == build_count + 1
    )


@pytest.mark.asyncio
async def test_singularity_cache_hash_honors_dockerignore(
    tmp_path, monkeypatch
):
    environment, _ = _singularity_env(tmp_path, monkeypatch)
    ignored = environment.environment_dir / "generated.log"
    included = environment.environment_dir / "input.txt"
    (environment.environment_dir / ".dockerignore").write_text("*.log\n")
    ignored.write_text("first")
    included.write_text("first")
    original = await environment._dockerfile_cache_path()

    ignored.write_text("second")
    assert await environment._dockerfile_cache_path() == original

    included.write_text("second")
    assert await environment._dockerfile_cache_path() != original

    included.write_text("first")
    assert (
        await environment._dockerfile_cache_path(build_args=("MODE=test",))
        != original
    )
    assert (
        await environment._dockerfile_cache_path(target="runtime") != original
    )


@pytest.mark.parametrize("invalid_content", [b"", b"broken"])
@pytest.mark.asyncio
async def test_singularity_rebuilds_invalid_cache(
    tmp_path, monkeypatch, invalid_content
):
    environment, commands = _singularity_env(tmp_path, monkeypatch)
    result = await environment._build_dockerfile_sif(False)
    build_count = sum(command[1] == "build" for command in commands)
    result.write_bytes(invalid_content)

    await environment._build_dockerfile_sif(False)

    assert sum(command[1] == "build" for command in commands) > build_count


@pytest.mark.asyncio
async def test_singularity_falls_back_when_podman_export_fails(
    tmp_path, monkeypatch
):
    environment, commands = _singularity_env(
        tmp_path,
        monkeypatch,
        builders=("podman", "docker"),
        fail=lambda command: command[0:2] == ("/usr/bin/podman", "save"),
    )

    await environment._build_dockerfile_sif(force_build=True)

    builds = [command for command in commands if command[1] == "build"]
    assert builds[0][0:3] == ("/usr/bin/podman", "build", "--pull")
    assert builds[1][0:3] == ("/usr/bin/docker", "build", "--pull")
    assert any(
        command[0:2] == ("/usr/bin/podman", "image") for command in commands
    )


@pytest.mark.asyncio
async def test_singularity_builds_with_buildah(tmp_path, monkeypatch):
    environment, commands = _singularity_env(
        tmp_path, monkeypatch, builders=("buildah",)
    )

    await environment._build_dockerfile_sif(force_build=True)

    assert commands[0][0:3] == ("/usr/bin/buildah", "build", "--pull")
    assert commands[1][0:2] == ("/usr/bin/buildah", "push")
    assert commands[1][3].startswith("docker-archive:")
    assert commands[-1][0:3] == ("/usr/bin/buildah", "rmi", "--force")


@pytest.mark.asyncio
async def test_singularity_build_requires_an_oci_builder(tmp_path, monkeypatch):
    environment, _ = _singularity_env(tmp_path, monkeypatch, builders=())

    with pytest.raises(
        RuntimeError, match="requires buildah, podman, or docker"
    ):
        await environment._build_dockerfile_sif(force_build=False)


@pytest.mark.asyncio
async def test_singularity_build_reports_all_builder_failures_and_cleans_tags(
    tmp_path, monkeypatch
):
    environment, commands = _singularity_env(
        tmp_path,
        monkeypatch,
        builders=("buildah", "docker"),
        fail=lambda command: command[1] == "build"
        and command[0] != "/usr/bin/singularity",
    )

    with pytest.raises(RuntimeError, match="No container builder succeeded"):
        await environment._build_dockerfile_sif(force_build=False)

    assert any(command[1:3] == ("rmi", "--force") for command in commands)
    assert any(
        command[1:4] == ("image", "rm", "--force") for command in commands
    )


@pytest.mark.asyncio
async def test_concurrent_builds_use_private_builder_tags(
    tmp_path, monkeypatch
):
    first, _ = _singularity_env(
        tmp_path / "first", monkeypatch, builders=("buildah",)
    )
    second, _ = _singularity_env(
        tmp_path / "second", monkeypatch, builders=("buildah",)
    )
    commands = []
    images = set()
    warnings = []
    pushes = 0
    both_pushed = asyncio.Event()

    async def fake_run(*command):
        nonlocal pushes
        commands.append(command)
        if command[0:2] == ("/usr/bin/buildah", "build"):
            tag = command[command.index("--tag") + 1]
            images.add(tag)
        elif command[0:2] == ("/usr/bin/buildah", "push"):
            tag = command[2]
            if tag not in images:
                raise RuntimeError("image not known")
            pushes += 1
            if pushes == 2:
                both_pushed.set()
            await both_pushed.wait()
        elif command[0:3] == ("/usr/bin/buildah", "rmi", "--force"):
            tag = command[3]
            if tag not in images:
                raise RuntimeError("image not known")
            images.remove(tag)
        elif command[0:2] == ("/usr/bin/singularity", "build"):
            Path(command[2]).write_text("sif")

    logger = SimpleNamespace(
        warning=lambda *args: warnings.append(args),
    )
    first._run = fake_run
    first.logger = logger
    second._run = fake_run
    second.logger = logger

    await asyncio.gather(
        first._build_dockerfile_sif(force_build=False),
        second._build_dockerfile_sif(force_build=False),
    )

    build_tags = [
        command[command.index("--tag") + 1]
        for command in commands
        if command[0:2] == ("/usr/bin/buildah", "build")
    ]
    pushed_tags = {
        command[2]
        for command in commands
        if command[0:2] == ("/usr/bin/buildah", "push")
    }
    removed_tags = {
        command[3]
        for command in commands
        if command[0:3] == ("/usr/bin/buildah", "rmi", "--force")
    }
    assert len(build_tags) == len(set(build_tags)) == 2
    assert pushed_tags == removed_tags == set(build_tags)
    assert not images
    assert not warnings


@pytest.mark.asyncio
async def test_concurrent_builds_share_one_cached_sif(tmp_path, monkeypatch):
    first, _ = _singularity_env(
        tmp_path / "first", monkeypatch, builders=("buildah",)
    )
    second, _ = _singularity_env(
        tmp_path / "second", monkeypatch, builders=("buildah",)
    )
    shared_cache = tmp_path / "shared-cache"
    first._image_cache_dir = shared_cache
    second._image_cache_dir = shared_cache
    commands = []
    build_started = asyncio.Event()
    release_build = asyncio.Event()

    async def fake_run(*command):
        commands.append(command)
        if command[0:2] == ("/usr/bin/buildah", "build"):
            build_started.set()
            await release_build.wait()
        elif command[0:2] == ("/usr/bin/singularity", "build"):
            Path(command[2]).write_text("sif")

    first._run = fake_run
    second._run = fake_run
    first_build = asyncio.create_task(first._build_dockerfile_sif(False))
    await build_started.wait()
    second_build = asyncio.create_task(second._build_dockerfile_sif(False))
    release_build.set()

    first_result, second_result = await asyncio.gather(
        first_build, second_build
    )

    builder_commands = [
        command
        for command in commands
        if command[0:2] == ("/usr/bin/buildah", "build")
    ]
    assert first_result == second_result
    assert len(builder_commands) == 1


@pytest.mark.asyncio
async def test_apptainer_only_installation_is_supported(tmp_path, monkeypatch):
    environment, commands = _singularity_env(
        tmp_path,
        monkeypatch,
        builders=("buildah",),
        runtime="apptainer",
    )

    await environment._build_dockerfile_sif(force_build=True)

    assert any(
        command[0:2] == ("/usr/bin/apptainer", "build") for command in commands
    )


def _compose_environment(
    tmp_path,
    monkeypatch,
    compose,
    *,
    extra_compose=None,
    task_env=None,
    runtime="singularity",
    mounts=None,
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    (environment_dir / "docker-compose.yaml").write_text(compose)
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            f"/usr/bin/{name}"
            if name in {runtime, "singularity-compose", "buildah"}
            else None
        ),
    )
    return DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="test",
        session_id="trial__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(env=task_env or {}),
        extra_docker_compose=extra_compose,
        mounts=mounts,
    )


@pytest.mark.asyncio
async def test_docker_compose_conversion_is_file_to_file(tmp_path):
    source = tmp_path / "docker-compose.yaml"
    source.write_text("services: {main: {image: busybox:latest}}\n")
    destination = tmp_path / "project" / "singularity-compose.yml"
    staging = tmp_path / "staging"
    staging.mkdir()

    async def resolve_image(name, service):
        assert name == "main"
        assert service["image"] == "busybox:latest"
        return "docker://busybox:latest"

    instances = await docker_compose_to_singularity_compose(
        source,
        destination,
        identity="test",
        image_resolver=resolve_image,
        staging_dir=staging,
    )

    generated = yaml.safe_load(destination.read_text())
    main = generated["instances"][next(iter(generated["instances"]))]
    assert main["image"] == "docker://busybox:latest"
    assert main["volumes"] == [f"{staging}:/staging"]
    assert main["start"]["options"] == ["fakeroot", "containall", "no-home"]
    assert instances["main"][0].endswith("1")


@pytest.mark.asyncio
async def test_docker_compose_conversion_can_disable_fakeroot(tmp_path):
    source = tmp_path / "docker-compose.yaml"
    source.write_text("services: {main: {image: busybox:latest}}\n")
    destination = tmp_path / "project" / "singularity-compose.yml"
    staging = tmp_path / "staging"
    staging.mkdir()

    async def resolve_image(_name, _service):
        return "docker://busybox:latest"

    await docker_compose_to_singularity_compose(
        source,
        destination,
        identity="test",
        image_resolver=resolve_image,
        staging_dir=staging,
        fakeroot=False,
    )

    generated = yaml.safe_load(destination.read_text())
    main = generated["instances"][next(iter(generated["instances"]))]
    assert main["start"]["options"] == ["containall", "no-home"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_list", [False, True])
async def test_docker_compose_conversion_reads_env_files(tmp_path, use_list):
    env_file = tmp_path / "task.env"
    env_file.write_text("FROM_FILE=loaded\nOVERRIDE=old\n")
    env_files = (
        [
            "./task.env",
            {"path": "./missing.env", "required": False},
        ]
        if use_list
        else "./task.env"
    )
    source = tmp_path / "docker-compose.yaml"
    source.write_text(
        yaml.safe_dump({
            "services": {
                "main": {
                    "image": "busybox:latest",
                    "env_file": env_files,
                    "environment": {"OVERRIDE": "new"},
                }
            }
        })
    )
    destination = tmp_path / "project" / "singularity-compose.yml"
    staging = tmp_path / "staging"
    staging.mkdir()

    async def resolve_image(_name, _service):
        return "docker://busybox:latest"

    await docker_compose_to_singularity_compose(
        source,
        destination,
        identity="test",
        image_resolver=resolve_image,
        staging_dir=staging,
    )

    generated = yaml.safe_load(destination.read_text())
    main = generated["instances"][next(iter(generated["instances"]))]
    mounted_env = next(
        Path(volume.split(":", 1)[0])
        for volume in main["volumes"]
        if volume.endswith(":/.singularity.d/env/91-ursa-compose.sh")
    ).read_text()
    assert "export FROM_FILE=loaded" in mounted_env
    assert "export OVERRIDE=new" in mounted_env


def test_singularity_compose_merges_overlays_after_normalizing_paths(
    tmp_path, monkeypatch
):
    overlay_dir = tmp_path / "overlay"
    overlay_dir.mkdir()
    overlay = overlay_dir / "compose.yaml"
    overlay.write_text("services: {main: {environment: {VALUE: set}}}\n")

    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {build: .}}\n",
        extra_compose=[overlay],
    )

    assert environment._load_compose_config()["services"]["main"] == {
        "build": str((tmp_path / "environment").resolve()),
        "environment": {"VALUE": "set"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["apptainer", "singularity"])
async def test_singularity_compose_uses_selected_runtime_shim(
    tmp_path, monkeypatch, runtime
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {}}\n",
        runtime=runtime,
    )
    environment._sif_path = tmp_path / "main.sif"
    environment._sif_path.write_text("main")
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    calls = []

    async def run(*command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(environment, "_run", run)

    try:
        await environment._prepare_compose_project(force_build=False)
        alternate = "singularity" if runtime == "apptainer" else "apptainer"
        monkeypatch.setattr(
            "shutil.which",
            lambda name: (
                "/usr/bin/singularity-compose"
                if name == "singularity-compose"
                else f"/usr/bin/{alternate}"
                if name == alternate
                else None
            ),
        )
        await environment._run_compose("up")

        runtime_path = f"/usr/bin/{runtime}"
        runtime_bin = environment._compose_project_dir / "bin"
        assert environment._instance_runtime() == runtime_path
        assert (runtime_bin / "singularity").readlink() == Path(runtime_path)
        assert calls[0][1]["env"]["PATH"].split(os.pathsep)[0] == str(
            runtime_bin
        )
    finally:
        environment._cleanup_compose_project()


@pytest.mark.asyncio
async def test_singularity_compose_dependency_invokes_runtime_shim(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {}}\n",
        runtime="apptainer",
    )
    environment._sif_path = tmp_path / "main.sif"
    environment._sif_path.write_text("main")
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()

    invocation_log = tmp_path / "runtime.log"
    fake_runtime = tmp_path / "apptainer"
    fake_runtime.write_text(
        "#!/bin/sh\n"
        'printf \'%s|%s\\n\' "$0" "$*" >> "$URSA_FAKE_RUNTIME_LOG"\n'
        'case " $* " in\n'
        "  *\" --version \"*) echo 'apptainer version test'; exit 0 ;;\n"
        '  *" instance list --json "*) '
        "echo '{\"instances\": []}'; exit 0 ;;\n"
        "esac\n"
        "exit 97\n"
    )
    fake_runtime.chmod(0o700)
    compose_executable = str(
        Path(sys.executable).with_name("singularity-compose")
    )

    monkeypatch.setenv("URSA_FAKE_RUNTIME_LOG", str(invocation_log))
    monkeypatch.setattr(
        "shutil.which",
        lambda name: {
            "apptainer": str(fake_runtime),
            "singularity-compose": compose_executable,
        }.get(name),
    )

    try:
        await environment._prepare_compose_project(force_build=False)
        await environment._run_compose("ps")

        shim = environment._compose_project_dir / "bin" / "singularity"
        invocations = invocation_log.read_text().splitlines()
        assert invocations
        assert all(line.split("|", 1)[0] == str(shim) for line in invocations)
        assert any("|--version" in line for line in invocations)
        assert any("instance list --json" in line for line in invocations)
    finally:
        environment._cleanup_compose_project()


@pytest.mark.asyncio
async def test_singularity_compose_reports_missing_executable_after_prepare(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {}}\n",
    )
    environment._compose_project_dir = tmp_path / "project"
    environment._compose_project_dir.mkdir()
    environment._compose_file = environment._compose_project_dir / "compose.yml"
    environment._compose_file.write_text("instances: {}\n")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="require singularity-compose"):
        await environment._run_compose("up")


@pytest.mark.asyncio
async def test_singularity_file_converter_reads_current_compose_source(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {image: busybox:latest}}\n",
    )
    compose_path = environment.environment_dir / "docker-compose.yaml"
    assert environment._load_compose_config()["services"]["main"]["image"] == (
        "busybox:latest"
    )
    compose_path.write_text("services: {main: {image: alpine:latest}}\n")
    assert environment._load_compose_config()["services"]["main"]["image"] == (
        "alpine:latest"
    )
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()

    try:
        assert await environment._build_main_sif(False) is None
        await environment._prepare_compose_project(force_build=False)
        generated = yaml.safe_load(environment._compose_file.read_text())
        main = generated["instances"][environment._compose_key("main")]
        assert main["image"] == "docker://alpine:latest"
    finally:
        environment._cleanup_compose_project()


@pytest.mark.asyncio
async def test_singularity_compose_translates_supported_services(
    tmp_path, monkeypatch
):
    sidecar = tmp_path / "environment" / "api"
    sidecar.mkdir(parents=True)
    (sidecar / "Dockerfile").write_text("FROM scratch\n")
    monkeypatch.setenv("TASK_TOKEN", "host-token")
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        """
services:
  main:
    depends_on: [api]
    environment:
      MAIN_VALUE: compose
  api:
    build:
      context: ./api
      args:
        BUILD_VALUE: example
      target: runtime
    command: [python, server.py]
    environment:
      API_VALUE: sidecar
    expose: [8000]
    ports: ["18000:8000"]
    deploy:
      replicas: 2
""",
        task_env={"TASK_TOKEN": "${TASK_TOKEN}"},
    )
    environment._sif_path = tmp_path / "main.sif"
    environment._sif_path.write_text("main")
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    builds = []

    async def build(_force, **kwargs):
        builds.append(kwargs)
        result = tmp_path / "api.sif"
        result.write_text("api")
        return result

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)

    await environment._prepare_compose_project(force_build=False)

    generated = yaml.safe_load(environment._compose_file.read_text())
    schema_check = subprocess.run(
        [
            str(Path(sys.executable).with_name("singularity-compose")),
            "--file",
            str(environment._compose_file),
            "check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert schema_check.returncode == 0, schema_check.stderr
    main_key = environment._compose_key("main")
    api_key = environment._compose_key("api")
    main = generated["instances"][main_key]
    api = generated["instances"][api_key]
    assert main["depends_on"] == [api_key]
    assert api["image"] == str(tmp_path / "api.sif")
    assert api["start"]["args"] == "python server.py"
    assert api["ports"] == ["18000:8000"]
    assert api["deploy"] == {"replicas": 2}
    assert environment._compose_instances["api"] == [
        f"{api_key}1",
        f"{api_key}2",
    ]
    assert builds == [
        {
            "dockerfile_path": sidecar / "Dockerfile",
            "context_dir": sidecar,
            "build_args": ("BUILD_VALUE=example",),
            "target": "runtime",
        }
    ]
    main_env = next(
        Path(volume.split(":", 1)[0])
        for volume in main["volumes"]
        if volume.endswith(":/.singularity.d/env/91-ursa-compose.sh")
    ).read_text()
    assert "MAIN_VALUE=compose" in main_env
    assert "TASK_TOKEN=host-token" in main_env


def test_singularity_compose_accepts_list_build_args(tmp_path, monkeypatch):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {build: {args: [BUILD_VALUE=example]}}}\n",
    )
    build = environment._load_compose_config()["services"]["main"]["build"]

    assert environment._compose_build_inputs(build)[2] == (
        "BUILD_VALUE=example",
    )


@pytest.mark.asyncio
async def test_singularity_compose_main_build_can_change_context(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("VERIFY", raising=False)
    task_dir = tmp_path / "task"
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    environment_dir.mkdir(parents=True)
    tests_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM scratch\nWORKDIR /default\n"
    )
    verifier_dockerfile = tests_dir / "Dockerfile"
    verifier_dockerfile.write_text(
        "FROM scratch\nCOPY tests/verify.py /opt/verify.py\n"
        "WORKDIR /workspace\n"
    )
    (tests_dir / "verify.py").write_text("print('ok')\n")
    compose_path = tests_dir / "docker-compose.yaml"
    compose_path.write_text(
        """
services:
  main:
    build:
      context: ..
      dockerfile: tests/Dockerfile
      args:
        VERIFY:
"""
    )
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            f"/usr/bin/{name}"
            if name in {"singularity", "singularity-compose", "buildah"}
            else None
        ),
    )

    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="test",
        session_id="trial__verifier",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        extra_docker_compose=[compose_path],
    )
    monkeypatch.setenv("VERIFY", "enabled")
    builds = []

    async def build(force_build, **kwargs):
        builds.append((force_build, kwargs))
        return tmp_path / "main.sif"

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)

    assert await environment._build_main_sif(False) == tmp_path / "main.sif"
    assert environment._workdir == "/workspace"
    assert builds == [
        (
            False,
            {
                "dockerfile_path": verifier_dockerfile,
                "context_dir": task_dir,
                "build_args": ("VERIFY=enabled",),
                "target": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_singularity_compose_main_can_use_prebuilt_image(
    tmp_path, monkeypatch
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "docker-compose.yaml").write_text(
        """
services:
  main:
    image: ubuntu:24.04
"""
    )
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            f"/usr/bin/{name}"
            if name in {"singularity", "singularity-compose", "buildah"}
            else None
        ),
    )

    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="test",
        session_id="trial__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
    )
    main = environment._load_compose_config()["services"]["main"]

    assert await environment._build_main_sif(False) is None
    assert await environment._compose_image("main", main, False) == (
        "docker://ubuntu:24.04"
    )
    assert environment._workdir == "/"


def test_singularity_compose_image_ignores_unrelated_dockerfile_workdir(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {image: ubuntu:24.04}}\n",
    )
    (environment.environment_dir / "Dockerfile").write_text(
        "FROM scratch\nWORKDIR /not-the-compose-workdir\n"
    )

    assert environment._resolve_workdir() == "/"


def test_singularity_compose_rejects_unsupported_service_fields(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="does not support.*healthcheck"):
        _compose_environment(
            tmp_path,
            monkeypatch,
            """
services:
  main:
    healthcheck:
      test: [CMD, true]
""",
        )


def test_singularity_compose_rejects_depends_on_conditions(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="only service_started"):
        _compose_environment(
            tmp_path,
            monkeypatch,
            """
services:
  main:
    depends_on:
      api:
        condition: service_healthy
  api:
    image: busybox:latest
""",
        )


@pytest.mark.parametrize(
    ("compose", "message"),
    [
        (
            "services: {main: {build: {context: ., network: host}}}\n",
            "build fields.*network",
        ),
        (
            "services: {main: {deploy: {resources: {}}}}\n",
            "only deploy.replicas",
        ),
        (
            "services: {main: {deploy: {replicas: 0}}}\n",
            "replicas.*positive integer",
        ),
        (
            "services: {main: {depends_on: [missing]}}\n",
            "depends on unknown services.*missing",
        ),
        (
            "services: {main: {volumes: ['./data:/data:ro']}}\n",
            "only writable bind mounts",
        ),
        (
            "services: {main: {volumes: ['named:/data']}}\n",
            "only writable bind mounts",
        ),
        (
            "services: {main: {ports: ['8000:8000/udp']}}\n",
            "only TCP published ports",
        ),
        (
            "services: {main: {command: [echo, 1]}}\n",
            "command.*only strings",
        ),
        (
            "services: {main: {environment: [VALUE=ok, 1]}}\n",
            "environment.*only strings",
        ),
        (
            "services: {main: {environment: {1: value}}}\n",
            "environment.*only strings",
        ),
        (
            "services: {main: {build: {args: [1]}}}\n",
            "build args.*string names",
        ),
        (
            "services: {main: {env_file: [1]}}\n",
            "env_file.*contain paths",
        ),
        (
            "services: {main: {depends_on: {1: {}}}}\n",
            "depends_on.*string names",
        ),
        (
            "services: {main: {volumes: {}}}\n",
            "volumes.*list",
        ),
        (
            "services: {main: {ports: {}}}\n",
            "ports.*list",
        ),
        (
            "services: {main: {env_file: {path: missing, required: 'false'}}}\n",
            "cannot represent env_file options",
        ),
        (
            "services: {main: {depends_on: api}, api: {}}\n",
            "depends_on.*list of names",
        ),
    ],
)
def test_singularity_compose_rejects_discarded_semantics(
    tmp_path, monkeypatch, compose, message
):
    with pytest.raises(ValueError, match=message):
        _compose_environment(tmp_path, monkeypatch, compose)


def test_singularity_compose_rejects_dependency_cycles(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="dependency cycle.*main.*api.*main"):
        _compose_environment(
            tmp_path,
            monkeypatch,
            "services: {main: {depends_on: [api]}, api: {depends_on: [main]}}\n",
        )


@pytest.mark.parametrize(
    "mount",
    [
        {"type": "volume", "source": "data", "target": "/data"},
        {
            "type": "bind",
            "source": "/host/data",
            "target": "/data",
            "read_only": True,
        },
    ],
)
def test_singularity_compose_rejects_unrepresentable_harbor_mounts(
    tmp_path, monkeypatch, mount
):
    with pytest.raises(ValueError, match="only Harbor bind|read-only"):
        _compose_environment(
            tmp_path,
            monkeypatch,
            "services: {main: {}}\n",
            mounts=[mount],
        )


def test_singularity_compose_normalizes_supported_long_syntax(
    tmp_path, monkeypatch
):
    data = tmp_path / "environment" / "data"
    data.mkdir(parents=True)
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        """
services:
  main:
    depends_on:
      api:
        condition: service_started
  api:
    image: busybox:latest
    volumes:
      - type: bind
        source: ./data
        target: /data
    ports:
      - target: 8000
        published: "18000"
        protocol: tcp
""",
    )

    config = environment._load_compose_config()

    assert config["services"]["main"]["depends_on"] == ["api"]
    assert config["services"]["api"]["volumes"] == [f"{data}:/data"]
    assert config["services"]["api"]["ports"] == ["18000:8000"]


def test_singularity_compose_uses_unique_instance_names(tmp_path, monkeypatch):
    compose = "services: {main: {}}\n"
    first = _compose_environment(tmp_path / "first", monkeypatch, compose)
    second = _compose_environment(tmp_path / "second", monkeypatch, compose)

    assert first.session_id == second.session_id
    assert first._compose_key("main") != second._compose_key("main")


def test_singularity_compose_uses_hostname_safe_instance_names(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {}}\n",
    )

    key = environment._compose_key("API_worker.v1-")

    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", key)
    assert len(f"{key}1") <= 63


@pytest.mark.asyncio
async def test_singularity_compose_start_adds_service_aliases_and_stops(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        """
services:
  main:
    depends_on: [api]
  api:
    image: busybox:latest
""",
    )
    commands = []

    async def build(_force):
        result = tmp_path / "main.sif"
        result.write_text("main")
        return result

    async def run(*command, cwd=None, env=None):
        commands.append((command, cwd))
        if (
            Path(command[0]).name == "singularity-compose"
            and command[-1] == "up"
        ):
            hosts = environment._compose_project_dir / "etc.hosts"
            hosts.write_text(
                "".join(
                    f"10.22.0.{index}\t{instance}\n"
                    for index, instance in enumerate(
                        (
                            names[0]
                            for names in environment._compose_instances.values()
                        ),
                        2,
                    )
                )
            )

    async def upload():
        pass

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
    monkeypatch.setattr(environment, "_run", run)
    monkeypatch.setattr(
        environment, "_upload_environment_dir_after_start", upload
    )

    await environment.start(force_build=False)

    hosts = (environment._compose_project_dir / "etc.hosts").read_text()
    assert "\tmain\n" in hosts
    assert "\tapi\n" in hosts
    assert environment._instance_started
    assert any(
        command[-1] == "up" and cwd == environment._compose_project_dir
        for command, cwd in commands
    )
    compose_actions = [
        command[-1]
        for command, _cwd in commands
        if Path(command[0]).name == "singularity-compose"
    ]
    assert compose_actions == ["up"]

    await environment.stop(delete=False)

    assert any("down" in command for command, _cwd in commands)
    assert environment._compose_project_dir is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "message"),
    [
        ("up", "up failed"),
        ("aliases", "did not assign an address"),
        ("readiness", "readiness failed"),
    ],
)
async def test_singularity_compose_failed_start_preserves_error_and_cleans_up(
    tmp_path, monkeypatch, failure_stage, message
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {}, api: {image: busybox:latest}}\n",
    )
    commands = []
    created = {}

    async def build(_force):
        result = tmp_path / "main.sif"
        result.write_text("main")
        return result

    async def run(*command, **_kwargs):
        commands.append(command)
        if Path(command[0]).name == "singularity-compose":
            if "up" in command:
                created["project"] = environment._compose_project_dir
                created["staging"] = environment._staging_dir
                if failure_stage == "up":
                    raise RuntimeError("up failed")
                hosts = environment._compose_project_dir / "etc.hosts"
                if failure_stage == "aliases":
                    hosts.write_text("")
                else:
                    hosts.write_text(
                        "".join(
                            f"10.22.0.{index}\t{names[0]}\n"
                            for index, names in enumerate(
                                environment._compose_instances.values(), 2
                            )
                        )
                    )
            elif "down" in command:
                raise RuntimeError("down cleanup failed")
        elif command[-1] == "true" and failure_stage == "readiness":
            raise RuntimeError("readiness failed")
        elif command[1:4] == ("instance", "stop", "--force"):
            raise RuntimeError("direct cleanup failed")

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
    monkeypatch.setattr(environment, "_run", run)

    with pytest.raises(RuntimeError, match=message):
        await environment.start(force_build=False)

    assert any("down" in command for command in commands)
    assert environment._compose_project_dir is None
    assert environment._staging_dir is None
    assert environment._compose_instances == {}
    assert not environment._instance_started
    assert not created["project"].exists()
    assert not created["staging"].exists()


@pytest.mark.asyncio
async def test_singularity_compose_serializes_concurrent_startup(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    environments = [
        _compose_environment(
            tmp_path / name,
            monkeypatch,
            "services: {main: {}}\n",
        )
        for name in ("first", "second")
    ]
    active = 0
    maximum_active = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    for environment in environments:

        async def build(_force, *, root=environment.environment_dir):
            result = root / "main.sif"
            result.write_text("main")
            return result

        async def run(*command, cwd=None, env=None, owner=environment):
            nonlocal active, maximum_active
            if (
                Path(command[0]).name != "singularity-compose"
                or command[-1] != "up"
            ):
                return
            active += 1
            maximum_active = max(maximum_active, active)
            if not first_started.is_set():
                first_started.set()
                await release_first.wait()
            hosts = owner._compose_project_dir / "etc.hosts"
            hosts.write_text(
                f"10.22.0.2\t{owner._compose_instances['main'][0]}\n"
            )
            active -= 1

        async def upload():
            pass

        monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
        monkeypatch.setattr(environment, "_run", run)
        monkeypatch.setattr(
            environment, "_upload_environment_dir_after_start", upload
        )

    starts = [
        asyncio.create_task(environment.start(force_build=False))
        for environment in environments
    ]
    await first_started.wait()
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(*starts)

    assert maximum_active == 1
    await asyncio.gather(
        *(environment.stop(delete=False) for environment in environments)
    )


@pytest.mark.asyncio
async def test_singularity_compose_stop_cleans_sidecars_after_main_stops(
    tmp_path, monkeypatch
):
    environment = _compose_environment(
        tmp_path,
        monkeypatch,
        "services: {main: {}, api: {image: busybox:latest}}\n",
    )
    environment._compose_project_dir = tmp_path / "project"
    environment._compose_project_dir.mkdir()
    environment._compose_file = environment._compose_project_dir / "compose.yml"
    environment._compose_file.write_text("instances: {}\n")
    environment._compose_instances = {"main": ["main1"], "api": ["api1"]}
    environment._instance_started = True
    commands = []

    async def run_compose(*command):
        commands.append(command)

    monkeypatch.setattr(environment, "_run_compose", run_compose)

    await environment.stop_service("main")
    assert not environment._instance_started
    with pytest.raises(RuntimeError, match="not running"):
        await environment.attach()
    await environment.stop(delete=False)

    assert commands == [
        ("stop", "--timeout", "0", "main1"),
        ("down", "--timeout", "0"),
    ]


def _network_environment(
    tmp_path,
    *,
    network_policy=None,
    phase_network_policies=(),
    **kwargs,
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    return DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="test",
        session_id="trial__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        network_policy=network_policy,
        phase_network_policies=phase_network_policies,
        **kwargs,
    )


def test_singularity_advertises_only_static_network_isolation():
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )

    capabilities = environment.capabilities

    assert capabilities.mounted
    assert capabilities.disable_internet
    assert not capabilities.network_allowlist
    assert not capabilities.dynamic_network_policy


def test_singularity_accepts_no_network_policy(tmp_path):
    environment = _network_environment(
        tmp_path,
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
    )

    assert environment._network_policy.network_mode == NetworkMode.NO_NETWORK


def test_singularity_rejects_allowlist_policy(tmp_path):
    with pytest.raises(ValueError, match="network_mode='allowlist'"):
        _network_environment(
            tmp_path,
            network_policy=NetworkPolicy(
                network_mode=NetworkMode.ALLOWLIST,
                allowed_hosts=["example.com"],
            ),
        )


def test_singularity_rejects_dynamic_phase_policy(tmp_path):
    with pytest.raises(ValueError, match="cannot change network policy"):
        _network_environment(
            tmp_path,
            network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
            phase_network_policies=[
                NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
            ],
        )


def test_singularity_accepts_identical_phase_policy(tmp_path):
    public = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

    environment = _network_environment(
        tmp_path,
        network_policy=public,
        phase_network_policies=[public],
    )

    assert environment._phase_network_policies == [public]


@pytest.mark.parametrize("value", ["", "home", "tmp", "bind-paths"])
def test_singularity_rejects_no_mount_configuration(tmp_path, value):
    with pytest.raises(ValueError, match="not configurable"):
        _network_environment(
            tmp_path,
            singularity_no_mount=value,
        )


def _instance_test_environment(tmp_path, network_mode=NetworkMode.PUBLIC):
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    environment._startup_timeout_sec = 300
    environment._mounts = []
    environment._workdir = "/workspace"
    environment._sif_path = tmp_path / "image.sif"
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    environment._instance_name = "ursatestinstance"
    environment._instance_started = False
    environment._network_policy = NetworkPolicy(network_mode=network_mode)
    environment._force_pull = False
    environment._fakeroot = True
    environment.logger = SimpleNamespace(
        debug=lambda *_args: None,
        warning=lambda *_args: None,
    )
    environment._runtime = lambda: "/usr/bin/singularity"
    return environment


@pytest.mark.asyncio
async def test_singularity_uploads_same_basename_through_unique_staging_paths(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    first = tmp_path / "first" / "payload.txt"
    second = tmp_path / "second" / "payload.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")
    commands = []
    both_started = asyncio.Event()

    async def execute(command, **_kwargs):
        commands.append(command)
        if len(commands) == 1:
            await both_started.wait()
        else:
            both_started.set()
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(environment, "exec", execute)

    await asyncio.gather(
        environment.upload_file(first, "/first/payload.txt"),
        environment.upload_file(second, "/second/payload.txt"),
    )

    staged_sources = [shlex.split(command)[1] for command in commands]
    assert len(set(staged_sources)) == 2
    assert not list(environment._staging.iterdir())


@pytest.mark.asyncio
async def test_singularity_upload_dir_removes_stale_staging_path(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    source = tmp_path / "source" / "payload"
    source.mkdir(parents=True)
    (source / "new.txt").write_text("new")
    monkeypatch.setattr(
        "ursa.integrations.harbor_singularity.secrets.token_hex",
        lambda _length: "fixed",
    )
    staged = environment._staging / "harbor-transfer-fixed-payload"
    staged.mkdir()
    (staged / "stale.txt").write_text("stale")
    calls = 0

    async def execute(_command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert (staged / "new.txt").read_text() == "new"
            assert not (staged / "stale.txt").exists()
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(environment, "exec", execute)

    await environment.upload_dir(source, "/workspace/payload")

    assert calls == 2
    assert not staged.exists()


@pytest.mark.asyncio
async def test_singularity_upload_dir_stops_when_mkdir_fails(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    source = tmp_path / "payload"
    source.mkdir()
    commands = []

    async def execute(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(return_code=31, stdout="", stderr="no mkdir")

    monkeypatch.setattr(environment, "exec", execute)

    with pytest.raises(
        RuntimeError, match="prepare upload directory.*no mkdir"
    ):
        await environment.upload_dir(source, "/workspace/payload")

    assert len(commands) == 1
    assert not list(environment._staging.iterdir())


@pytest.mark.asyncio
async def test_singularity_cancelled_download_cleans_staging(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    started = asyncio.Event()

    async def execute(command, **_kwargs):
        container_path = shlex.split(command)[2]
        staged = environment._staging / Path(container_path).name
        staged.write_text("partial")
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(environment, "exec", execute)
    task = asyncio.create_task(
        environment.download_file("/workspace/payload", tmp_path / "result")
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not list(environment._staging.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_singularity_compose_downloads_from_sidecar(
    tmp_path, monkeypatch, directory
):
    environment = _instance_test_environment(tmp_path)
    environment._compose_instances = {"main": ["main1"], "api": ["api1"]}

    async def service_exec(command, *, service):
        assert service == "api"
        staged = environment._staging / shlex.split(command)[-1].split("/")[-1]
        if directory:
            staged.mkdir()
            (staged / "result.txt").write_text("sidecar directory")
        else:
            staged.write_text("sidecar file")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(environment, "service_exec", service_exec)
    target = tmp_path / "download"

    if directory:
        await environment.service_download_dir("/output", target, service="api")
        assert (target / "result.txt").read_text() == "sidecar directory"
    else:
        await environment.service_download_file(
            "/output.txt", target, service="api"
        )
        assert target.read_text() == "sidecar file"
    assert not list(environment._staging.iterdir())


def test_singularity_public_instance_uses_36_flags(tmp_path):
    environment = _instance_test_environment(tmp_path)

    command = environment._instance_start_command()

    assert command[:3] == [
        "/usr/bin/singularity",
        "instance",
        "start",
    ]
    assert "--fakeroot" in command
    assert "--containall" in command
    assert "--no-home" in command
    assert "--writable-tmpfs" in command
    assert "--pwd" not in command
    assert "--no-mount" not in command
    assert "--net" not in command
    assert command[-2:] == [
        str(environment._sif_path),
        environment._instance_name,
    ]


def test_singularity_instance_can_disable_fakeroot(tmp_path):
    environment = _instance_test_environment(tmp_path)
    environment._fakeroot = False

    command = environment._instance_start_command()

    assert "--fakeroot" not in command
    assert "--containall" in command


def test_singularity_no_network_uses_none_network(tmp_path):
    environment = _instance_test_environment(
        tmp_path, network_mode=NetworkMode.NO_NETWORK
    )

    command = environment._instance_start_command()

    network_index = command.index("--net")
    assert command[network_index : network_index + 3] == [
        "--net",
        "--network",
        "none",
    ]


def test_singularity_instance_mounts_staging_and_configured_binds(tmp_path):
    environment = _instance_test_environment(tmp_path)
    environment._mounts = [
        {"type": "bind", "source": "/host/logs", "target": "/logs"},
        {
            "type": "bind",
            "source": "/host/input",
            "target": "/input",
            "read_only": True,
        },
    ]

    command = environment._instance_start_command()

    binds = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "-B"
    ]
    assert binds == [
        f"{environment._staging_dir}:/staging",
        "/host/logs:/logs",
        "/host/input:/input:ro",
    ]


def test_singularity_rejects_unsupported_harbor_mount_type(tmp_path):
    environment = _instance_test_environment(tmp_path)
    environment._mounts = [
        {"type": "volume", "source": "data", "target": "/data"}
    ]

    with pytest.raises(ValueError, match="only Harbor bind mounts"):
        environment._instance_start_command()


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["singularity", "apptainer"])
async def test_singularity_attach_execs_shell_in_main_instance(
    tmp_path, monkeypatch, runtime
):
    environment = _instance_test_environment(tmp_path)
    environment._instance_started = True
    environment._instance_name = "compose-main1"
    environment._runtime_path = f"/usr/bin/{runtime}"
    calls = []
    monkeypatch.setenv("SINGULARITYENV_SECRET", "singularity-secret")
    monkeypatch.setenv("APPTAINERENV_SECRET", "apptainer-secret")
    monkeypatch.setenv("PRESERVED", "value")
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda executable, command, env: calls.append((
            executable,
            command,
            env,
        )),
    )

    await environment.attach()

    expected = [
        f"/usr/bin/{runtime}",
        "shell",
        "--cleanenv",
        "--pwd",
        "/workspace",
        "instance://compose-main1",
    ]
    assert calls[0][0:2] == (f"/usr/bin/{runtime}", expected)
    assert calls[0][2]["PRESERVED"] == "value"
    assert "SINGULARITYENV_SECRET" not in calls[0][2]
    assert "APPTAINERENV_SECRET" not in calls[0][2]


@pytest.mark.asyncio
async def test_singularity_attach_requires_running_instance(tmp_path):
    environment = _instance_test_environment(tmp_path)

    with pytest.raises(RuntimeError, match="not running"):
        await environment.attach()


@pytest.mark.asyncio
async def test_singularity_start_exec_and_stop_use_one_instance(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    environment._staging_dir = None
    environment._validate_definition = lambda: None
    commands = []

    async def fake_build(_force_build):
        return tmp_path / "image.sif"

    async def fake_run(*command):
        commands.append(command)

    async def fake_upload():
        assert environment._instance_started
        assert environment._staging_dir is not None

    monkeypatch.setattr(environment, "_build_dockerfile_sif", fake_build)
    monkeypatch.setattr(environment, "_run", fake_run)
    monkeypatch.setattr(
        environment, "_upload_environment_dir_after_start", fake_upload
    )

    await environment.start(force_build=False)
    await environment.stop(delete=False)

    assert commands[0][0:3] == (
        "/usr/bin/singularity",
        "instance",
        "start",
    )
    assert commands[1] == (
        "/usr/bin/singularity",
        "exec",
        "--cleanenv",
        "--pwd",
        "/workspace",
        "instance://ursatestinstance",
        "true",
    )
    assert commands[2] == (
        "/usr/bin/singularity",
        "instance",
        "stop",
        "--force",
        "ursatestinstance",
    )
    assert environment._staging_dir is None


@pytest.mark.asyncio
async def test_apptainer_instance_uses_runtime_directly(tmp_path, monkeypatch):
    environment = _instance_test_environment(tmp_path)
    environment._staging_dir = None
    environment._runtime = lambda: "/usr/bin/apptainer"
    environment._validate_definition = lambda: None
    commands = []

    async def fake_build(_force_build):
        return tmp_path / "image.sif"

    async def fake_run(*command):
        commands.append(command)

    async def fake_upload():
        pass

    monkeypatch.setattr(environment, "_build_dockerfile_sif", fake_build)
    monkeypatch.setattr(environment, "_run", fake_run)
    monkeypatch.setattr(
        environment, "_upload_environment_dir_after_start", fake_upload
    )

    await environment.start(force_build=False)

    assert commands[0][0:3] == (
        "/usr/bin/apptainer",
        "instance",
        "start",
    )
    assert commands[1][0] == "/usr/bin/apptainer"


@pytest.mark.asyncio
async def test_singularity_failed_start_cleans_instance_and_staging(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    environment._staging_dir = None
    environment._validate_definition = lambda: None
    commands = []

    async def fake_build(_force_build):
        return tmp_path / "image.sif"

    async def fake_run(*command):
        commands.append(command)
        if command[1] == "exec":
            raise RuntimeError("readiness failed")

    monkeypatch.setattr(environment, "_build_dockerfile_sif", fake_build)
    monkeypatch.setattr(environment, "_run", fake_run)

    with pytest.raises(RuntimeError, match="readiness failed"):
        await environment.start(force_build=False)

    assert commands[-1][1:3] == ("instance", "stop")
    assert environment._staging_dir is None
    assert not environment._instance_started


@pytest.mark.asyncio
async def test_singularity_startup_timeout_cleans_possible_instance(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    environment._staging_dir = None
    environment._startup_timeout_sec = 0.01
    environment._validate_definition = lambda: None
    commands = []

    async def fake_build(_force_build):
        return tmp_path / "image.sif"

    async def fake_run(*command):
        commands.append(command)
        if command[1:3] == ("instance", "start"):
            await asyncio.Event().wait()

    monkeypatch.setattr(environment, "_build_dockerfile_sif", fake_build)
    monkeypatch.setattr(environment, "_run", fake_run)

    with pytest.raises(TimeoutError, match="did not become ready"):
        await environment.start(force_build=False)

    assert commands[-1][1:3] == ("instance", "stop")
    assert environment._staging_dir is None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_singularity_rejects_invalid_startup_timeout(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        DockerfileSingularityEnvironment(
            singularity_startup_timeout_sec=timeout
        )


@pytest.mark.asyncio
async def test_singularity_build_command_reaps_process_on_cancellation():
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    task = asyncio.create_task(environment._run("bash", "-c", "sleep 30"))
    await asyncio.sleep(0.05)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)


def _exec_test_environment(tmp_path):
    environment = _instance_test_environment(tmp_path)
    environment._instance_started = True
    environment._merge_env = lambda env: env
    environment._resolve_user = lambda user: user
    return environment


def test_apptainer_exec_env_uses_native_prefix(tmp_path, monkeypatch):
    environment = _exec_test_environment(tmp_path)
    environment._runtime = lambda: "/usr/bin/apptainer"
    monkeypatch.setenv("SINGULARITYENV_OLD", "remove")
    monkeypatch.setenv("APPTAINERENV_OLD", "remove")

    runtime_env = environment._runtime_environment({"TOKEN": "secret"})

    assert runtime_env["APPTAINERENV_TOKEN"] == "secret"
    assert "SINGULARITYENV_TOKEN" not in runtime_env
    assert "SINGULARITYENV_OLD" not in runtime_env
    assert "APPTAINERENV_OLD" not in runtime_env


@pytest.mark.asyncio
async def test_singularity_compose_sidecar_exec_is_isolated(
    tmp_path, monkeypatch
):
    environment = _exec_test_environment(tmp_path)
    environment._compose_instances = {
        "main": ["main1"],
        "api": ["api1"],
    }
    environment.default_user = "agent"
    environment._merge_env = lambda env: {"PERSISTENT": "bad", **(env or {})}
    calls = []

    class Process:
        pid = 12345
        returncode = 0

        async def communicate(self):
            return b"sidecar", b""

    async def create_process(*command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    result = await environment.service_exec(
        "printf sidecar", service="api", env={"LOCAL": "yes"}
    )

    assert result.stdout == "sidecar"
    command, kwargs = calls[0]
    assert "instance://api1" in command
    assert command[command.index("--pwd") + 1] == "/"
    assert kwargs["env"]["SINGULARITYENV_LOCAL"] == "yes"
    assert "SINGULARITYENV_PERSISTENT" not in kwargs["env"]
    assert "su agent" not in command[-1]


@pytest.mark.asyncio
async def test_singularity_compose_stops_every_service_replica(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    environment._compose_instances = {
        "main": ["main1"],
        "api": ["api1", "api2"],
    }
    commands = []

    async def run_compose(*command):
        commands.append(command)

    monkeypatch.setattr(environment, "_run_compose", run_compose)

    await environment.stop_service("api")

    assert commands == [("stop", "--timeout", "0", "api1", "api2")]


@pytest.mark.asyncio
async def test_singularity_compose_down_falls_back_to_direct_instance_stop(
    tmp_path, monkeypatch
):
    environment = _instance_test_environment(tmp_path)
    environment._compose_instances = {
        "main": ["main1"],
        "api": ["api1", "api2"],
    }
    commands = []

    async def run_compose(*_command):
        raise RuntimeError("compose down failed")

    async def run(*command):
        commands.append(command)

    monkeypatch.setattr(environment, "_run_compose", run_compose)
    monkeypatch.setattr(environment, "_run", run)

    await environment._stop_compose(warn=True)

    assert {command[-1] for command in commands} == {"main1", "api1", "api2"}
    assert all(
        command[1:4] == ("instance", "stop", "--force") for command in commands
    )


@pytest.mark.asyncio
async def test_singularity_exec_uses_instance_and_closes_stdin(
    tmp_path, monkeypatch
):
    environment = _exec_test_environment(tmp_path)
    calls = []

    class Process:
        pid = 12345
        returncode = 0

        async def communicate(self):
            return b"done", b""

    async def create_process(*command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    first = await environment.exec(
        "printf done",
        cwd="/workspace/project",
        env={"TOKEN": "value with spaces"},
        user="agent",
    )
    second = await environment.exec("true")

    assert first.return_code == 0
    assert first.stdout == "done"
    assert second.return_code == 0
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[:2] == (
            "/usr/bin/singularity",
            "exec",
        )
        assert "instance://ursatestinstance" in command
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
        assert "--cleanenv" in command
    shell_command = calls[0][0][-1]
    assert "/workspace/project" in shell_command
    assert "TOKEN=value with spaces" not in shell_command
    assert "su agent" in shell_command
    assert calls[0][1]["env"]["SINGULARITYENV_TOKEN"] == "value with spaces"
    assert "SINGULARITYENV_TOKEN" not in calls[1][1]["env"]


@pytest.mark.asyncio
async def test_singularity_exec_without_fakeroot_does_not_switch_user(
    tmp_path, monkeypatch
):
    environment = _exec_test_environment(tmp_path)
    environment._fakeroot = False
    environment.default_user = "agent"
    calls = []

    class Process:
        pid = 12345
        returncode = 0

        async def communicate(self):
            return b"done", b""

    async def create_process(*command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    await environment.exec("printf explicit", user="root")
    await environment.exec("printf default")

    assert len(calls) == 2
    assert all("su " not in command[-1] for command, _kwargs in calls)


@pytest.mark.asyncio
async def test_singularity_exec_timeout_returns_124_and_cleans_remote(
    tmp_path, monkeypatch
):
    environment = _exec_test_environment(tmp_path)
    release = asyncio.Event()
    cleaned = []

    class Process:
        pid = 12345
        returncode = None

        async def communicate(self):
            await release.wait()
            return b"partial", b""

    process = Process()

    async def create_process(*_command, **_kwargs):
        return process

    async def cleanup(received_process, pid_file):
        cleaned.append(pid_file)
        assert received_process is process
        process.returncode = -15
        release.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(environment, "_cleanup_exec_process", cleanup)

    result = await environment.exec("sleep 30", timeout_sec=0.01)

    assert result.return_code == 124
    assert result.stdout == "partial"
    assert "timed out" in result.stderr
    assert len(cleaned) == 1


@pytest.mark.asyncio
async def test_singularity_exec_cancellation_cleans_remote(
    tmp_path, monkeypatch
):
    environment = _exec_test_environment(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    class Process:
        pid = 12345
        returncode = None

        async def communicate(self):
            started.set()
            await release.wait()
            return b"", b""

    process = Process()

    async def create_process(*_command, **_kwargs):
        return process

    async def cleanup(received_process, _pid_file):
        assert received_process is process
        process.returncode = -15
        release.set()
        cleaned.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(environment, "_cleanup_exec_process", cleanup)
    task = asyncio.create_task(environment.exec("sleep 30"))
    await started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_singularity_remote_cleanup_terminates_descendants(tmp_path):
    pid_file = tmp_path / "exec.pid"
    child_file = tmp_path / "child.pid"
    cleanup = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        DockerfileSingularityEnvironment._terminate_process_tree_command(
            str(pid_file)
        ),
    )
    await asyncio.sleep(0.1)
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        f"echo $$ > {pid_file}; sleep 30 & echo $! > {child_file}; wait",
    )
    for _ in range(100):
        if pid_file.is_file() and child_file.is_file():
            break
        await asyncio.sleep(0.01)
    child_pid = int(child_file.read_text())

    assert await cleanup.wait() == 0
    await asyncio.wait_for(process.wait(), timeout=3)
    for _ in range(100):
        if not Path(f"/proc/{child_pid}").exists():
            break
        await asyncio.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()
