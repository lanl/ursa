import asyncio
import base64
import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

harbor = pytest.importorskip("harbor")

from harbor.environments.singularity import (  # noqa: E402
    singularity as harbor_singularity,
)
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
    _attach_mcp_tools,
    _capture_output,
    _close_checkpoint,
    _export_checkpoint,
    _usage,
)
from ursa.integrations.harbor_runner import (  # noqa: E402
    main as _runner_main,
)
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
)


def _config(path: Path) -> Path:
    path.write_text("llm_model:\n  model: gpt-4.1-nano\n")
    return path


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


def _singularity_env(
    tmp_path,
    monkeypatch,
    builders=("docker",),
    fail=None,
    runtime="singularity",
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True)
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


def _singularity_preflight_result(command, *, builder_returncode=0):
    arguments = tuple(command[1:])
    help_text = {
        ("instance", "start", "--help"): (
            "--fakeroot --containall --no-home --writable-tmpfs --net --network"
        ),
        ("exec", "--help"): "--cleanenv --pwd",
        ("instance", "stop", "--help"): "--force",
    }
    if arguments in help_text:
        return SimpleNamespace(
            returncode=0, stdout=help_text[arguments], stderr=""
        )
    if arguments == ("info",):
        return SimpleNamespace(
            returncode=builder_returncode,
            stdout="",
            stderr="builder unavailable" if builder_returncode else "",
        )
    raise AssertionError(f"Unexpected preflight command: {command}")


def test_singularity_preflight_accepts_runtime_and_builder(monkeypatch):
    commands = []

    def which(name):
        return {
            "apptainer": "/usr/bin/apptainer",
            "buildah": "/usr/bin/buildah",
        }.get(name)

    def run(command, **_kwargs):
        commands.append(command)
        return _singularity_preflight_result(command)

    monkeypatch.setattr("shutil.which", which)
    monkeypatch.setattr("subprocess.run", run)

    DockerfileSingularityEnvironment.preflight()

    assert commands[-1] == ["/usr/bin/buildah", "info"]


def test_singularity_preflight_rejects_missing_runtime(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(SystemExit, match="Apptainer or Singularity"):
        DockerfileSingularityEnvironment.preflight()


def test_singularity_preflight_rejects_incompatible_runtime(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/apptainer" if name == "apptainer" else None,
    )

    def run(command, **_kwargs):
        result = _singularity_preflight_result(command)
        if command[1:4] == ["instance", "start", "--help"]:
            result.stdout = result.stdout.replace("--network", "")
        return result

    monkeypatch.setattr("subprocess.run", run)

    with pytest.raises(SystemExit, match="required options.*--network"):
        DockerfileSingularityEnvironment.preflight()


def test_singularity_preflight_rejects_missing_builder(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/singularity" if name == "singularity" else None,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **_kwargs: _singularity_preflight_result(command),
    )

    with pytest.raises(SystemExit, match="requires buildah, podman, or docker"):
        DockerfileSingularityEnvironment.preflight()


def test_singularity_preflight_rejects_unusable_builder(monkeypatch):
    def which(name):
        return {
            "singularity": "/usr/bin/singularity",
            "docker": "/usr/bin/docker",
        }.get(name)

    monkeypatch.setattr("shutil.which", which)
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **_kwargs: _singularity_preflight_result(
            command, builder_returncode=1
        ),
    )

    with pytest.raises(
        SystemExit, match="No usable.*docker: builder unavailable"
    ):
        DockerfileSingularityEnvironment.preflight()


def test_singularity_uses_a_shared_default_cache(tmp_path, monkeypatch):
    received = []

    def fake_init(_self, *args, **kwargs):
        received.append(kwargs["singularity_image_cache_dir"])
        _self._phase_network_policies = []
        _self.session_id = "test"

    monkeypatch.setattr(
        harbor_singularity.SingularityEnvironment, "__init__", fake_init
    )
    monkeypatch.delenv("URSA_HARBOR_SIF_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))

    DockerfileSingularityEnvironment()
    configured = tmp_path / "configured-cache"
    monkeypatch.setenv("URSA_HARBOR_SIF_CACHE", str(configured))
    DockerfileSingularityEnvironment()
    explicit = tmp_path / "explicit-cache"
    DockerfileSingularityEnvironment(singularity_image_cache_dir=explicit)

    assert received == [
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
        config_only="true",
    )

    runtime_config, _ = agent._runtime_config()

    assert runtime_config["agent_name"] == "supplied"
    assert runtime_config["llm_model"]["model"] == "gpt-5.4-nano"
    assert runtime_config["llm_model"]["max_completion_tokens"] == 321


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

    async def fake_exec_as_root(environment, command, **kwargs):
        commands.append(command)

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
    assert any(
        "uv venv --managed-python --python 3.13" in command
        and "sys.version_info[:2] == (3, 13)" in command
        for command in commands
    )
    assert any(
        "uv pip install" in command
        and "ursa-ai[image]==1.2" in command
        and "numpy" in command
        and "scipy" in command
        for command in commands
    )
    runtime_config, destination, mode = uploads[0]
    assert destination == "/tmp/ursa-config.json"
    assert mode == 0o600
    assert runtime_config["inference_providers"]["openai"]["api_key"] == {
        "env": "URSA_HARBOR_SECRET_0"
    }
    assert agent._secret_env == {"URSA_HARBOR_SECRET_0": "host-openai-key"}
    assert agent._workspace == "/app"


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
        ursa_source_dir=source,
    )
    uploaded = []

    async def fake_exec_as_root(*args, **kwargs):
        pass

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
            stdout='URSA_HARBOR_RESULT={"result": null}\n',
        )

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    await agent.run("task", object(), SimpleNamespace())

    assert observed_timeout is None
    assert observed_env["URSA_HARBOR_SECRET_0"] == "resolved-on-host"
    assert observed_command is not None
    encoded = shlex.split(observed_command)[-1]
    payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
    assert payload["log_path"] == "/logs/agent/ursa.log"


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


def test_checkpoint_is_exported_to_harbor_artifacts(tmp_path):
    class Agent:
        den = tmp_path / "den"

    checkpoint = Agent.den / "db" / "checkpointer.db"
    checkpoint.parent.mkdir(parents=True)
    with sqlite3.connect(checkpoint) as database:
        database.execute("CREATE TABLE checkpoints (value TEXT)")
        database.execute("INSERT INTO checkpoints VALUES ('saved')")

    exported = _export_checkpoint(Agent(), tmp_path / "artifacts")

    assert exported == tmp_path / "artifacts" / "ursa" / "checkpointer.db"
    with sqlite3.connect(exported) as database:
        assert database.execute("SELECT value FROM checkpoints").fetchone() == (
            "saved",
        )


def test_checkpoint_already_in_artifacts_is_not_copied(tmp_path):
    destination = tmp_path / "artifacts" / "ursa" / "checkpointer.db"
    destination.parent.mkdir(parents=True)
    connection = sqlite3.connect(destination)
    connection.execute("CREATE TABLE checkpoints (value TEXT)")

    class Checkpointer:
        conn = connection

    class Agent:
        checkpointer = Checkpointer()

    assert _export_checkpoint(Agent(), tmp_path / "artifacts") == destination
    connection.close()


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


@pytest.mark.asyncio
async def test_singularity_builds_dockerfile_on_demand(tmp_path, monkeypatch):
    environment, commands = _singularity_env(tmp_path, monkeypatch)

    result = await environment._build_dockerfile_sif(force_build=False)

    assert result.is_file()
    assert commands[0][1] == "build"
    assert commands[1][1] == "save"
    assert commands[2][0:2] == ("/usr/bin/singularity", "build")
    assert commands[2][3].startswith("docker-archive://")

    build_count = sum(command[1] == "build" for command in commands)
    assert await environment._build_dockerfile_sif(False) == result
    assert sum(command[1] == "build" for command in commands) == build_count


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


@pytest.mark.parametrize("value", ["bind-paths", "home,bind-paths"])
def test_singularity_36_rejects_unsupported_no_mount(tmp_path, value):
    with pytest.raises(ValueError, match="Singularity 3.6"):
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
    environment._memory_limit_exceeded = None
    environment._force_pull = False
    environment.logger = SimpleNamespace(
        debug=lambda *_args: None,
        warning=lambda *_args: None,
    )
    environment._runtime = lambda: "/usr/bin/singularity"
    return environment


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
        {"type": "volume", "source": "ignored", "target": "/ignored"},
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
    ]


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
