import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

harbor = pytest.importorskip("harbor")

from harbor.models.task.config import MCPServerConfig  # noqa: E402

from ursa.agents import BaseAgent  # noqa: E402
from ursa.agents.base import AgentWithTools  # noqa: E402
from ursa.cli.config import UrsaConfig  # noqa: E402
from ursa.integrations.harbor import (  # noqa: E402
    UrsaHarborAgent,
    make_harbor_agent,
)
from ursa.integrations.harbor_runner import (  # noqa: E402
    _apply_harbor_overrides,
    _attach_mcp_tools,
    _export_checkpoint,
)
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
)


def _config(path: Path) -> Path:
    path.write_text("llm_model:\n  model: gpt-4.1-nano\n")
    return path


def _singularity_env(tmp_path, monkeypatch, builders=("docker",), fail=None):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
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
        if command[0:3] == ("singularity", "sif", "list"):
            if Path(command[3]).read_text() == "broken":
                raise RuntimeError("invalid SIF")
        if command[:2] == ("singularity", "build"):
            Path(command[2]).write_text("sif")

    monkeypatch.setattr(
        "shutil.which",
        lambda command: f"/usr/bin/{command}" if command in builders else None,
    )
    monkeypatch.setattr(environment, "_run", fake_run)
    return environment, commands


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

    class FakeEnvironment:
        async def upload_file(self, source, destination):
            uploads.append((source, destination))

    uploads = []

    monkeypatch.setattr(agent, "exec_as_root", fake_exec_as_root)

    await agent.install(FakeEnvironment())

    assert "command -v tar" in commands[0]
    assert "curl ca-certificates tar" in commands[0]
    assert "unknown-linux-musl.tar.gz" in commands[0]
    assert "case $(uname -m)" in commands[0]
    assert any("uv venv --python 3.12" in command for command in commands)
    assert any(
        "uv pip install" in command
        and "ursa-ai[image]==1.2" in command
        and "numpy" in command
        and "scipy" in command
        for command in commands
    )
    assert uploads == [(config_file, "/tmp/ursa-config.yaml")]


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


@pytest.mark.asyncio
async def test_run_leaves_trial_timeout_to_harbor(tmp_path, monkeypatch):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4.1-nano",
        config_file=_config(tmp_path / "ursa.yaml"),
    )
    agent._remote_config_file = "/tmp/ursa-config.yaml"
    observed_timeout = object()

    async def fake_exec_as_agent(*args, **kwargs):
        nonlocal observed_timeout
        observed_timeout = kwargs["timeout_sec"]
        return SimpleNamespace(
            return_code=0,
            stdout='URSA_HARBOR_RESULT={"result": null}\n',
        )

    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)

    await agent.run("task", object(), SimpleNamespace())

    assert observed_timeout is None


def test_harbor_model_overrides_ursa_model():
    config = UrsaConfig.model_validate({
        "inference_providers": {
            "ollama": {
                "model_provider": "ollama",
                "base_url": "http://localhost:11434",
            }
        },
        "llm_model": {"model": "old", "inference_provider": "openai"},
    })

    resolved = _apply_harbor_overrides(config, "ollama/gemma4", {}).resolve()

    assert resolved.llm_model.model == "gemma4"
    assert resolved.llm_model.inference_provider == "ollama"
    assert resolved.llm_model.model_provider == "ollama"


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


def test_harbor_mcp_servers_merge_and_override_config():
    config = UrsaConfig.model_validate({
        "mcp_servers": {
            "replaced": {
                "transport": "stdio",
                "command": "old-command",
            }
        }
    })
    result = _apply_harbor_overrides(
        config,
        None,
        {
            "replaced": {
                "transport": "stdio",
                "command": "new-command",
                "args": ["--serve"],
            },
            "events": {"transport": "sse", "url": "http://events/sse"},
        },
    )

    assert result.mcp_servers["replaced"].command == "new-command"
    assert set(result.mcp_servers) == {"replaced", "events"}


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


@pytest.mark.asyncio
async def test_singularity_builds_dockerfile_on_demand(tmp_path, monkeypatch):
    environment, commands = _singularity_env(tmp_path, monkeypatch)

    result = await environment._build_dockerfile_sif(force_build=False)

    assert result.is_file()
    assert commands[0][1] == "build"
    assert commands[1][1] == "save"
    assert commands[2][0:2] == ("singularity", "build")
    assert commands[2][3].startswith("docker-archive://")

    build_count = sum(command[1] == "build" for command in commands)
    assert await environment._build_dockerfile_sif(False) == result
    assert sum(command[1] == "build" for command in commands) == build_count


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
async def test_singularity_build_command_reaps_process_on_cancellation():
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    task = asyncio.create_task(environment._run("bash", "-c", "sleep 30"))
    await asyncio.sleep(0.05)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
