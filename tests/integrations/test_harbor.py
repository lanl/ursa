import asyncio
import json
import os
import sqlite3
import subprocess
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
    _close_checkpoint,
    _export_checkpoint,
    _usage,
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


def _singularity_env(
    tmp_path,
    monkeypatch,
    builders=("docker",),
    fail=None,
    runtime="singularity",
):
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


def test_singularity_bootstrap_mounts_are_idempotent():
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    environment._mounts = []

    environment._ensure_bootstrap_mounts()
    environment._ensure_bootstrap_mounts()

    targets = [mount["target"] for mount in environment._mounts]
    assert targets == [
        "/staging/bootstrap-upstream.sh",
        "/staging/bootstrap.sh",
    ]
    assert all(Path(mount["source"]).is_file() for mount in environment._mounts)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("3.9", "https://bootstrap.pypa.io/pip/3.9/get-pip.py"),
        ("3.10", "https://bootstrap.pypa.io/get-pip.py"),
        (None, "https://bootstrap.pypa.io/get-pip.py"),
    ],
)
def test_singularity_bootstrap_selects_compatible_get_pip(
    tmp_path, version, expected
):
    upstream = tmp_path / "upstream.sh"
    upstream.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' https://bootstrap.pypa.io/get-pip.py \"$1\"\n"
    )
    wrapper = Path(
        "src/ursa/integrations/harbor_singularity_bootstrap.sh"
    ).resolve()
    env = os.environ | {
        "_URSA_HARBOR_UPSTREAM_BOOTSTRAP": str(upstream),
        "_URSA_HARBOR_PATCHED_BOOTSTRAP": str(tmp_path / "patched.sh"),
    }
    if version is None:
        env["_URSA_HARBOR_SYSTEM_PYTHON"] = str(tmp_path / "missing-python")
    else:
        env["_URSA_HARBOR_PYTHON_VERSION"] = version

    result = subprocess.run(
        ["/bin/sh", wrapper, "argument"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines() == [expected, "argument"]


def test_singularity_bootstrap_installs_python39_distutils(tmp_path):
    upstream = tmp_path / "upstream.sh"
    upstream.write_text("#!/bin/bash\nprintf upstream\\n\n")
    python = tmp_path / "python3"
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    marker = tmp_path / "apt-arguments"
    apt_get = tmp_path / "apt-get"
    apt_get.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >>{marker}\n")
    apt_get.chmod(0o755)
    wrapper = Path(
        "src/ursa/integrations/harbor_singularity_bootstrap.sh"
    ).resolve()
    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "_URSA_HARBOR_PYTHON_VERSION": "3.9",
        "_URSA_HARBOR_SYSTEM_PYTHON": str(python),
        "_URSA_HARBOR_UPSTREAM_BOOTSTRAP": str(upstream),
        "_URSA_HARBOR_PATCHED_BOOTSTRAP": str(tmp_path / "patched.sh"),
    }

    subprocess.run(["/bin/sh", wrapper], check=True, env=env)

    assert marker.read_text().splitlines() == [
        "update -qq",
        "install -y -qq python3-distutils",
    ]


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


def test_runtime_config_does_not_require_unused_openai_secret(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  ollama:\n"
        "    model_provider: ollama\n"
        "    base_url: http://localhost:11434\n"
        "llm_model:\n"
        "  model: ignored\n"
        "  inference_provider: ollama\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="ollama/gemma4:latest",
        config_file=config_file,
    )

    runtime_config, secret_env = agent._runtime_config()

    assert secret_env == {}
    assert "api_key" not in runtime_config["inference_providers"]["openai"]


def test_runtime_config_drops_model_secret_when_switching_provider(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OLD_OPENAI_KEY", raising=False)
    config_file = tmp_path / "ursa.yaml"
    config_file.write_text(
        "inference_providers:\n"
        "  ollama:\n"
        "    model_provider: ollama\n"
        "    base_url: http://localhost:11434\n"
        "llm_model:\n"
        "  model: old-model\n"
        "  model_provider: openai\n"
        "  api_key:\n"
        "    env: OLD_OPENAI_KEY\n"
        "  azure_deployment: old-deployment\n"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="ollama/gemma4:latest",
        config_file=config_file,
    )

    runtime_config, secret_env = agent._runtime_config()

    assert secret_env == {}
    assert "api_key" not in runtime_config["llm_model"]
    assert runtime_config["llm_model"]["inference_provider"] == "ollama"
    assert runtime_config["llm_model"].get("model_provider") is None
    assert "azure_deployment" not in runtime_config["llm_model"]


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

    async def fake_exec_as_agent(*args, **kwargs):
        nonlocal observed_env, observed_timeout
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


@pytest.mark.asyncio
async def test_apptainer_shim_is_used_for_harbor_server(tmp_path, monkeypatch):
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    environment._force_pull = False
    environment._mounts = []
    environment._validate_definition = lambda: None
    environment._runtime = lambda: "/usr/bin/apptainer"

    async def fake_build(force_build):
        return tmp_path / "image.sif"

    async def fake_start_server():
        shim = Path(os.environ["PATH"].split(os.pathsep, 1)[0]) / "singularity"
        assert shim.resolve() == Path("/usr/bin/apptainer")
        assert environment._singularity_no_mount == "home,tmp,bind-paths"
        assert any(
            mount.get("target") == "/etc/resolv.conf"
            for mount in environment._mounts
        )

    async def fake_upload():
        pass

    monkeypatch.setattr(environment, "_build_dockerfile_sif", fake_build)
    monkeypatch.setattr(environment, "_start_server", fake_start_server)
    monkeypatch.setattr(
        environment, "_upload_environment_dir_after_start", fake_upload
    )

    await environment.start(force_build=False)


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


@pytest.mark.asyncio
async def test_singularity_defers_unbounded_exec_timeout_to_harbor(monkeypatch):
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    timeouts = []
    commands = []

    async def fake_exec(
        self, command, cwd=None, env=None, timeout_sec=None, user=None
    ):
        commands.append(command)
        timeouts.append(timeout_sec)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(
        "harbor.environments.singularity.singularity.SingularityEnvironment.exec",
        fake_exec,
    )

    await environment.exec("true")
    await environment.exec("true", timeout_sec=12)

    assert timeouts == [7 * 24 * 60 * 60, 12]
    assert all("</dev/null" in command for command in commands)


@pytest.mark.asyncio
async def test_singularity_exec_closes_inherited_stdin(monkeypatch):
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )

    async def fake_exec(
        self, command, cwd=None, env=None, timeout_sec=None, user=None
    ):
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        finally:
            assert process.stdin is not None
            process.stdin.close()
            if process.returncode is None:
                process.terminate()
                await process.wait()
        assert process.stdout is not None
        stdout = await process.stdout.read()
        return SimpleNamespace(
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr="",
        )

    monkeypatch.setattr(
        "harbor.environments.singularity.singularity.SingularityEnvironment.exec",
        fake_exec,
    )

    result = await environment.exec(
        "read -r value || true; printf done", timeout_sec=1
    )

    assert result.return_code == 0
    assert result.stdout == "done"


@pytest.mark.asyncio
async def test_singularity_cancellation_terminates_remote_process(monkeypatch):
    environment = DockerfileSingularityEnvironment.__new__(
        DockerfileSingularityEnvironment
    )
    started = asyncio.Event()
    commands = []

    async def fake_exec(
        self, command, cwd=None, env=None, timeout_sec=None, user=None
    ):
        commands.append(command)
        if len(commands) == 1:
            started.set()
            await asyncio.Event().wait()
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(
        "harbor.environments.singularity.singularity.SingularityEnvironment.exec",
        fake_exec,
    )
    task = asyncio.create_task(environment.exec("sleep 30"))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(commands) == 2
    assert "descendants" in commands[1]
    assert "kill -TERM" in commands[1]


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
