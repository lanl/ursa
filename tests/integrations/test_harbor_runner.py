import base64
import json
import shlex
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
from harbor.models.agent.context import AgentContext
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.tools import tool
from langgraph.checkpoint.base import empty_checkpoint
from packaging.requirements import Requirement

from ursa.agents import BaseAgent
from ursa.cli import agent_management
from ursa.integrations import harbor as harbor_integration
from ursa.integrations.harbor import (
    UrsaHarborAgent,
    _jsonl_token_usage,
)
from ursa.integrations.harbor_runner import _run as _runner_run
from ursa.util import Checkpointer
from ursa.util.events import AgentEvents


@tool
async def runner_echo(text: str) -> dict[str, str]:
    """Echo text through the runner's tool lifecycle."""
    return {"echo": text}


class HarborLifecycleAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lifecycle_model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="runner answer",
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 7,
                        "total_tokens": 19,
                    },
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 7,
                            "completion_tokens_details": {
                                "reasoning_tokens": 4
                            },
                            "prompt_tokens_details": {"cached_tokens": 6},
                        }
                    },
                )
            ]
        )

    def _build_graph(self) -> None:
        pass

    async def _ainvoke(self, inputs: Any, **config: Any) -> dict[str, Any]:
        async def lifecycle(
            _inputs: Any, config: RunnableConfig
        ) -> dict[str, Any]:
            await AgentEvents(agent="ExecutionAgent", config=config).aemit(
                "Inspecting the workspace", stage="work"
            )
            answer = await self.lifecycle_model.ainvoke([], config=config)
            if any(
                getattr(message, "content", None) == "fail after usage"
                for message in _inputs.get("messages", [])
            ):
                raise RuntimeError("agent failed after chat")
            echoed = await runner_echo.ainvoke(
                {"text": "needle"}, config=config
            )
            await self.checkpointer.conn.execute(
                "CREATE TABLE runner_probe (value TEXT)"
            )
            await self.checkpointer.conn.execute(
                "INSERT INTO runner_probe VALUES ('persisted')"
            )
            return {"answer": answer.content, "tool": echoed}

        return (
            await RunnableLambda(lifecycle)
            .with_config({"run_name": "runner_lifecycle"})
            .ainvoke(inputs, config=self.build_config(**config))
        )

    def format_result(self, result: dict[str, Any]) -> str:
        return str(result["answer"])


class SetupFailingAgent(BaseAgent):
    connection: ClassVar[Any | None] = None

    def __init__(self, *, checkpointer: Any, **kwargs: Any) -> None:
        type(self).connection = checkpointer.conn
        raise RuntimeError("agent setup failed")

    def _build_graph(self) -> None:
        pass


def _native_config_file(path: Path) -> Path:
    path.write_text(
        "llm_model:\n  model: gpt-4.1-nano\n  model_provider: openai\nagent_config:\n  harbor_lifecycle:\n    enable_metrics: false\n"
    )
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _checkpoint(trial: Path) -> Path:
    path = trial / "agent/db/checkpointer.db"
    checkpointer = Checkpointer.from_workspace(trial / "agent")
    checkpointer.put(
        {"configurable": {"thread_id": "trial-thread", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    checkpointer.conn.close()
    return path


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda command: f"/bin/{command}"
    )


def test_native_config_records_evaluation_group(tmp_path):
    config = _native_config_file(tmp_path / "ursa.yaml")
    config.write_text(config.read_text() + "group: science\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=config,
    )
    runtime, _ = agent._materialize_native_config()
    assert runtime["group"] == agent._runtime_group == "science"


@pytest.mark.parametrize("inline", [False, True])
def test_native_config_accepts_file_and_inline_sources(tmp_path, inline):
    native = {
        "llm_model": {
            "model": "gpt-4.1-nano",
            "model_provider": "openai",
        }
    }
    source: Path | dict[str, Any]
    if inline:
        source = native
    else:
        source = _native_config_file(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=source,
    )

    runtime, _ = agent._materialize_native_config()

    assert agent.capabilities.native_config
    assert runtime["llm_model"]["model"] == "gpt-4.1-nano"


@pytest.mark.parametrize(
    ("native_config", "provider"),
    [
        (
            {
                "inference_providers": {
                    "research-gateway": {
                        "model_provider": "openai",
                        "api_key": {"keyring": True},
                    }
                },
                "llm_model": {
                    "model": "gpt-4.1-nano",
                    "inference_provider": "research-gateway",
                },
            },
            "research-gateway",
        ),
        (
            {
                "llm_model": {
                    "model": "anthropic:claude",
                    "api_key": {"keyring": True},
                }
            },
            "openai",
        ),
    ],
)
def test_native_config_keyring_uses_inference_provider(
    tmp_path, monkeypatch, native_config, provider
):
    calls = []
    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(
            get_password=lambda service, username: (
                calls.append((service, username)) or "key"
            )
        ),
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=native_config,
    )

    _, secret_env = agent._materialize_native_config()

    assert calls == [("ursa", provider)]
    assert secret_env == {"URSA_HARBOR_SECRET_0": "key"}


@pytest.mark.parametrize(
    "source_kind",
    ["implicit", "project", "wheel", "sdist", "version", "git", "url"],
)
@pytest.mark.asyncio
async def test_install_preserves_requirement_and_uploads_local_sources(
    tmp_path, monkeypatch, source_kind
):
    config = _native_config_file(tmp_path / "ursa.yaml")
    checkout = tmp_path / "checkout with spaces"
    (checkout / "pyproject.toml").parent.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'ursa-ai'\n")
    wheel = tmp_path / "ursa_ai-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    sdist = tmp_path / "ursa_ai-1.0.0.tar.gz"
    sdist.write_bytes(b"sdist")
    targets = {
        "implicit": f" @ {checkout.as_uri()}",
        "project": f" @ {checkout.as_uri()}",
        "wheel": f" @ {wheel.as_uri()}#sha256=abc",
        "sdist": f" @ {sdist.as_uri()}",
        "version": ">=0.17.3,<1",
        "git": " @ git+https://github.com/lanl/ursa.git@feature/harbor",
        "url": " @ https://example.org/ursa.whl#sha256=abc",
    }
    spec = f'ursa-ai[office_readers]{targets[source_kind]} ; python_version >= "3.12"'
    monkeypatch.setattr(
        UrsaHarborAgent, "_default_install_spec", staticmethod(lambda: spec)
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=config,
        ursa_install_spec=None if source_kind == "implicit" else spec,
        ursa_extras="image,harbor",
    )
    built: list[Path] = []

    def build(source: Path, destination: Path) -> Path:
        assert source == checkout
        wheel = destination / "ursa_ai-1.0.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        built.append(wheel)
        return wheel

    class Environment:
        def __init__(self) -> None:
            self.files: list[tuple[Path, str]] = []
            self.directories: list[str] = []

        async def upload_file(self, source: Path, destination: str) -> None:
            self.files.append((source, destination))

        async def upload_dir(self, source: Path, destination: str) -> None:
            assert (source / "pyproject.toml").read_text() == (
                checkout / "pyproject.toml"
            ).read_text()
            self.directories.append(destination)

    environment = Environment()
    monkeypatch.setattr(
        agent, "_materialize_native_config", lambda **_kwargs: ({}, {})
    )
    monkeypatch.setattr(agent, "_network_use_web", lambda _env: None)
    monkeypatch.setattr(
        agent,
        "exec_as_agent",
        AsyncMock(return_value=SimpleNamespace(stdout="/")),
    )
    root = AsyncMock(return_value=SimpleNamespace(stdout="x86_64"))
    monkeypatch.setattr(agent, "exec_as_root", root)
    monkeypatch.setattr(agent, "_upload_native_config", AsyncMock())
    monkeypatch.setattr(agent, "_build_wheel", build)
    monkeypatch.setattr(
        agent,
        "_stage_uv_binary",
        lambda _machine, destination: destination.write_bytes(b"uv"),
    )

    await agent.install(environment)

    tokens = shlex.split(root.call_args_list[-1].kwargs["command"])
    installed = Requirement(tokens[tokens.index("&&") - 1])
    original = Requirement(spec)
    assert installed.extras == {"harbor", "image", "office-readers"}
    assert installed.marker == original.marker
    assert installed.specifier == original.specifier
    assert str(agent.ursa_install_spec) == str(original)
    assert len(built) == (source_kind == "implicit")
    assert environment.directories == (
        ["/installed-agent/tmp/ursa-source"] if source_kind == "project" else []
    )
    if source_kind in {"implicit", "wheel", "sdist"}:
        filename = sdist.name if source_kind == "sdist" else wheel.name
        destination = f"/installed-agent/tmp/{filename}"
        fragment = "#sha256=abc" if source_kind == "wheel" else ""
        assert installed.url == f"file://{destination}{fragment}"
        assert any(remote == destination for _, remote in environment.files)
    elif source_kind == "project":
        assert installed.url == "file:///installed-agent/tmp/ursa-source"
    elif source_kind == "git":
        assert installed.url == (
            "https://github.com/lanl/ursa/archive/feature%2Fharbor.tar.gz"
        )
    else:
        assert installed.url == original.url


@pytest.mark.parametrize(
    "value",
    [
        "",
        "./checkout",
        "/tmp/ursa.whl",
        "git+https://github.com/lanl/ursa.git",
        "ursa-ai=>1",
        Path("checkout"),
    ],
)
def test_install_spec_requires_pep508(value):
    with pytest.raises(ValueError, match="PEP 508"):
        UrsaHarborAgent._parse_install_spec(value)


def test_install_spec_requires_ursa_package():
    with pytest.raises(ValueError, match="ursa-ai package"):
        UrsaHarborAgent._parse_install_spec("another-package>=1")


def test_install_spec_validates_local_sources(tmp_path):
    missing = tmp_path / "missing.whl"
    with pytest.raises(ValueError, match="does not exist"):
        UrsaHarborAgent._local_install_source(
            Requirement(f"ursa-ai @ {missing.as_uri()}")
        )
    with pytest.raises(ValueError, match="not a Python project"):
        UrsaHarborAgent._local_install_source(
            Requirement(f"ursa-ai @ {tmp_path.as_uri()}")
        )
    for url in ("file:relative/path", "file://remote-host/path"):
        with pytest.raises(ValueError, match="absolute local file URL"):
            UrsaHarborAgent._local_install_source(
                Requirement(f"ursa-ai @ {url}")
            )


def test_handoff_imports_checkpoint_as_local_named_agent(
    tmp_path, monkeypatch, cli
):
    cache, trial = tmp_path / "cache", tmp_path / "install pytorch__abc123"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", cache)
    (cache / "science").mkdir(parents=True)
    source = _checkpoint(trial)
    (trial / "result.json").write_text(
        json.dumps({"agent_result": {"metadata": {"group": "science"}}})
    )
    assert UrsaHarborAgent.handoff(trial, tmp_path / "workspace") == [
        "ursa",
        "--name",
        "harbor-install-pytorch__abc123",
        "--group",
        "science",
    ]
    copied = (
        cache
        / "science/agents/harbor-install-pytorch__abc123/db/checkpointer.db"
    )
    with sqlite3.connect(copied) as db:
        assert db.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall() == [("ursa",)]
    with sqlite3.connect(source) as db:
        assert db.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall() == [("trial-thread",)]


def test_handoff_validates_cli_checkpoint_group_and_destination(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda _command: None
    )
    with pytest.raises(ValueError, match="ursa CLI not found"):
        UrsaHarborAgent.handoff(tmp_path / "trial", tmp_path)
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda command: f"/bin/{command}"
    )
    with pytest.raises(ValueError, match="URSA checkpoint not found"):
        UrsaHarborAgent.handoff(tmp_path / "trial", tmp_path)
    trial = tmp_path / "trial"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", tmp_path / "cache")
    _checkpoint(trial)
    with pytest.raises(
        ValueError, match="does not record the URSA evaluation group"
    ):
        UrsaHarborAgent.handoff(trial, tmp_path)
    (trial / "result.json").write_text(
        json.dumps({"agent_result": {"metadata": {"group": "default"}}})
    )
    UrsaHarborAgent.handoff(trial, tmp_path)
    with pytest.raises(ValueError, match="Destination agent already exists"):
        UrsaHarborAgent.handoff(trial, tmp_path)


def _lifecycle_agent(tmp_path):
    logs, workspace, config = (
        tmp_path / "agent",
        tmp_path / "workspace",
        _native_config_file(tmp_path / "ursa.yaml"),
    )
    workspace.mkdir()
    agent = UrsaHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="openai/gpt-4.1-nano",
        config=config,
        agent_import_path=f"{__name__}:HarborLifecycleAgent",
    )
    agent._remote_config_file, agent._workspace = str(config), str(workspace)
    return agent, logs, workspace


async def _local_runner(agent, *, check_payload=False):
    async def run(_environment, **kwargs):
        if check_payload:
            assert (
                kwargs["timeout_sec"] is None
                and kwargs["cwd"] == agent._workspace
            )
        payload = json.loads(
            base64.urlsafe_b64decode(
                shlex.split(kwargs["command"])[-1]
            ).decode()
        )
        if check_payload:
            assert (
                payload["log_dir"] == str(agent.logs_dir)
                and payload["workspace"] == agent._workspace
            )
        await _runner_run(payload)
        return SimpleNamespace(stdout=None)

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("instruction", "error"),
    [
        ("solve it", None),
        ("fail after usage", "agent failed after chat"),
    ],
)
async def test_adapter_runner_preserves_usage(
    tmp_path, monkeypatch, instruction, error
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent, logs, _ = _lifecycle_agent(tmp_path)
    monkeypatch.setattr(
        agent, "exec_as_agent", await _local_runner(agent, check_payload=True)
    )
    context = AgentContext()
    if error:
        with pytest.raises(RuntimeError, match=error):
            await agent.run(instruction, object(), context)
    else:
        await agent.run(instruction, object(), context)
    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
    ) == (12, 6, 7)
    result = logs / "ursa_result.out"
    if error:
        assert not result.exists()
        return
    assert context.metadata["group"] == "default"
    assert (
        result.read_text() == "runner answer"
        and "Inspecting the workspace" in (logs / "ursa.log").read_text()
    )
    records = _records(logs / "ursa.jsonl")
    events = {(record["type"], record["event"]) for record in records}
    assert {
        ("chain", "start"),
        ("agent", "progress"),
        ("chat", "end"),
        ("tool", "end"),
        ("chain", "end"),
    } <= events
    assert next(
        record
        for record in records
        if record["type"] == "chat" and record["event"] == "end"
    )["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "reasoning_tokens": 4,
        "cached_tokens": 6,
    }
    assert [record["timestamp_monotonic_ns"] for record in records] == sorted(
        record["timestamp_monotonic_ns"] for record in records
    )
    with sqlite3.connect(logs / "db/checkpointer.db") as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == (
            "ok",
        ) and db.execute("SELECT value FROM runner_probe").fetchall() == [
            ("persisted",)
        ]


@pytest.mark.asyncio
async def test_runner_closes_setup_connection_on_agent_setup_failure(
    tmp_path, monkeypatch
):
    SetupFailingAgent.connection = None
    with pytest.raises(RuntimeError, match="agent setup failed"):
        await _runner_run({
            "agent_import_path": f"{__name__}:SetupFailingAgent",
            "config_file": str(_native_config_file(tmp_path / "bad.yaml")),
            "workspace": str(tmp_path / "workspace"),
            "log_dir": str(tmp_path / "failed"),
            "instruction": "solve it",
        })
    with pytest.raises(ValueError, match="no active connection"):
        await SetupFailingAgent.connection.execute("SELECT 1")


def test_jsonl_token_usage_ignores_invalid_records(tmp_path):
    path = tmp_path / "ursa.jsonl"
    path.write_bytes(
        b'\xffnot-json\n{"type":"chat","event":"end","usage":{}}\n{"type":"chat","event":"end","usage":{"input_tokens":true,"cached_tokens":-1,"output_tokens":1.5}}\n'
    )
    assert _jsonl_token_usage(path) == (None, None, None)
    path.write_text(
        path.read_text(errors="ignore")
        + '{"type":"chat","event":"end","usage":{"input_tokens":9007199254740993}}\n'
    )
    assert _jsonl_token_usage(path) == (9007199254740993, None, None)
