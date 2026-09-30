import base64
import json
import shlex
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from harbor.models.agent.context import AgentContext
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

from ursa.agents import BaseAgent
from ursa.integrations.harbor import UrsaHarborAgent, _jsonl_token_usage
from ursa.integrations.harbor_runner import _run as _runner_run
from ursa.util.events import AgentEvents


@tool
async def runner_echo(text: str) -> dict[str, str]:
    """Echo text for the runner integration test."""
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
        runnable_config = self.build_config(**config)

        async def lifecycle(
            _inputs: Any, config: RunnableConfig
        ) -> dict[str, Any]:
            await AgentEvents(agent="ExecutionAgent", config=config).aemit(
                "Inspecting the workspace",
                stage="work",
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

        runnable = RunnableLambda(lifecycle).with_config({
            "run_name": "runner_lifecycle"
        })
        return await runnable.ainvoke(inputs, config=runnable_config)

    def format_result(self, result: dict[str, Any]) -> str:
        return str(result["answer"])


class SetupFailingAgent(BaseAgent):
    connection: ClassVar[Any | None] = None

    def __init__(self, *, checkpointer: Any, **kwargs: Any) -> None:
        type(self).connection = checkpointer.conn
        raise RuntimeError("agent setup failed")

    def _build_graph(self) -> None:
        pass


def _runtime_config(path: Path) -> Path:
    path.write_text(
        "llm_model:\n"
        "  model: gpt-4.1-nano\n"
        "  model_provider: openai\n"
        "agent_config:\n"
        "  harbor_lifecycle:\n"
        "    enable_metrics: false\n"
    )
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_harbor_adapter_runs_real_runner_and_populates_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    logs = tmp_path / "agent"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        agent_import_path=f"{__name__}:HarborLifecycleAgent",
    )
    agent._remote_config_file = str(config_file)
    agent._workspace = str(workspace)

    async def run_locally(_environment: Any, **kwargs: Any) -> Any:
        assert kwargs["timeout_sec"] is None
        assert kwargs["cwd"] == str(workspace)
        encoded = shlex.split(kwargs["command"])[-1]
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
        assert payload["log_dir"] == str(logs)
        assert payload["workspace"] == str(workspace)
        await _runner_run(payload)
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "exec_as_agent", run_locally)
    context = AgentContext()

    await agent.run("solve it", object(), context)

    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 6
    assert context.n_output_tokens == 7
    assert (logs / "ursa_result.out").read_text() == "runner answer"
    assert "Inspecting the workspace" in (logs / "ursa.log").read_text()

    records = _records(logs / "ursa.jsonl")
    assert [(record["type"], record["event"]) for record in records] == [
        ("chain", "start"),
        ("agent", "progress"),
        ("chat", "start"),
        ("chat", "end"),
        ("tool", "start"),
        ("tool", "end"),
        ("chain", "end"),
    ]
    assert records[0]["name"] == "runner_lifecycle"
    assert records[3]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "reasoning_tokens": 4,
        "cached_tokens": 6,
    }
    timestamps = [record["timestamp_monotonic_ns"] for record in records]
    assert timestamps == sorted(timestamps)

    checkpoint_path = logs / "db" / "checkpointer.db"
    with sqlite3.connect(checkpoint_path) as database:
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert database.execute(
            "SELECT value FROM runner_probe"
        ).fetchall() == [("persisted",)]


@pytest.mark.asyncio
async def test_harbor_context_keeps_usage_when_runner_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    logs = tmp_path / "agent"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        agent_import_path=f"{__name__}:HarborLifecycleAgent",
    )
    agent._remote_config_file = str(config_file)
    agent._workspace = str(workspace)

    async def run_locally(_environment: Any, **kwargs: Any) -> Any:
        encoded = shlex.split(kwargs["command"])[-1]
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
        await _runner_run(payload)
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "exec_as_agent", run_locally)
    context = AgentContext()

    with pytest.raises(RuntimeError, match="agent failed after chat"):
        await agent.run("fail after usage", object(), context)

    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 6
    assert context.n_output_tokens == 7
    assert not (logs / "ursa_result.out").exists()


@pytest.mark.asyncio
async def test_runner_closes_real_checkpointer_when_agent_setup_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    SetupFailingAgent.connection = None

    with pytest.raises(RuntimeError, match="agent setup failed"):
        await _runner_run({
            "agent_import_path": f"{__name__}:SetupFailingAgent",
            "config_file": str(_runtime_config(tmp_path / "ursa.yaml")),
            "workspace": str(tmp_path / "workspace"),
            "log_dir": str(tmp_path / "agent"),
            "instruction": "solve it",
        })

    assert SetupFailingAgent.connection is not None
    with pytest.raises(ValueError, match="no active connection"):
        await SetupFailingAgent.connection.execute("SELECT 1")


def test_jsonl_token_usage_streams_and_ignores_invalid_records(tmp_path):
    path = tmp_path / "ursa.jsonl"
    path.write_bytes(
        b"\xffnot-json\n"
        b'{"type":"chat","event":"end","usage":{}}\n'
        b'{"type":"chat","event":"end","usage":'
        b'{"input_tokens":true,"cached_tokens":-1,"output_tokens":1.5}}\n'
    )

    assert _jsonl_token_usage(path) == (None, None, None)

    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            '{"type":"chat","event":"end","usage":'
            '{"input_tokens":9007199254740993}}\n'
        )
    assert _jsonl_token_usage(path) == (9007199254740993, None, None)
