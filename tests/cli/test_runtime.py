# ruff: noqa: TID251

import asyncio
import io
import logging
from pathlib import Path
from random import random
from sys import executable
from unittest.mock import MagicMock

import pytest
from fastmcp.client import Client
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import StdioServerParameters
from pydantic import ValidationError
from rich.console import Console as RealConsole

from ursa.agents.base import AgentWithTools
from ursa.cli.callbacks import HITLLogEventHandler
from ursa.cli.config import EmbModelConfig, UrsaConfig
from ursa.cli.runtime import HITL, AgentHITL
from ursa.util.events import DEFAULT_EVENT_NAME
from ursa.util.has_optional_dep_group import has_optional_dep_group
from ursa.util.rendering import event_artifact

LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def stub_duckduckgo(monkeypatch):
    class DummyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, *args, **kwargs):
            yield {
                "href": "https://example.com",
                "title": "Example Result",
                "body": "Example summary",
            }

    monkeypatch.setattr(
        "ursa.agents.acquisition_agents.DDGS",
        lambda: DummyDDGS(),
        raising=False,
    )
    monkeypatch.setattr(
        "ursa.agents.hypothesizer_agent.DDGS",
        lambda: DummyDDGS(),
        raising=False,
    )


@pytest.fixture(scope="function")
def ursa_config(tmpdir, chat_model, embedding_model):
    config = UrsaConfig(
        workspace=Path(tmpdir),
        llm_model=chat_model._testing_only_kwargs,
        emb_model=embedding_model._testing_only_kwargs,
    )
    print("ursa config:", config)  # Displayed on test failure
    return config


async def test_default_config_smoke(ursa_config):
    hitl = HITL(ursa_config)
    assert hitl is not None
    assert set(hitl.agents.keys()) >= {"chat", "plan", "execute"}
    out = await hitl.run_agent("chat", "Hello! What is your name?")
    print("chat out:", out)
    assert len(out) > 0


DOCS_ROOT = Path(__file__).resolve().parents[2]
DOC_EXAMPLE_CONFIG = DOCS_ROOT / "configs" / "example.yaml"


async def test_agents_use_configured_workspace(ursa_config, tmp_path):
    workspace = tmp_path / "custom-workspace"
    ursa_config.workspace = workspace

    hitl = HITL(ursa_config)
    agent = await hitl.get_agent("chat")
    assert agent._agent is not None
    assert agent._agent.workspace == workspace


@pytest.mark.asyncio
async def test_unnamed_cli_agent_does_not_create_checkpointer(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    workspace = tmp_path / "ephemeral-workspace"
    hitl = HITL(UrsaConfig(workspace=workspace))

    async def unexpected_checkpointer(_checkpoint_path):
        pytest.fail("Unnamed CLI sessions must not create a checkpointer")

    monkeypatch.setattr(hitl, "_get_checkpointer", unexpected_checkpointer)

    agent = await hitl.get_agent("chat")

    assert agent._agent is not None
    assert agent._agent.checkpointer is None
    assert not (workspace / "db" / "checkpointer.db").exists()


@pytest.mark.asyncio
async def test_named_cli_agent_still_gets_async_checkpointer(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    workspace = tmp_path / "workspace"
    hitl = HITL(UrsaConfig(workspace=workspace, agent_name="persistent-agent"))
    persistent_den = tmp_path / "persistent-agent-den"
    expected_checkpointer = object()
    requested_paths = []

    class DummyPersistentAgent:
        def __init__(self, **_kwargs):
            self.den = persistent_den
            self.checkpointer = None

    async def fake_get_checkpointer(checkpoint_path):
        requested_paths.append(checkpoint_path)
        return expected_checkpointer

    hitl.agents["chat"] = AgentHITL(agent_class=DummyPersistentAgent)
    monkeypatch.setattr(hitl, "_get_checkpointer", fake_get_checkpointer)

    agent = await hitl.get_agent("chat")

    assert agent._agent is not None
    assert requested_paths == [persistent_den]
    assert agent._agent.checkpointer is expected_checkpointer


@pytest.mark.asyncio
async def test_named_cli_agent_resources_close_once(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(
        UrsaConfig(
            workspace=tmp_path / "workspace",
            agent_name="persistent-agent",
        )
    )
    persistent_den = tmp_path / "persistent-agent-den"

    class DummyPersistentAgent:
        def __init__(self, **_kwargs):
            self.den = persistent_den
            self.checkpointer = None
            self.async_close_count = 0
            self.close_count = 0

        async def aclose(self):
            self.async_close_count += 1

        def close(self):
            self.close_count += 1

    hitl.agents["chat"] = AgentHITL(agent_class=DummyPersistentAgent)
    wrapper = await hitl.get_agent("chat")
    assert wrapper._agent is not None
    checkpointer = wrapper._agent.checkpointer
    assert isinstance(checkpointer, AsyncSqliteSaver)
    assert checkpointer.conn.is_alive()

    await hitl.close()
    await hitl.aclose()

    assert wrapper._agent.async_close_count == 1
    assert wrapper._agent.close_count == 1
    assert checkpointer.conn._connection is None
    assert not checkpointer.conn.is_alive()


def _stub_hitl_dependencies(monkeypatch):
    fake_llm = MagicMock(name="llm")
    fake_embedding = MagicMock(name="embedding")
    monkeypatch.setattr("ursa.cli.config.init_chat_model", lambda **_: fake_llm)
    monkeypatch.setattr(
        "ursa.cli.config.init_embeddings", lambda **_: fake_embedding
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.start_mcp_client", lambda servers: None
    )
    return fake_llm, fake_embedding


@pytest.mark.parametrize(
    "agent_name",
    [
        "chat",
        "arxiv",
        "dsi",
        "execute",
        "hypothesize",
        "plan",
        "web",
    ]
    + ["dsi"]
    if has_optional_dep_group("dsi")
    else [],
)
async def test_agents_apply_agent_config_overrides(
    agent_name, tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)

    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding"),
    )

    overrides = {}
    overrides[agent_name] = {
        "workspace": tmp_path / f"{agent_name}-workspace",
        "enable_metrics": random() > 0.5,
    }

    config.agent_config = overrides

    hitl = HITL(config)

    agent = await hitl.get_agent(agent_name)
    override = overrides[agent_name]
    assert agent._agent is not None
    assert agent._agent.workspace == override["workspace"]
    assert agent._agent.telemetry.enable == override["enable_metrics"]


@pytest.mark.asyncio
async def test_thread_id_propagates_from_config(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        thread_id="custom-thread",
        emb_model=EmbModelConfig(model="fake-embedding"),
    )

    hitl = HITL(config)
    assert hitl.thread_id == "custom-thread"

    agent = await hitl.get_agent("chat")
    assert agent._agent is not None
    assert agent._agent.thread_id == "custom-thread"


@pytest.mark.asyncio
async def test_hitl_run_agent_forwards_callbacks(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding"),
    )
    hitl = HITL(config)
    captured = {}
    previous_agent = object()
    current_agent = object()

    class DummyAgent:
        _agent = current_agent

        async def __call__(
            self,
            prompt: str,
            last_agent_result: str | None = None,
            last_agent=None,
            callbacks=None,
        ) -> str:
            captured["prompt"] = prompt
            captured["last_agent_result"] = last_agent_result
            captured["last_agent"] = last_agent
            captured["callbacks"] = callbacks
            return "agent result"

    async def fake_get_agent(name: str):
        assert name == "chat"
        return DummyAgent()

    hitl.last_agent_result = "previous result"
    hitl.last_agent = previous_agent
    monkeypatch.setattr(hitl, "get_agent", fake_get_agent)

    callbacks = ["callback-1"]
    result = await hitl.run_agent("chat", "hello", callbacks=callbacks)

    assert result == "agent result"
    assert captured == {
        "prompt": "hello",
        "last_agent_result": "previous result",
        "last_agent": previous_agent,
        "callbacks": callbacks,
    }
    assert hitl.last_agent_result == "agent result"
    assert hitl.last_agent is current_agent


@pytest.mark.asyncio
async def test_agent_hitl_passes_extra_callbacks_only():
    captured = {}
    custom_callback = object()

    class DummyAgent:
        telemetry = type("Telemetry", (), {"callbacks": ["telemetry"]})()

        def format_query(self, prompt: str, state=None):
            captured["prompt"] = prompt
            captured["state"] = state
            return {"messages": [prompt]}

        async def ainvoke(self, query, config=None):
            captured["query"] = query
            captured["config"] = config
            return {"messages": ["done"]}

        def format_result(self, result):
            return "done"

    wrapper = AgentHITL(agent_class=object)
    wrapper._agent = DummyAgent()

    result = await wrapper("hello", callbacks=[custom_callback])

    assert result == "done"
    assert captured["prompt"] == "hello"
    assert captured["state"] is None
    assert captured["query"] == {"messages": ["hello"]}
    assert captured["config"] == {"callbacks": [custom_callback]}


def test_hitl_log_event_handler_renders_events(tmp_path):
    output = io.StringIO()
    console = RealConsole(
        file=output,
        force_terminal=False,
        force_interactive=False,
        color_system=None,
        width=80,
    )
    handler = HITLLogEventHandler(console=console, workspace=tmp_path)

    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "PlanningAgent",
                "stage": "reflect_result",
                "message": "Plan needs another pass",
                "approved": False,
                "reason": "Need one more concrete step.",
            },
            run_id="agent-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "phase": "end",
                "message": "File updated",
                "artifact": event_artifact(
                    "--- app.py\n+++ app.py\n-old\n+new",
                    "text/x-diff",
                    metadata={
                        "title": "Edit diff",
                        "path": "repo/app.py",
                    },
                ),
            },
            run_id="edit-tool-artifact-run",
        )
    )
    asyncio.run(
        handler.on_tool_start(
            {"name": "run_command"},
            '{"query":"uname -s"}',
            run_id="tool-run",
            inputs={"query": "uname -s"},
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_command",
                "stage": "execute",
                "phase": "end",
                "message": "Command finished",
                "artifacts": [
                    event_artifact(
                        "Darwin",
                        "text/plain",
                        metadata={"title": "stdout"},
                    ),
                    event_artifact(
                        "warning",
                        "text/plain",
                        metadata={"title": "stderr"},
                    ),
                ],
            },
            run_id="tool-run",
        )
    )
    asyncio.run(
        handler.on_tool_end(
            "STDOUT:\nDarwin\nSTDERR:\n",
            run_id="tool-run",
        )
    )
    asyncio.run(
        handler.on_tool_start(
            {"name": "write_code_with_repo"},
            '{"filename":"repo/app.py"}',
            run_id="write-tool-run",
            inputs={"filename": "repo/app.py"},
        )
    )
    asyncio.run(
        handler.on_tool_end(
            "File repo/app.py written successfully.",
            run_id="write-tool-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "phase": "start",
                "message": "Editing file",
                "path": str(tmp_path / "repo" / "app.py"),
            },
            run_id="edit-tool-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "message": "No changes made",
                "filename": "repo/app.py",
                "reason": "'old_code' not found in file.",
            },
            run_id="edit-tool-noop-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_web_search",
                "stage": "search",
                "message": "Searching Web",
                "query": "ursa events",
            },
            run_id="search-tool-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_web_search",
                "stage": "search_result",
                "message": "Web search complete",
                "query": "ursa events",
                "result_chars": 42,
            },
            run_id="search-tool-result-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "choose_potential",
                "phase": "end",
                "message": "Potential chosen",
                "chosen_index": 2,
                "potential_id": "pot-2",
                "rationale": "Best fit for the requested elements.",
            },
            run_id="lammps-choice-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "author_input",
                "phase": "end",
                "message": "LAMMPS input authored",
                "preview": "units metal\nrun 100",
                "language": "bash",
                "path": str(tmp_path / "in.lammps"),
            },
            run_id="lammps-author-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "fix_input",
                "phase": "end",
                "message": "LAMMPS input rewritten",
                "old_code": "run 100",
                "new_code": "run 200",
                "path": str(tmp_path / "in.lammps"),
            },
            run_id="lammps-fix-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "run",
                "phase": "error",
                "message": "LAMMPS run failed",
                "returncode": 1,
                "error_output": "ERROR: Invalid pair style",
            },
            run_id="lammps-failed-run",
        )
    )

    rendered = output.getvalue()

    assert "Edit diff" in rendered
    assert "-old" in rendered
    assert "+new" in rendered

    repo_app_string = str(
        Path("repo") / "app.py"
    )  # Rendering OS specific path string

    assert "Plan" in rendered
    assert "Plan needs another pass" in rendered
    assert "Need one more concrete step." in rendered
    assert "Running command: uname -s" in rendered
    assert "Command finished: uname -s" in rendered
    assert "stdout" in rendered
    assert "stderr" in rendered
    assert "warning" in rendered
    assert any(
        "stdout" in line and "stderr" in line for line in rendered.splitlines()
    )
    assert "Darwin" in rendered
    assert f"Writing file: {repo_app_string}" in rendered
    assert f"File written: {repo_app_string}" in rendered
    assert f"Editing file: {repo_app_string}" in rendered
    assert f"No changes made: {repo_app_string}" in rendered
    assert "'old_code' not found in file." in rendered
    assert "Searching Web: ursa events" in rendered
    assert "Web search complete: ursa events" in rendered
    assert "42 chars" in rendered
    assert "LAMMPS" in rendered
    assert "Chosen Potential" in rendered
    assert "pot-2" in rendered
    assert "Best fit for the requested elements." in rendered
    assert "LAMMPS input authored" in rendered
    assert "units metal" in rendered
    assert "LAMMPS input diff" in rendered
    assert "run 100" in rendered
    assert "run 200" in rendered
    assert "Run error/output" in rendered
    assert "ERROR: Invalid pair style" in rendered


def test_hitl_log_event_handler_renders_named_agent_tool_artifacts(tmp_path):
    output = io.StringIO()
    console = RealConsole(
        file=output,
        force_terminal=False,
        force_interactive=False,
        color_system=None,
        width=80,
    )
    handler = HITLLogEventHandler(console=console, workspace=tmp_path)

    async def emit_events() -> None:
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "dummy_bot_3000",
                "tool": "write_code",
                "stage": "write",
                "phase": "end",
                "message": "File written",
                "filename": "first_10_integers.py",
                "artifact": event_artifact(
                    "for i in range(1, 11):\n    print(i)\n",
                    "text/x-python",
                    metadata={"title": "File written"},
                ),
            },
            run_id="named-write-tool-run",
        )
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "dummy_bot_3000",
                "tool": "run_command",
                "stage": "execute",
                "phase": "end",
                "message": "Command finished",
                "query": "python first_10_integers.py",
                "artifacts": [
                    event_artifact(
                        "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
                        "text/plain",
                        metadata={"title": "stdout"},
                    )
                ],
            },
            run_id="named-command-tool-run",
        )

    asyncio.run(emit_events())

    rendered = output.getvalue()
    assert "File written" in rendered
    assert "for i in range(1, 11):" in rendered
    assert "print(i)" in rendered
    assert "stdout" in rendered
    assert "1" in rendered
    assert "10" in rendered


def test_agent_config_unknown_agent_raises(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding"),
    )
    config.agent_config = {
        "ghost": {"workspace": tmp_path / "ghost-workspace"},
    }

    with pytest.raises(AssertionError, match="Unknown agent ghost"):
        HITL(config)


def test_agent_config_none_value_errors(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding"),
    )
    with pytest.raises(ValidationError):
        config.agent_config = {"chat": None}


@pytest.mark.asyncio
async def test_agent_config_unknown_option_raises(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding"),
    )
    config.agent_config = {"chat": {"nonexistent_option": True}}

    hitl = HITL(config)

    with pytest.raises(TypeError, match="nonexistent_option"):
        await hitl.get_agent("chat")


async def test_chat(ursa_config):
    hitl = HITL(ursa_config)
    out = await hitl.run_agent(
        "chat",
        "What is your name?",
    )
    print(out)
    assert out is not None


DUMMY_MCP_SERVER_PATH = Path(__file__).parent.parent.joinpath(
    "tools", "dummy_mcp_server.py"
)


async def test_mcp_tools(ursa_config: UrsaConfig):
    ursa_config.mcp_servers["demo"] = StdioServerParameters(
        command=executable,
        args=[str(DUMMY_MCP_SERVER_PATH.resolve())],
    )
    hitl = HITL(ursa_config)
    agent = await hitl.get_agent("execute")
    assert agent._agent is not None
    assert isinstance(agent._agent, AgentWithTools)
    assert "add" in agent._agent.tools
    assert agent.tool_sources["add"] == "demo"


@pytest.fixture
async def mcp_server(ursa_config):
    hitl = HITL(ursa_config)
    server = hitl.as_mcp_server()
    async with Client(transport=server) as client:
        yield client


async def test_mcp_smoke(mcp_server: Client):
    tools = await mcp_server.list_tools()
    assert len(tools) > 0
    await mcp_server.list_resources()
    await mcp_server.list_prompts()


@pytest.mark.parametrize("agent,query", [("chat", "Who are you?")])
async def test_mcp_agents(mcp_server: Client, agent: str, query: str):
    response = await mcp_server.call_tool(agent, {"prompt": query})
    assert isinstance(response.structured_content["result"], str)
