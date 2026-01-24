import io
import logging
import re
from pathlib import Path
from random import random
from sys import executable
from unittest.mock import MagicMock, patch

import pytest
from fastmcp.client import Client
from mcp import StdioServerParameters
from pydantic import ValidationError
from rich.console import Console as RealConsole

from ursa.agents.base import AgentWithTools
from ursa.cli.config import ModelConfig, UrsaConfig
from ursa.cli.hitl import HITL, UrsaRepl


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
        "ursa.agents.acquisition_agents.DDGS", lambda: DummyDDGS(), raising=False
    )
    monkeypatch.setattr(
        "ursa.agents.hypothesizer_agent.DDGS", lambda: DummyDDGS(), raising=False
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


def test_example_config_smoke():
    assert DOC_EXAMPLE_CONFIG.is_file()
    ursa_config = UrsaConfig.from_file(DOC_EXAMPLE_CONFIG)
    hitl = HITL(ursa_config)
    repl = UrsaRepl(hitl)
    for name in hitl.agents.keys():
        assert hasattr(repl, f"do_{name}")


def test_has_all_agent_do_methods(ursa_config):
    hitl = HITL(ursa_config)
    repl = UrsaRepl(hitl)
    for name in hitl.agents.keys():
        assert hasattr(repl, f"do_{name}")


async def test_agents_use_configured_workspace(ursa_config, tmp_path):
    workspace = tmp_path / "custom-workspace"
    ursa_config.workspace = workspace

    hitl = HITL(ursa_config)
    agent = await hitl.get_agent("chat")
    assert agent._agent is not None
    assert agent._agent.workspace == workspace


def _stub_hitl_dependencies(monkeypatch):
    fake_llm = MagicMock(name="llm")
    fake_embedding = MagicMock(name="embedding")
    monkeypatch.setattr("ursa.cli.hitl.init_chat_model", lambda **_: fake_llm)
    monkeypatch.setattr(
        "ursa.cli.hitl.init_embeddings", lambda **_: fake_embedding
    )
    monkeypatch.setattr("ursa.cli.hitl.start_mcp_client", lambda servers: None)
    return fake_llm, fake_embedding


@pytest.mark.parametrize(
    "agent_name",
    [
        "chat",
        "arxiv",
        "execute",
        "hypothesize",
        "plan",
        "web",
        "recall",
    ],
)
async def test_agents_apply_agent_config_overrides(
    agent_name, tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)

    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=ModelConfig(model="fake-embedding"),
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
        emb_model=ModelConfig(model="fake-embedding"),
    )

    hitl = HITL(config)
    assert hitl.thread_id == "custom-thread"

    agent = await hitl.get_agent("chat")
    assert agent._agent is not None
    assert agent._agent.thread_id == "custom-thread_chat"


def test_agent_config_unknown_agent_raises(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=ModelConfig(model="fake-embedding"),
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
        emb_model=ModelConfig(model="fake-embedding"),
    )
    with pytest.raises(ValidationError):
        config.agent_config = {"chat": None}


@pytest.mark.asyncio
async def test_agent_config_unknown_option_raises(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=ModelConfig(model="fake-embedding"),
    )
    config.agent_config = {"chat": {"nonexistent_option": True}}

    hitl = HITL(config)

    with pytest.raises(TypeError, match="nonexistent_option"):
        await hitl.get_agent("chat")


def check_script(
    ursa_config: UrsaConfig,
    input_expected: list[tuple[str, str | int | re.Pattern | None]],
):
    stdout = io.StringIO()
    stdout_pos = 0

    def console_factory(*args, **kwargs):
        kwargs["record"] = True
        kwargs["force_terminal"] = False
        kwargs["force_interactive"] = False
        return RealConsole(*args, **kwargs)

    # Patch the Console constructor so we can snoop
    with patch("ursa.cli.hitl.Console", new=console_factory):
        shell = UrsaRepl(HITL(ursa_config), stdout=stdout)

    # Feed the REPL with the script and check the output matches
    # expectations
    trace = []
    for input, ref in input_expected:
        logging.info(f"input: {input}")
        shell.onecmd(input)
        console_output = shell.console.export_text()
        stdout_value = stdout.getvalue()
        stdout_delta = stdout_value[stdout_pos:]
        stdout_pos = len(stdout_value)
        output = stdout_delta or console_output
        logging.info(f"output: {output}")
        match ref:
            case str():
                assert output == ref
            case int():
                assert len(output.strip()) >= ref
            case re.Pattern():
                assert ref.search(output) is not None
            case None:
                pass
            case _:
                assert False, f"Unknown reference type: {ref}"

        trace.append({"input": input, "output": output})

    return trace


def test_repl_smoke(ursa_config):
    def docstr_header(cls) -> str:
        docs = cls.__doc__
        assert isinstance(docs, str)
        return docs.split("\n", maxsplit=1)[0]

    trace = check_script(
        ursa_config,
        [
            ("What is your name?", None),
            ("help", re.compile(r".*Documented commands")),
            ("?", re.compile(r".*Documented commands")),
            ("agents", re.compile(r".*chat:")),
            ("exit", re.compile(r".*Exiting ursa")),
        ],
    )
    print(trace)


async def test_chat(ursa_config):
    hitl = HITL(ursa_config)
    out = await hitl.run_agent(
        "chat",
        "What is your name?",
    )
    print(out)
    assert out is not None


@pytest.mark.slow
@pytest.mark.parametrize(
    "agent",
    ["chat", "execute", "hypothesize", "plan", "web", "recall"],
)
def test_agent_repl_smoke(ursa_config: UrsaConfig, agent: str):
    if agent == "plan":
        # Planning eats tokens
        ursa_config.llm_model.max_completion_tokens = 128000

    trace = check_script(
        ursa_config,
        [(f"{agent} What is your purpose?", None)],
    )
    print(trace)


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


@pytest.mark.parametrize(
    "agent,query", [("chat", "Who are you?"), ("recall", "Who am I?")]
)
async def test_mcp_agents(mcp_server: Client, agent: str, query: str):
    response = await mcp_server.call_tool(agent, {"prompt": query})
    assert isinstance(response.structured_content["result"], str)
