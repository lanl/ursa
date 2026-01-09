import io
import logging
import re
from pathlib import Path
from sys import executable
from unittest.mock import patch

import pytest
from fastmcp.client import Client
from mcp import StdioServerParameters
from rich.console import Console as RealConsole

from ursa.agents.base import AgentWithTools
from ursa.cli.config import UrsaConfig
from ursa.cli.hitl import HITL, UrsaRepl


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


def test_has_all_agent_do_methods(ursa_config):
    hitl = HITL(ursa_config)
    repl = UrsaRepl(hitl)
    for name in hitl.agents.keys():
        assert hasattr(repl, f"do_{name}")


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
            # ("?execute", re.compile(docstr_header(ExecutionAgent))),
            # ("exit", "Exiting ursa..."),
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


@pytest.mark.parametrize(
    "agent",
    ["chat", "execute", "hypothesize", "plan", "web", "recall"],
)
def test_agent_repl_smoke(ursa_config: UrsaConfig, agent: str):
    if agent == "plan":
        # Planning eats tokens
        ursa_config.llm_model["max_completion_tokens"] = 128000

    trace = check_script(
        ursa_config,
        [(f"{agent} What is your purpose?", None)],
    )
    print(trace)


DUMMY_MCP_SERVER_PATH = Path(__file__).parent.joinpath(
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
        print("Hey")


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
