import asyncio
import threading
import time
from types import SimpleNamespace

from ursa.cli.tui.agent_info import load_agent_details, load_agent_tools


class ToolArgs:
    @classmethod
    def model_json_schema(cls):
        return {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }


class Tool:
    name = "search"
    description = "Search configured sources."
    args_schema = ToolArgs
    return_direct = True


async def test_agent_details_preserve_runtime_order_without_loading_agents():
    wrappers = {
        "execute": SimpleNamespace(
            description="Execute work.",
            config={"mode": "safe"},
            tool_sources={"search": "laboratory"},
            _agent=None,
        ),
        "chat": SimpleNamespace(
            description="Answer questions.",
            config={},
            tool_sources={},
            _agent=None,
        ),
    }
    calls = []

    async def get_agent(name):
        calls.append(name)
        wrapper = wrappers[name]
        wrapper._agent = SimpleNamespace(tools={"search": Tool()})
        return wrapper

    hitl = SimpleNamespace(agents=wrappers, get_agent=get_agent)

    details = load_agent_details(hitl)

    assert [agent.name for agent in details] == ["execute", "chat"]
    assert calls == []
    assert details[0].config == (("mode", "safe"),)
    assert details[0].tools == ()
    assert not details[0].tools_loaded

    tools = await load_agent_tools(hitl, "execute")

    assert calls == ["execute"]
    assert [tool.name for tool in tools] == ["search"]
    assert tools[0].arguments[0].name == "query"
    assert tools[0].mcp_server == "laboratory"


async def test_agent_details_expose_tools_from_initialized_agent():
    wrapper = SimpleNamespace(
        description="Execute work.",
        config={},
        tool_sources={},
        _agent=SimpleNamespace(tools={"search": Tool()}),
    )
    calls = []

    async def get_agent(name):
        calls.append(name)
        return wrapper

    details = load_agent_details(
        SimpleNamespace(agents={"execute": wrapper}, get_agent=get_agent)
    )

    assert calls == []
    assert not details[0].tools_loaded
    assert [tool.name for tool in details[0].tools] == ["search"]
    assert details[0].tools[0].arguments == ()

    tools = await load_agent_tools(
        SimpleNamespace(get_agent=get_agent), "execute"
    )
    assert calls == ["execute"]
    assert [tool.name for tool in tools] == ["search"]


def test_initialized_agent_snapshot_does_not_generate_tool_schemas():
    class UnexpectedSchema:
        @classmethod
        def model_json_schema(cls):
            raise AssertionError("schema generation must be deferred")

    tool = Tool()
    tool.args_schema = UnexpectedSchema
    wrapper = SimpleNamespace(
        description="Initialized agent",
        config={},
        tool_sources={},
        _agent=SimpleNamespace(tools={"search": tool}),
    )

    details = load_agent_details(SimpleNamespace(agents={"execute": wrapper}))

    assert not details[0].tools_loaded
    assert [tool.name for tool in details[0].tools] == ["search"]


async def test_agent_tool_schema_conversion_runs_off_event_loop():
    event_loop_thread = threading.get_ident()
    schema_threads = []

    class SlowSchema:
        @classmethod
        def model_json_schema(cls):
            schema_threads.append(threading.get_ident())
            time.sleep(0.05)
            return {"properties": {}}

    tool = Tool()
    tool.args_schema = SlowSchema
    wrapper = SimpleNamespace(
        _agent=SimpleNamespace(tools={"search": tool}),
        tool_sources={},
    )

    async def get_agent(_name):
        return wrapper

    ticks = 0
    loading = True

    async def ticker():
        nonlocal ticks
        while loading:
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    await load_agent_tools(SimpleNamespace(get_agent=get_agent), "execute")
    loading = False
    await ticker_task

    assert schema_threads[0] != event_loop_thread
    assert ticks > 1
