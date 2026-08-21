from types import SimpleNamespace

from ursa.cli.agent_info import load_agent_details


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


async def test_agent_details_preserve_runtime_order_and_load_actual_tools():
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

    details = await load_agent_details(hitl)

    assert [agent.name for agent in details] == ["execute", "chat"]
    assert calls == ["execute", "chat"]
    assert details[0].config == (("mode", "safe"),)
    assert [tool.name for tool in details[0].tools] == ["search"]
    assert details[0].tools[0].arguments[0].name == "query"
    assert details[0].tools[0].mcp_server == "laboratory"
