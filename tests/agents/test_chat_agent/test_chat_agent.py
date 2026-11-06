import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.tools import ToolException, tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tests.agents.utils import ScriptedRecordingChatModel
from ursa.agents.chat_agent import ChatAgent


async def test_chat_agent_appends_ai_response(chat_model, tmpdir):
    agent = ChatAgent(llm=chat_model, workspace=tmpdir)
    user_prompt = "Share a quick greeting."
    initial_message = HumanMessage(content=user_prompt)

    result = await agent.ainvoke({
        "messages": [initial_message],
        "thread_id": agent.thread_id,
    })

    assert "messages" in result
    messages = result["messages"]
    assert len(messages) >= 2
    assert messages[0].type == "human"
    assert messages[0].content == user_prompt

    ai_message = messages[-1]
    assert isinstance(ai_message, AIMessage)
    assert ai_message.type == "ai"
    assert ai_message.usage_metadata["total_tokens"] > 0
    assert result["thread_id"] == agent.thread_id


def test_chat_agent_explicitly_enables_openai_parallel_tool_calls(tmp_path):
    """OpenAI tool bindings explicitly allow batching."""
    model = ChatOpenAI(model="gpt-5.4", api_key="test-key")

    agent = ChatAgent(llm=model, workspace=tmp_path)

    assert agent.llm.kwargs["parallel_tool_calls"] is True


async def test_chat_agent_runs_independent_tool_calls_concurrently(tmp_path):
    """One tool-call turn uses LangGraph's concurrent ToolNode execution."""
    model = ScriptedRecordingChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "first_independent_action",
                        "args": {},
                        "id": "first",
                        "type": "tool_call",
                    },
                    {
                        "name": "second_independent_action",
                        "args": {},
                        "id": "second",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Both independent actions completed."),
        ]
    )
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    @tool
    async def first_independent_action() -> str:
        """Complete the first independent action."""
        first_started.set()
        await asyncio.wait_for(second_started.wait(), timeout=5)
        return "first complete"

    @tool
    async def second_independent_action() -> str:
        """Complete the second independent action."""
        second_started.set()
        await asyncio.wait_for(first_started.wait(), timeout=5)
        return "second complete"

    agent = ChatAgent(llm=model, workspace=tmp_path)
    agent.add_tool([first_independent_action, second_independent_action])

    result = await agent.ainvoke("Run the independent actions together.")

    assert (
        result["messages"][-1].content == "Both independent actions completed."
    )
    assert [m.content for m in result["messages"] if m.type == "tool"] == [
        "first complete",
        "second complete",
    ]


async def test_spawn_startup_timeout_is_a_recoverable_tool_error(
    chat_model, tmp_path
):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)

    async def failed_start(_name, _message, _agent_type):
        raise RuntimeError(
            "Agent 'session:sub:root/slow' did not become ready within 120 seconds"
        )

    agent._process_runtime = SimpleNamespace(spawn_subagent=failed_start)
    with pytest.raises(ToolException, match="did not become ready"):
        await agent.tools["spawn_agent"].ainvoke({
            "name": "slow",
            "message": "Work on an independent task.",
        })


async def test_chat_agent_exposes_hierarchy_tools(chat_model, tmp_path):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)

    assert {
        "spawn_agent",
        "send_message",
        "wait_for",
        "stop_agent",
        "list_agents",
    } <= agent.tools.keys()
    spawned = await agent.tools["spawn_agent"].ainvoke({
        "name": "reviewer",
        "message": "Review the greeting.",
    })

    assert spawned.startswith("Spawned agent root/reviewer\n\n")
    reviewer = agent.resolve_agent("root/reviewer")
    assert reviewer.agent_path == "root/reviewer"

    nested = await reviewer.tools["spawn_agent"].ainvoke({
        "name": "critic",
        "message": "Critique the greeting.",
    })
    assert nested.startswith("Spawned agent root/reviewer/critic\n\n")
    assert (await reviewer.tools["list_agents"].ainvoke({})).splitlines() == [
        "root",
        "root/reviewer (self)",
        "root/reviewer/critic",
    ]

    listing = await agent.tools["list_agents"].ainvoke({})
    assert listing.splitlines() == [
        "root (self)",
        "root/reviewer",
        "root/reviewer/critic",
    ]

    response = await agent.tools["send_message"].ainvoke({
        "target": "root/reviewer",
        "message": "Share a quick greeting.",
    })
    assert isinstance(response, str)
    assert response

    await agent.stop_message_loop(descendants=True)


async def test_spawn_agent_limit_becomes_recoverable_tool_error(
    chat_model, tmp_path
):
    agent = ChatAgent(
        llm=chat_model,
        workspace=tmp_path,
        subagent_max_agents=1,
    )
    await agent.tools["spawn_agent"].ainvoke({
        "name": "first",
        "message": "Handle the first task.",
    })

    with pytest.raises(ToolException) as error:
        await agent.tools["spawn_agent"].ainvoke({
            "name": "overflow",
            "message": "This should not launch.",
        })

    assert "subagents.max_agents=1" in str(error.value)
    assert agent.handle_tool_errors(error.value) == str(error.value)
    assert [member.agent_path for member in agent.walk_agents()] == [
        "root",
        "root/first",
    ]

    listing = await agent.tools["list_agents"].ainvoke({})
    assert "root/first" in listing
    await agent.stop_message_loop(descendants=True)


async def test_local_spawn_agent_rejects_host_agent_type_selection(
    chat_model, tmp_path
):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)

    with pytest.raises(ToolException, match="managed AgentRuntime"):
        await agent.tools["spawn_agent"].ainvoke({
            "name": "planner",
            "message": "Create a plan.",
            "agent_type": "plan",
        })

    assert [member.agent_path for member in agent.walk_agents()] == ["root"]
    await agent.stop_message_loop(descendants=True)


@pytest.mark.parametrize(
    "bridge_method", [None, "send_message", "wait_for_agent"]
)
async def test_unknown_agent_path_is_a_recoverable_hierarchy_tool_error(
    chat_model, tmp_path, monkeypatch, bridge_method
):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)
    spawned = await agent.tools["spawn_agent"].ainvoke({
        "name": "reviewer",
        "message": "Review this normally.",
    })
    assert spawned.startswith("Spawned agent root/reviewer")
    caller = agent.resolve_agent("root/reviewer")
    tool_name = "send_message"
    options = {"target": "../foo", "message": "Please review this."}
    if bridge_method is not None:
        caller = agent
        error = RuntimeError("Unknown agent path '../foo' from 'root'")
        bridge = SimpleNamespace(**{
            bridge_method: AsyncMock(side_effect=error)
        })
        monkeypatch.setattr(agent, "_process_runtime", bridge, raising=False)
        if bridge_method == "wait_for_agent":
            tool_name = "wait_for"
            options = {"target": "../foo", "timeout_seconds": 0}
    with pytest.raises(ToolException, match="Unknown agent path"):
        await caller.tools[tool_name].ainvoke(options)
    if bridge_method is not None:
        monkeypatch.delattr(agent, "_process_runtime")
    assert "root/reviewer" in await agent.tools["list_agents"].ainvoke({})
    await agent.stop_message_loop(descendants=True)


async def test_managed_hierarchy_tool_queues_broadcast_in_input_order(
    chat_model, tmp_path
):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)

    expected = [
        f"Message queued for {target}."
        for target in ("first", "missing", "second")
    ]
    send_to_runtime = AsyncMock(
        return_value=[{"result": ack} for ack in expected]
    )
    agent._process_runtime = SimpleNamespace(send_messages=send_to_runtime)
    response = await agent.tools["send_message"].ainvoke({
        "target": ["first", "missing", "second"],
        "message": "Review this.",
    })
    send_to_runtime.assert_awaited_once_with(
        ["first", "missing", "second"], "Review this."
    )
    assert response.split("\n\n") == expected
    await agent.stop_message_loop(descendants=True)
