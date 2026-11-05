from types import SimpleNamespace

import pytest
from langchain.tools import ToolException
from langchain_core.messages import AIMessage, HumanMessage

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


async def test_chat_agent_exposes_hierarchy_tools(chat_model, tmp_path):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)

    assert {"spawn_agent", "send_message", "list_agents"} <= agent.tools.keys()
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


async def test_unknown_agent_path_is_a_recoverable_hierarchy_tool_error(
    chat_model, tmp_path
):
    agent = ChatAgent(llm=chat_model, workspace=tmp_path)

    spawned = await agent.tools["spawn_agent"].ainvoke({
        "name": "reviewer",
        "message": "Review this normally.",
    })
    assert spawned.startswith("Spawned agent root/reviewer")
    reviewer = agent.resolve_agent("root/reviewer")

    with pytest.raises(ToolException, match="Unknown agent path"):
        await reviewer.tools["send_message"].ainvoke({
            "target": "../foo",
            "message": "Please review this.",
        })

    async def invalid_target(_target, _message):
        raise RuntimeError("Unknown agent path '../foo' from 'root'")

    agent._process_runtime = SimpleNamespace(send_message=invalid_target)
    try:
        with pytest.raises(ToolException, match="Unknown agent path"):
            await agent.tools["send_message"].ainvoke({
                "target": "../foo",
                "message": "Please review this.",
            })
    finally:
        del agent._process_runtime

    assert "root/reviewer" in await agent.tools["list_agents"].ainvoke({})
    await agent.stop_message_loop(descendants=True)
