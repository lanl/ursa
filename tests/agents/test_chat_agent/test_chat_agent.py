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
    spawned = agent.tools["spawn_agent"].invoke({"name": "reviewer"})

    assert spawned == "Spawned agent root/reviewer"
    reviewer = agent.resolve_agent("root/reviewer")
    assert reviewer.agent_path == "root/reviewer"

    nested = reviewer.tools["spawn_agent"].invoke({"name": "critic"})
    assert nested == "Spawned agent root/reviewer/critic"
    assert reviewer.tools["list_agents"].invoke({}).splitlines() == [
        "root",
        "root/reviewer (self)",
        "root/reviewer/critic",
    ]

    listing = agent.tools["list_agents"].invoke({})
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
    agent.tools["spawn_agent"].invoke({"name": "first"})

    with pytest.raises(ToolException) as error:
        agent.tools["spawn_agent"].invoke({"name": "overflow"})

    assert "subagents.max_agents=1" in str(error.value)
    assert agent.handle_tool_errors(error.value) == str(error.value)
    assert [member.agent_path for member in agent.walk_agents()] == [
        "root",
        "root/first",
    ]

    listing = agent.tools["list_agents"].invoke({})
    assert "root/first" in listing
    await agent.stop_message_loop(descendants=True)
