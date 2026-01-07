import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ursa.agents.chat_agent import ChatAgent


@pytest.mark.anyio
async def test_chat_agent_appends_ai_response(chat_model):
    agent = ChatAgent(llm=chat_model)
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
