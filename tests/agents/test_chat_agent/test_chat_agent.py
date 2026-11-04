from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ursa.agents.chat_agent import ChatAgent
from ursa.tools.term_tool import get_supported_term_tools


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


class _ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def test_chat_agent_dispatches_opt_in_terminal_tools(
    monkeypatch, tmp_path
):
    model = _ToolCallingModel(
        messages=iter([
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "term",
                        "args": {"cmd": "python", "session": True},
                        "id": "launch",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Terminal operations complete."),
        ])
    )
    safety_calls = []

    def assess_safety(command, runtime):
        safety_calls.append(command)
        return SimpleNamespace(is_safe=True, reason="test")

    class Manager:
        def supports_screen(self):
            return False

        async def create(self, cmd, **kwargs):
            assert cmd == "python"
            return SimpleNamespace(term_id="PyTerm01")

    monkeypatch.setattr(
        "ursa.tools.term_tool.assess_command_safety", assess_safety
    )
    monkeypatch.setattr("ursa.tools.term_tool.term_manager", Manager())
    agent = ChatAgent(llm=model, workspace=tmp_path)
    assert "run_command" in agent.tools
    agent.add_tool(get_supported_term_tools())
    assert "term" in agent.tools
    assert "term_screenshot" not in agent.tools

    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Use the terminal")],
        "thread_id": agent.thread_id,
    })

    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert safety_calls
    assert [message.content for message in tool_messages] == [
        "Terminal ID: PyTerm01"
    ]
    assert result["messages"][-1].content == "Terminal operations complete."
