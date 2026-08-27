from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
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


class _TermCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def test_chat_agent_dispatches_async_term_safety_check(
    monkeypatch, tmp_path
):
    """A start-session request reaches ``term`` through the async ToolNode."""
    messages = iter([
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "term",
                    "args": {"cmd": "python", "session": True},
                    "id": "start-python",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Python terminal started."),
    ])
    model = _TermCallingModel(messages=messages)
    safety_awaited = False

    async def assess_safety(command, runtime):
        nonlocal safety_awaited
        safety_awaited = True
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

    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Start a python terminal session")],
        "thread_id": agent.thread_id,
    })

    assert safety_awaited is True
    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "Terminal ID: PyTerm01"
    assert result["messages"][-1].content == "Python terminal started."


async def test_chat_agent_recovers_from_unknown_terminal_id(
    monkeypatch, tmp_path
):
    """A stale terminal ID becomes a ToolMessage and the graph continues."""
    messages = iter([
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "term_read",
                    "args": {"term_id": "Missing1"},
                    "id": "read-missing-terminal",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="That terminal no longer exists."),
    ])
    model = _TermCallingModel(messages=messages)

    class Manager:
        def supports_screen(self):
            return False

        async def read(self, term_id, **kwargs):
            raise KeyError(term_id)

    monkeypatch.setattr("ursa.tools.term_tool.term_manager", Manager())
    agent = ChatAgent(llm=model, workspace=tmp_path)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Read terminal Missing1")],
        "thread_id": agent.thread_id,
    })

    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].content == (
        "Unknown terminal ID 'Missing1'. Use an ID returned by the term tool."
    )
    assert result["messages"][-1].content == "That terminal no longer exists."
