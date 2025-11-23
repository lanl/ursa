from langchain_core.messages import AIMessage, SystemMessage

from ursa.agents.execution_agent import ExecutionAgent


class StubLLM:
    """Minimal LLM stub: records calls; .bind_tools returns self."""

    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages, config=None):
        self.calls.append(("invoke", messages, config))
        # Return a dummy “no-tool” reply if ever called (shouldn't be in the first assertion)
        return AIMessage(content="(stub)")


def _ai_with_unresolved_tools():
    # Simulate an assistant that proposed a tool call but hasn't gotten any ToolMessage yet.
    return AIMessage(
        content="fetch data",
        tool_calls=[
            {"id": "call_123", "name": "write_bytes", "args": {"filename": "x"}}
        ],
    )


def _agent(messages):
    llm = StubLLM()
    agent = ExecutionAgent(llm=llm, log_state=False, tool_log=False)
    state = {
        "messages": messages,
        "workspace": "tmp_waits_ws",
        "code_files": [],
        "symlinkdir": {},
        "current_progress": "",
    }
    return agent, llm, state


def test_query_executor_does_not_invoke_llm_while_missing_tool_outputs():
    # First message will be replaced by system by ExecutionAgent; include a placeholder
    msgs = [
        SystemMessage(content="placeholder system"),
        _ai_with_unresolved_tools(),
    ]
    agent, llm, state = _agent(msgs)

    out = agent.query_executor(state)

    # Should *not* have called the model, because the last AI has unresolved tool calls
    assert len(llm.calls) == 0, "LLM was called despite pending tool replies"
    # Should return state unchanged except for the normalized system prompt handling
    assert "messages" in out
    # And the tail still has the unresolved AI with tools
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].tool_calls


def test_query_executor_invokes_llm_after_tool_messages_present():
    # Same tool call, but now followed by a matching ToolMessage -> no longer missing
    ai = _ai_with_unresolved_tools()
    from langchain_core.messages import ToolMessage

    tool = ToolMessage(content="ok", tool_call_id=ai.tool_calls[0]["id"])
    msgs = [SystemMessage(content="placeholder system"), ai, tool]
    agent, llm, state = _agent(msgs)

    out = agent.query_executor(state)

    # Now it's allowed to call the model
    assert len(llm.calls) == 1, (
        "LLM wasn't called after tool replies were present"
    )
    # And we got a model response appended
    assert isinstance(out["messages"][-1], AIMessage)
    assert not getattr(out["messages"][-1], "tool_calls", None)
