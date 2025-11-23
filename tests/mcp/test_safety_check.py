from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ursa.agents.execution_agent import ExecutionAgent


class SafetyStubLLM:
    """Returns safe/unsafe based on the asked text; bind_tools is a no-op."""

    def __init__(self, unsafe=True):
        self.unsafe = unsafe
        self.seen = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        # Capture the safety prompt text for sanity checks
        self.seen.append((config.get("tags") if config else None, messages))
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="[NO] definitely unsafe"
            if self.unsafe
            else "[YES] go ahead"
        )


def _agent_with(ai_msg, unsafe=True):
    llm = SafetyStubLLM(unsafe=unsafe)
    agent = ExecutionAgent(llm=llm, log_state=False, tool_log=False)
    state = {
        "messages": [SystemMessage(content="sys"), ai_msg],
        "workspace": "tmp_safety_ws",
        "code_files": ["ok.py"],
        "symlinkdir": {},
        "current_progress": "",
    }
    return agent, llm, state


def _ai_proposing_run_cmd(cmd: str):
    return AIMessage(
        content=f"running {cmd}",
        tool_calls=[
            {"id": "call_run", "name": "run_cmd", "args": {"query": cmd}}
        ],
    )


def test_safety_check_injects_unsafe_tool_message_and_blocks():
    ai = _ai_proposing_run_cmd("rm -rf /")
    agent, _, state = _agent_with(ai, unsafe=True)

    out = agent.safety_check(state)

    # We should see a ToolMessage injected with [UNSAFE]
    tools = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tools, "No ToolMessage injected by safety_check"
    assert any("[UNSAFE]" in (t.content or "") for t in tools)

    # And command_safe should return "unsafe"
    from ursa.agents.execution_agent import command_safe

    assert command_safe(out) == "unsafe"


def test_safety_check_allows_when_safe():
    ai = _ai_proposing_run_cmd("echo ok")
    agent, _, state = _agent_with(ai, unsafe=False)

    out = agent.safety_check(state)

    # No ToolMessage added when safe; the agent should proceed to action next
    from langchain_core.messages import ToolMessage

    assert not any(isinstance(m, ToolMessage) for m in out["messages"][1:])

    # And command_safe should return "safe"
    from ursa.agents.execution_agent import command_safe

    assert command_safe(out) == "safe"
