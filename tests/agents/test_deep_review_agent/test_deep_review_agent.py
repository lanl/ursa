from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from tests.agents.utils import (
    ToolReadyFakeChatModel,
    message_stream_with_usage,
    message_with_usage,
    recording_model,
)
from ursa.agents.deep_review_agent import DeepReviewAgent


class WorkspaceToolCallingFakeChatModel(ToolReadyFakeChatModel):
    """Calls list_workspace_files once, then answers normally."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        saw_tool_result = any(isinstance(msg, ToolMessage) for msg in messages)
        if not saw_tool_result and not getattr(
            self, "_called_workspace", False
        ):
            self._called_workspace = True
            message = message_with_usage(
                content="",
                tool_calls=[
                    {
                        "name": "list_workspace_files",
                        "args": {"pattern": "*.txt", "max_results": 10},
                        "id": "call-workspace-1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = message_with_usage(content="workspace-informed response")
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def chat_model():
    return ToolReadyFakeChatModel(messages=message_stream_with_usage("ok"))


async def test_deep_review_agent_ainvoke_without_hidden_web_search(
    chat_model,
    tmpdir: Path,
) -> None:
    agent = DeepReviewAgent(llm=chat_model, workspace=tmpdir)

    assert "list_workspace_files" in agent.tools
    assert "read_file" in agent.tools
    assert "run_web_search" not in agent.tools
    assert "run_arxiv_search" not in agent.tools
    assert "run_osti_search" not in agent.tools

    result = await agent.ainvoke({
        "question": "How can we reduce the cooling energy usage in edge data centers?",
        "current_iteration": 0,
        "max_iterations": 1,
    })

    assert isinstance(result["agent1_solution"], Sequence)
    assert isinstance(result["agent2_critiques"], Sequence)
    assert isinstance(result["agent3_perspectives"], Sequence)
    assert len(result["agent1_solution"]) >= 1
    assert len(result["agent2_critiques"]) >= 1
    assert len(result["agent3_perspectives"]) >= 1
    assert isinstance(result["solution"], str)
    assert isinstance(result["summary_report"], str)
    if result["summary_report"].strip():
        assert "\\documentclass" in result["summary_report"]
    assert result["current_iteration"] == 1
    assert result["visited_sites"] == set()
    assert isinstance(result["question_search_query"], str)

    generated_logs = list(agent.workspace.glob("iteration_details_*.txt"))
    assert generated_logs, "Expected iteration history files to be written"


async def test_deep_review_agent_can_autonomously_use_workspace_tools(
    tmpdir: Path,
) -> None:
    Path(tmpdir, "notes.txt").write_text("cooling note", encoding="utf-8")
    model = WorkspaceToolCallingFakeChatModel(
        messages=message_stream_with_usage("ok")
    )
    agent = DeepReviewAgent(llm=model, workspace=tmpdir)

    result = await agent.ainvoke({
        "question": "What does the workspace say about cooling?",
        "current_iteration": 0,
        "max_iterations": 1,
    })

    tool_messages = [
        message
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    assert tool_messages
    assert "notes.txt" in tool_messages[0].content
    assert result["agent1_solution"][0] == "workspace-informed response"


def test_deep_review_agent_exposes_web_tools_only_when_enabled(
    chat_model, tmpdir
):
    no_web = DeepReviewAgent(llm=chat_model, workspace=tmpdir, use_web=False)
    with_web = DeepReviewAgent(llm=chat_model, workspace=tmpdir, use_web=True)

    assert "run_web_search" not in no_web.tools
    assert "run_arxiv_search" not in no_web.tools
    assert "run_osti_search" not in no_web.tools
    assert "run_web_search" in with_web.tools
    assert "run_arxiv_search" in with_web.tools
    assert "run_osti_search" in with_web.tools


def _system_prefix_ok(messages) -> bool:
    seen_non_system = False
    for message in messages:
        if isinstance(message, SystemMessage):
            if seen_non_system:
                return False
        else:
            seen_non_system = True
    return True


def _distinct_stream() -> Iterator[AIMessage]:
    n = 0
    while True:
        n += 1
        yield message_with_usage(content=f"turn-{n:02d}")


_DD_QUESTION = {
    "question": "How can we reduce cooling energy usage?",
    "current_iteration": 0,
    "max_iterations": 1,
}


async def test_phase_prompts_lead_requests_without_persisting_in_history(
    tmpdir,
):
    # Issue 294: from the second debate phase onward, requests carried
    # mid-conversation system messages, which langchain-anthropic rejects.
    log: list = []
    agent = DeepReviewAgent(llm=recording_model(log, "ok"), workspace=tmpdir)

    result = await agent.ainvoke(dict(_DD_QUESTION))

    assert log, "no model requests captured"
    bad = [
        [type(message).__name__ for message in request]
        for request in log
        if not _system_prefix_ok(request)
    ]
    assert not bad, f"requests with mid-conversation system messages: {bad}"

    persisted_systems = [
        message
        for message in result.get("messages", [])
        if isinstance(message, SystemMessage)
    ]
    assert persisted_systems == [], (
        "phase role prompts must not accumulate in the message channel"
    )

    leading_prompts = [
        request[0].content
        for request in log
        if request and isinstance(request[0], SystemMessage)
    ]
    assert len(leading_prompts) >= 3, (
        "each debate phase should lead its request with a system prompt"
    )
    assert len(set(leading_prompts)) >= 3, (
        "phases received identical leading role prompts"
    )


async def test_phase_model_error_propagates_immediately(tmpdir):
    # Issue 294: phase errors were swallowed into AIMessage critiques
    # ("Deep-review phase error: ...") and fed to later phases, so a
    # failing model still produced a normal-looking run. The first
    # failure must propagate before any further model call happens.
    calls: list = []
    model = ToolReadyFakeChatModel(messages=message_stream_with_usage("unused"))

    def fail(messages, stop=None, run_manager=None, **kwargs):
        calls.append(len(messages))
        raise RuntimeError("provider exploded")

    model._generate = fail
    agent = DeepReviewAgent(llm=model, workspace=tmpdir)

    with pytest.raises(RuntimeError, match="provider exploded"):
        await agent.ainvoke(dict(_DD_QUESTION))

    assert len(calls) == 1, (
        "the first phase failure must propagate immediately, not be "
        "swallowed into critique text while later phases keep calling "
        f"the model (saw {len(calls)} calls)"
    )


@pytest.mark.parametrize("iterations", [1, 2])
async def test_outputs_and_report_match_real_iterations(tmpdir, iterations):
    # Each real iteration must produce exactly one entry per output list;
    # the previous >= 1 assertions were green with every entry doubled.
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=message_stream_with_usage("ok")),
        workspace=tmpdir,
    )

    result = await agent.ainvoke({**_DD_QUESTION, "max_iterations": iterations})

    for key in ("agent1_solution", "agent2_critiques", "agent3_perspectives"):
        assert result[key] == ["ok"] * iterations, f"{key}: {result[key]}"

    # Report sections also follow real iterations, guarding finalization order.
    text = sorted(agent.workspace.glob("iteration_details_*.txt"))[
        -1
    ].read_text(encoding="utf-8")
    assert text.count("\\subsection") == iterations


async def test_chained_runs_report_real_iterations_only(tmpdir):
    # Chaining a second run on the first run's returned state (the AgentRuntime
    # pattern) must report exactly the real iterations, with each report
    # section attributed to its own iteration's content.
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=_distinct_stream()),
        workspace=tmpdir,
    )

    run1 = await agent.ainvoke(dict(_DD_QUESTION))
    run2 = await agent.ainvoke(
        agent.format_query(
            "How can we reduce cooling energy usage?", state=run1
        )
    )

    for key in ("agent1_solution", "agent2_critiques", "agent3_perspectives"):
        assert len(run2[key]) == 2, f"{key}: {run2[key]}"
    details = sorted(agent.workspace.glob("iteration_details_*.txt"))
    text = details[-1].read_text(encoding="utf-8")
    sections = text.split("\\subsection")
    assert len(sections) - 1 == 2, (
        f"expected 2 report sections, found {len(sections) - 1}"
    )
    assert "turn-01" in sections[1]
    assert "turn-06" in sections[2]


async def test_seeded_visited_sites_survive(tmpdir):
    # Guard (green before and after the fix): visited sites provided in
    # the input state survive to the returned state.
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=message_stream_with_usage("ok")),
        workspace=tmpdir,
    )

    result = await agent.ainvoke({
        **_DD_QUESTION,
        "visited_sites": {"https://seeded.example/x"},
    })

    assert "https://seeded.example/x" in result["visited_sites"]
