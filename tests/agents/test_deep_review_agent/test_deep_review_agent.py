from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ursa.agents.deep_review_agent import DeepReviewAgent


class ToolReadyFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class WorkspaceToolCallingFakeChatModel(GenericFakeChatModel):
    """Calls list_workspace_files once, then answers normally."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        saw_tool_result = any(isinstance(msg, ToolMessage) for msg in messages)
        if not saw_tool_result and not getattr(
            self, "_called_workspace", False
        ):
            self._called_workspace = True
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_workspace_files",
                        "args": {"pattern": "*.txt", "max_results": 10},
                        "id": "call-workspace-1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
        else:
            message = AIMessage(
                content="workspace-informed response",
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        )


@pytest.fixture
def chat_model():
    return ToolReadyFakeChatModel(messages=_message_stream("ok"))


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_deep_review_agent_can_autonomously_use_workspace_tools(
    tmpdir: Path,
) -> None:
    Path(tmpdir, "notes.txt").write_text("cooling note", encoding="utf-8")
    model = WorkspaceToolCallingFakeChatModel(messages=_message_stream("ok"))
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


def _recording_model(log: list) -> ToolReadyFakeChatModel:
    """Fake model that records every request's message list."""

    class _Recorder(ToolReadyFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            log.append(list(messages))
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    return _Recorder(messages=_message_stream("ok"))


def _exploding_model(calls: list) -> ToolReadyFakeChatModel:
    """Fake model that raises on every call and counts the calls."""

    class _Exploder(ToolReadyFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls.append(len(messages))
            raise RuntimeError("provider exploded")

    return _Exploder(messages=_message_stream("unused"))


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
        yield AIMessage(
            content=f"turn-{n:02d}",
            usage_metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        )


_DD_QUESTION = {
    "question": "How can we reduce cooling energy usage?",
    "current_iteration": 0,
    "max_iterations": 1,
}


@pytest.mark.asyncio
async def test_phase_requests_keep_system_prompts_leading(tmpdir):
    # Issue 294: from the second debate phase onward, requests carried
    # mid-conversation system messages, which langchain-anthropic rejects.
    log: list = []
    agent = DeepReviewAgent(llm=_recording_model(log), workspace=tmpdir)

    await agent.ainvoke(dict(_DD_QUESTION))

    assert log, "no model requests captured"
    bad = [
        [type(message).__name__ for message in request]
        for request in log
        if not _system_prefix_ok(request)
    ]
    assert not bad, f"requests with mid-conversation system messages: {bad}"


@pytest.mark.asyncio
async def test_phase_prompts_not_persisted_as_system_messages(tmpdir):
    log: list = []
    agent = DeepReviewAgent(llm=_recording_model(log), workspace=tmpdir)

    result = await agent.ainvoke(dict(_DD_QUESTION))

    persisted_systems = [
        message
        for message in result.get("messages", [])
        if isinstance(message, SystemMessage)
    ]
    assert persisted_systems == [], (
        "phase role prompts must not accumulate in the message channel"
    )


@pytest.mark.asyncio
async def test_each_phase_request_leads_with_its_own_role_prompt(tmpdir):
    log: list = []
    agent = DeepReviewAgent(llm=_recording_model(log), workspace=tmpdir)

    await agent.ainvoke(dict(_DD_QUESTION))

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


@pytest.mark.asyncio
async def test_phase_model_error_propagates_immediately(tmpdir):
    # Issue 294: phase errors were swallowed into AIMessage critiques
    # ("Deep-review phase error: ...") and fed to later phases, so a
    # failing model still produced a normal-looking run. The first
    # failure must propagate before any further model call happens.
    calls: list = []
    agent = DeepReviewAgent(llm=_exploding_model(calls), workspace=tmpdir)

    with pytest.raises(RuntimeError, match="provider exploded"):
        await agent.ainvoke(dict(_DD_QUESTION))

    assert len(calls) == 1, (
        "the first phase failure must propagate immediately, not be "
        "swallowed into critique text while later phases keep calling "
        f"the model (saw {len(calls)} calls)"
    )

async def test_output_lists_have_exactly_one_entry_per_iteration(tmpdir):
    # One real iteration must produce exactly one entry per output list;
    # the previous >= 1 assertions were green with every entry doubled.
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=_message_stream("ok")),
        workspace=tmpdir,
    )

    result = await agent.ainvoke(dict(_DD_QUESTION))

    assert result["agent1_solution"] == ["ok"]
    assert result["agent2_critiques"] == ["ok"]
    assert result["agent3_perspectives"] == ["ok"]


@pytest.mark.asyncio
async def test_output_lists_scale_with_iterations(tmpdir):
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=_message_stream("ok")),
        workspace=tmpdir,
    )

    result = await agent.ainvoke({**_DD_QUESTION, "max_iterations": 2})

    for key in ("agent1_solution", "agent2_critiques", "agent3_perspectives"):
        assert len(result[key]) == 2, f"{key}: {result[key]}"


@pytest.mark.asyncio
async def test_chained_runs_report_real_iterations_only(tmpdir):
    # Chaining a second run on the first run's returned state (the HITL
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


@pytest.mark.asyncio
async def test_seeded_visited_sites_survive(tmpdir):
    # Guard (green before and after the fix): visited sites provided in
    # the input state survive to the returned state.
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=_message_stream("ok")),
        workspace=tmpdir,
    )

    result = await agent.ainvoke({
        **_DD_QUESTION,
        "visited_sites": {"https://seeded.example/x"},
    })

    assert "https://seeded.example/x" in result["visited_sites"]


@pytest.mark.asyncio
async def test_details_file_has_one_section_per_iteration(tmpdir):
    # Guard (green before and after the fix): a single run's report has
    # exactly one section per real iteration, protecting the
    # finalize-summarize-print ordering against future rewiring.
    agent = DeepReviewAgent(
        llm=ToolReadyFakeChatModel(messages=_message_stream("ok")),
        workspace=tmpdir,
    )

    await agent.ainvoke({**_DD_QUESTION, "max_iterations": 2})

    text = sorted(agent.workspace.glob("iteration_details_*.txt"))[
        -1
    ].read_text(encoding="utf-8")
    assert text.count("\\subsection") == 2
