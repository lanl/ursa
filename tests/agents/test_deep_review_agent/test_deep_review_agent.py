from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

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
