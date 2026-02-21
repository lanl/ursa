from pathlib import Path
from typing import Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ursa.agents import GitGoAgent


class ToolReadyFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=content)


def test_git_go_agent_tools(tmpdir):
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))
    agent = GitGoAgent(llm=chat_model, workspace=workspace)

    tool_names = set(agent.tools.keys())
    # Git tools
    assert "git_status" in tool_names
    assert "git_diff" in tool_names
    assert "git_commit" in tool_names
    assert "git_add" in tool_names
    assert "git_switch" in tool_names
    assert "git_create_branch" in tool_names
    assert "git_log" in tool_names
    assert "git_ls_files" in tool_names
    # Go tools
    assert "go_build" in tool_names
    assert "go_test" in tool_names
    assert "go_vet" in tool_names
    assert "go_mod_tidy" in tool_names
    assert "golangci_lint" in tool_names
    # Code formatting
    assert "gofmt_files" in tool_names
    # Removed tools
    assert "run_command" not in tool_names
    assert "run_web_search" not in tool_names
    assert "run_osti_search" not in tool_names
    assert "run_arxiv_search" not in tool_names
    # Configuration
    assert "go" in agent.safe_codes
