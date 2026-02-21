from pathlib import Path
from typing import Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ursa.agents import GitGoAgent, GitAgent, make_git_agent


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


def test_git_agent_generic_has_only_git_tools(tmpdir):
    """GitAgent with no language tools should only have git tools."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))
    agent = GitAgent(llm=chat_model, workspace=workspace)

    tool_names = set(agent.tools.keys())
    # Git tools present
    assert "git_status" in tool_names
    assert "git_diff" in tool_names
    assert "git_commit" in tool_names
    assert "git_add" in tool_names
    assert "git_switch" in tool_names
    assert "git_create_branch" in tool_names
    assert "git_log" in tool_names
    assert "git_ls_files" in tool_names
    # No Go tools
    assert "go_build" not in tool_names
    assert "go_test" not in tool_names
    assert "gofmt_files" not in tool_names
    assert "golangci_lint" not in tool_names
    # Removed tools
    assert "run_command" not in tool_names
    assert "run_web_search" not in tool_names


def test_make_git_agent_go_matches_git_go_agent(tmpdir):
    """make_git_agent(language='go') should produce the same tool set as GitGoAgent."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))

    go_agent = GitGoAgent(llm=chat_model, workspace=workspace)
    factory_agent = make_git_agent(llm=chat_model, language="go", workspace=workspace)

    assert set(go_agent.tools.keys()) == set(factory_agent.tools.keys())
    assert go_agent.safe_codes == factory_agent.safe_codes


def test_make_git_agent_generic(tmpdir):
    """make_git_agent(language='generic') should produce a git-only agent."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))

    agent = make_git_agent(llm=chat_model, language="generic", workspace=workspace)

    tool_names = set(agent.tools.keys())
    assert "git_status" in tool_names
    assert "go_build" not in tool_names


def test_make_git_agent_unknown_language_raises(tmpdir):
    """make_git_agent with an unknown language should raise ValueError."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))

    try:
        make_git_agent(llm=chat_model, language="rust", workspace=workspace)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "rust" in str(exc)
