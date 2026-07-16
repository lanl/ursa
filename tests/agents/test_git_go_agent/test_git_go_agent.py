import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

if shutil.which("git") is None:
    pytest.skip(
        "Skipping git agent tests: `git` executable is not available on PATH. "
        "Install git and ensure it is available in the active shell.",
        allow_module_level=True,
    )

try:
    from ursa.agents import GitAgent, GitGoAgent, make_git_agent
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        "Skipping git agent tests: git-related Python tooling could not be imported. "
        "Install the project test dependencies and verify git tool integrations are available. "
        f"Import error: {exc}",
        allow_module_level=True,
    )


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
    factory_agent = make_git_agent(
        llm=chat_model, language="go", workspace=workspace
    )

    assert set(go_agent.tools.keys()) == set(factory_agent.tools.keys())
    assert go_agent.safe_codes == factory_agent.safe_codes


def test_make_git_agent_generic(tmpdir):
    """make_git_agent(language='generic') should produce a git-only agent."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))

    agent = make_git_agent(
        llm=chat_model, language="generic", workspace=workspace
    )

    tool_names = set(agent.tools.keys())
    assert "git_status" in tool_names
    assert "go_build" not in tool_names


def test_make_git_agent_unknown_language_defaults_to_git_only(tmpdir):
    """make_git_agent with an unknown language should fall back to git-only."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))

    agent = make_git_agent(llm=chat_model, language="rust", workspace=workspace)

    tool_names = set(agent.tools.keys())
    # Should have git tools
    assert "git_status" in tool_names
    # Should not have language-specific tools
    assert "go_build" not in tool_names


def test_make_git_agent_explicit_tools_bypass_registry(tmpdir):
    """make_git_agent accepts explicit tools/prompt/safe_codes, bypassing registry."""
    chat_model = ToolReadyFakeChatModel(messages=_message_stream("ok"))
    workspace = Path(str(tmpdir))

    # Create agent with explicit safe codes for a hypothetical "adoc" language
    agent = make_git_agent(
        llm=chat_model,
        language="adoc",  # Not in registry
        safe_codes=["asciidoc", "podman"],  # Explicit safe codes
        workspace=workspace,
    )

    # Should be git-only (adoc not in registry), but with custom safe codes
    assert agent.safe_codes == {"asciidoc", "podman"}
    tool_names = set(agent.tools.keys())
    assert "git_status" in tool_names
