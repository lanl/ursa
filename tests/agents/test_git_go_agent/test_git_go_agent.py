import shutil
from pathlib import Path

import pytest

from tests.agents.utils import ToolReadyFakeChatModel, message_stream

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


@pytest.fixture
def agent_factory(tmpdir):
    chat_model = ToolReadyFakeChatModel(messages=message_stream("ok"))
    workspace = Path(str(tmpdir))

    def create(factory, **kwargs):
        return factory(llm=chat_model, **kwargs, workspace=workspace)

    return create


@pytest.mark.parametrize("go", [True, False])
def test_git_agent_language_tools(agent_factory, go):
    """Both agents keep Git tools; only the Go agent adds its language tools."""
    agent = agent_factory(GitGoAgent if go else GitAgent)
    tool_names = set(agent.tools.keys())
    assert "git_status" in tool_names
    assert "git_diff" in tool_names
    assert "git_commit" in tool_names
    assert "git_add" in tool_names
    assert "git_switch" in tool_names
    assert "git_create_branch" in tool_names
    assert "git_log" in tool_names
    assert "git_ls_files" in tool_names
    if go:
        assert "go_build" in tool_names
        assert "go_test" in tool_names
        assert "go_vet" in tool_names
        assert "go_mod_tidy" in tool_names
        assert "golangci_lint" in tool_names
        assert "gofmt_files" in tool_names
    else:
        assert "go_build" not in tool_names
        assert "go_test" not in tool_names
        assert "gofmt_files" not in tool_names
        assert "golangci_lint" not in tool_names
    assert "run_command" not in tool_names
    assert "run_web_search" not in tool_names
    if go:
        assert "run_osti_search" not in tool_names
        assert "run_arxiv_search" not in tool_names
        assert "go" in agent.safe_codes


def test_make_git_agent_go_matches_git_go_agent(agent_factory):
    """make_git_agent(language='go') should produce the same tool set as GitGoAgent."""
    go_agent = agent_factory(GitGoAgent)
    factory_agent = agent_factory(make_git_agent, language="go")

    assert set(go_agent.tools.keys()) == set(factory_agent.tools.keys())
    assert go_agent.safe_codes == factory_agent.safe_codes


@pytest.mark.parametrize("language", ["generic", "rust"])
def test_make_git_agent_language_fallback(agent_factory, language):
    """Generic and unknown languages both produce a Git-only agent."""
    agent = agent_factory(make_git_agent, language=language)

    tool_names = set(agent.tools.keys())
    assert "git_status" in tool_names
    assert "go_build" not in tool_names


def test_make_git_agent_explicit_tools_bypass_registry(agent_factory):
    """make_git_agent accepts explicit tools/prompt/safe_codes, bypassing registry."""
    agent = agent_factory(
        make_git_agent,
        language="adoc",  # Not in registry
        safe_codes=["asciidoc", "podman"],  # Explicit safe codes
    )

    # Should be git-only (adoc not in registry), but with custom safe codes
    assert agent.safe_codes == {"asciidoc", "podman"}
    tool_names = set(agent.tools.keys())
    assert "git_status" in tool_names
