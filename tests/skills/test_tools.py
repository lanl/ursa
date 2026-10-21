from pathlib import Path

import pytest
from pydantic import ValidationError

from ursa.skills.activation import annotate_skill_requests
from ursa.skills.discovery import split_frontmatter
from ursa.skills.tools import (
    SKILL_TOOL_NAME,
    build_skill_tool,
    load_skill_instructions,
)


@pytest.fixture
def home(monkeypatch, tmp_path):
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: root)
    return root


def write_skill(root, name, description, body="Do the thing."):
    path = root / ".ursa" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_no_tool_when_no_skills_exist(home, tmp_path):
    assert build_skill_tool(tmp_path, skills={}) is None


def test_tool_lists_every_skill_in_its_description(home, tmp_path):
    write_skill(home, "user-skill", "From the user root")
    write_skill(tmp_path, "write-python", "House style for Python")

    tool = build_skill_tool(tmp_path)

    assert tool.name == SKILL_TOOL_NAME
    assert list(tool.tool_call_schema.model_json_schema()["properties"]) == [
        "name"
    ]
    assert "- write-python: House style for Python" in tool.description
    assert "- user-skill: From the user root" in tool.description


def test_tool_returns_the_skill_instructions(home, tmp_path):
    write_skill(tmp_path, "write-python", "Python", body="Run with `uv run`.")

    result = build_skill_tool(tmp_path).invoke({"name": "write-python"})

    assert split_frontmatter(result)[0]["name"] == "write-python"
    assert "Run with `uv run`." in result
    assert str(tmp_path) in split_frontmatter(result)[0]["source"]


def test_tool_schema_stays_consistent_with_its_catalog(home, tmp_path):
    write_skill(tmp_path, "first", "First")
    tool = build_skill_tool(tmp_path)

    write_skill(tmp_path, "second", "Second")

    assert "second" not in tool.args["name"]["enum"]
    with pytest.raises(ValidationError):
        tool.invoke({"name": "second"})


def test_unknown_skill_returns_a_recoverable_message(home, tmp_path):
    write_skill(tmp_path, "known", "Known")

    result = load_skill_instructions("nope", tmp_path)

    assert "No skill named 'nope'" in result
    assert "known" in result


@pytest.mark.parametrize(
    "name", ["../../../etc/passwd", "/etc/passwd", "..", "a/b"]
)
def test_path_like_names_cannot_escape_the_skills_roots(home, tmp_path, name):
    write_skill(tmp_path, "known", "Known")

    result = load_skill_instructions(name, tmp_path)

    assert result.startswith("No skill named")


def test_leading_dollar_from_the_model_is_tolerated(home, tmp_path):
    write_skill(tmp_path, "write-python", "Python")

    result = load_skill_instructions("$write-python", tmp_path)

    assert split_frontmatter(result)[0]["name"] == "write-python"


def test_project_skill_shadows_user_skill_through_the_tool(home, tmp_path):
    write_skill(home, "shared", "User", body="User body")
    write_skill(tmp_path, "shared", "Project", body="Project body")

    result = load_skill_instructions("shared", tmp_path)

    assert "Project body" in result
    assert "User body" not in result


def test_dollar_reference_becomes_a_tool_directive(home, tmp_path):
    write_skill(tmp_path, "write-python", "Python")

    annotated = annotate_skill_requests(
        "Tidy this up $write-python", project_root=tmp_path
    )

    assert annotated.startswith("Tidy this up $write-python")
    assert "write-python" in annotated.splitlines()[-1]
    assert f"`{SKILL_TOOL_NAME}`" in annotated


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_graph_injects_runtime_character_limit(
    home, tmp_path, chat_model, asynchronous
):
    from langchain_core.messages import AIMessage
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from ursa.agents.base import AgentContext

    write_skill(
        tmp_path, "large", "Large instructions", body="0123456789" * 1000
    )
    tool = build_skill_tool(tmp_path)
    graph = StateGraph(MessagesState, context_schema=AgentContext)
    graph.add_node("tools", ToolNode([tool]))
    graph.set_entry_point("tools")
    graph.set_finish_point("tools")
    compiled = graph.compile()
    context = AgentContext(
        llm=chat_model,
        workspace=tmp_path,
        agent_name="test",
        group="default",
        den=tmp_path,
        tool_character_limit=512,
    )
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "skill",
                        "args": {"name": "large"},
                        "id": "skill-call",
                    }
                ],
            )
        ]
    }
    if asynchronous:
        result = await compiled.ainvoke(state, context=context)
    else:
        result = compiled.invoke(state, context=context)
    message = result["messages"][-1]
    assert message.status == "success"
    assert len(message.content) == 512
    assert "Truncated" in message.content
    assert "0123456789" in message.content
    assert "source:" in message.content


async def test_tool_activation_reuses_the_catalog(home, tmp_path, monkeypatch):
    from ursa.skills import discovery

    write_skill(tmp_path, "example", "Example")
    tool = build_skill_tool(tmp_path)

    def no_scan(*args, **kwargs):
        pytest.fail("Activation must not scan the skills roots")

    monkeypatch.setattr(discovery, "discover_skills_in", no_scan)
    assert "Do the thing." in tool.invoke({"name": "example"})
    assert "Do the thing." in await tool.ainvoke({"name": "example"})


def test_missing_instruction_file_returns_a_recoverable_error(home, tmp_path):
    path = write_skill(tmp_path, "removed", "Removed")
    tool = build_skill_tool(tmp_path)
    path.unlink()
    assert "Could not load skill 'removed'" in tool.invoke({"name": "removed"})


def test_tool_rejects_unknown_names(home, tmp_path):
    write_skill(tmp_path, "known", "Known")
    with pytest.raises(ValidationError):
        build_skill_tool(tmp_path).invoke({"name": "unknown"})
