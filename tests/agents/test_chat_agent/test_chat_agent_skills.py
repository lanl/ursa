import pytest

from ursa.agents.chat_agent import ChatAgent
from ursa.skills import SKILL_TOOL_NAME


def write_skill(root, name, description, body="Do the thing."):
    path = root / ".ursa" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8"
    )


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_skill_tool_is_bound_when_skills_exist(chat_model, project):
    write_skill(project, "write-python", "House style for Python")

    agent = ChatAgent(llm=chat_model, workspace=project)

    assert SKILL_TOOL_NAME in agent.tools
    assert (
        "write-python: House style for Python"
        in agent.tools[SKILL_TOOL_NAME].description
    )


def test_no_skill_tool_without_skills(chat_model, project):
    agent = ChatAgent(llm=chat_model, workspace=project)

    assert SKILL_TOOL_NAME not in agent.tools


def test_use_skills_false_disables_the_tool(chat_model, project):
    write_skill(project, "write-python", "House style for Python")

    agent = ChatAgent(llm=chat_model, workspace=project, use_skills=False)

    assert SKILL_TOOL_NAME not in agent.tools


def test_dollar_reference_becomes_a_tool_directive(chat_model, project):
    write_skill(project, "write-python", "House style for Python")
    agent = ChatAgent(llm=chat_model, workspace=project)

    state = agent.format_query("tidy this up $write-python")

    message = state["messages"][-1]
    assert message.type == "human"
    assert message.content.startswith("tidy this up $write-python")
    assert f"`{SKILL_TOOL_NAME}`" in message.content


def test_unknown_dollar_reference_is_left_alone(chat_model, project):
    write_skill(project, "write-python", "House style for Python")
    agent = ChatAgent(llm=chat_model, workspace=project)

    state = agent.format_query("pay me $500")

    assert state["messages"][-1].content == "pay me $500"


def test_prompt_is_untouched_when_skills_are_disabled(chat_model, project):
    write_skill(project, "write-python", "House style for Python")
    agent = ChatAgent(llm=chat_model, workspace=project, use_skills=False)

    state = agent.format_query("tidy this up $write-python")

    assert state["messages"][-1].content == "tidy this up $write-python"


async def test_skill_tool_returns_instructions_through_the_agent(
    chat_model, project
):
    write_skill(project, "write-python", "Python", body="Run with `uv run`.")
    agent = ChatAgent(llm=chat_model, workspace=project)

    result = agent.tools[SKILL_TOOL_NAME].invoke({"name": "write-python"})

    assert "Run with `uv run`." in result
