from pathlib import Path

from ursa.agents.chat_agent import ChatAgent
from ursa.agents.execution_agent import ExecutionAgent


def _write_skill(cwd: Path, name: str, description: str, body: str) -> None:
    skill_dir = cwd / ".ursa" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_execution_agent_loads_skill_catalog_and_tool(
    tmp_path, chat_model, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path, "widgets", "Builds widgets", "How to build widgets.")

    agent = ExecutionAgent(llm=chat_model, workspace=tmp_path / "ws")

    assert "load_skill" in agent.tools
    assert "widgets: Builds widgets" in agent.executor_prompt


def test_execution_agent_no_skills_is_noop(tmp_path, chat_model, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    agent = ExecutionAgent(llm=chat_model, workspace=tmp_path / "ws")

    assert "load_skill" not in agent.tools
    assert "Available Skills" not in agent.executor_prompt


def test_chat_agent_loads_skill_catalog_and_tool(
    tmp_path, chat_model, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path, "widgets", "Builds widgets", "How to build widgets.")

    agent = ChatAgent(llm=chat_model, workspace=tmp_path / "ws")
    state = agent.format_query("hi")

    assert "load_skill" in agent.tools
    assert "widgets: Builds widgets" in state["messages"][0].content


def test_chat_agent_no_skills_is_noop(tmp_path, chat_model, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    agent = ChatAgent(llm=chat_model, workspace=tmp_path / "ws")
    state = agent.format_query("hi")

    assert "load_skill" not in agent.tools
    assert "Available Skills" not in state["messages"][0].content
