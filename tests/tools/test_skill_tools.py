from pathlib import Path

from langchain.chat_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from tests.tools.utils import invoke_with_event_recorder, make_runtime
from ursa.tools.skill_tools import load_skill


def _write_skill(cwd: Path, name: str, description: str, body: str) -> None:
    skill_dir = cwd / ".ursa" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_load_skill_returns_body(
    tmp_path: Path,
    chat_model: BaseChatModel,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    # Point HOME elsewhere so the global root does not interfere.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path, "greeter", "Greets the user", "Say hello nicely.")

    runtime = make_runtime(tmp_path, llm=chat_model)
    result, recorder = invoke_with_event_recorder(
        load_skill.func,
        name="greeter",
        runtime=runtime,
    )

    assert "Say hello nicely." in result
    # The loaded body is prefixed with the skill directory so relative script
    # paths and bundled files resolve correctly.
    skill_dir = tmp_path / ".ursa" / "skills" / "greeter"
    assert str(skill_dir) in result
    _, event = recorder.events[-1]
    assert event["message"] == "Skill loaded"
    assert event["name"] == "greeter"


def test_load_skill_unknown_lists_available(
    tmp_path: Path,
    chat_model: BaseChatModel,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path, "greeter", "Greets the user", "Say hello.")

    runtime = make_runtime(tmp_path, llm=chat_model)
    result, recorder = invoke_with_event_recorder(
        load_skill.func,
        name="nope",
        runtime=runtime,
    )

    assert "not found" in result
    assert "greeter" in result
    _, event = recorder.events[-1]
    assert event["phase"] == "error"


def test_load_skill_repeat_load_is_short_circuited(
    tmp_path: Path,
    chat_model: BaseChatModel,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path, "greeter", "Greets the user", "Say hello nicely.")

    store = InMemoryStore()
    runtime = make_runtime(tmp_path, llm=chat_model, store=store)

    first, _ = invoke_with_event_recorder(
        load_skill.func, name="greeter", runtime=runtime
    )
    assert "Say hello nicely." in first

    second, recorder = invoke_with_event_recorder(
        load_skill.func, name="greeter", runtime=runtime
    )
    assert "already loaded" in second
    _, event = recorder.events[-1]
    assert event["message"] == "Skill already loaded"
