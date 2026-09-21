from pathlib import Path

import pytest

from tests.cli._app_fakes import FakeHITL, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.widgets import (
    HotlistScreen,
    InformationScreen,
    PromptArea,
)


def write_skill(root, name, description):
    path = root / ".agents" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ndescription: {description}\n---\n\nDo {name}.\n",
        encoding="utf-8",
    )


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def test_dollar_picker_inserts_only_the_skill_name(project):
    write_skill(project, "write-python", "House style for Python")
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("$")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        assert app.screen.candidates == [
            "write-python — House style for Python"
        ]

        await pilot.press("p", "y", "enter")
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await wait_for(pilot, lambda: prompt.text == "$write-python ")
        assert prompt.text == "$write-python "


async def test_dollar_picker_works_mid_prompt_and_reaches_the_agent(project):
    write_skill(project, "write-python", "House style for Python")
    hitl = FakeHITL(project)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("tidy this up ")
        prompt.move_cursor((0, 13))

        await pilot.press("$", "w", "r", "enter")
        await pilot.pause()
        await wait_for(
            pilot, lambda: prompt.text == "tidy this up $write-python "
        )

        await pilot.press("enter")
        await wait_for(
            pilot,
            lambda: hitl.calls == [("chat", "tidy this up $write-python")],
        )
        assert hitl.calls == [("chat", "tidy this up $write-python")]


async def test_escaping_the_dollar_picker_leaves_a_literal_dollar(project):
    write_skill(project, "write-python", "House style for Python")
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("$")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        await pilot.press("escape")
        assert await wait_for(
            pilot, lambda: not isinstance(app.screen, HotlistScreen)
        )

        prompt = app.query_one(PromptArea)
        assert prompt.text == "$"
        assert prompt.has_focus


async def test_dollar_picker_lists_user_and_project_skills(project):
    write_skill(Path.home(), "user-skill", "From the user root")
    write_skill(project, "project-skill", "From the project root")
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("$")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        assert app.screen.candidates == [
            "project-skill — From the project root",
            "user-skill — From the user root",
        ]


async def test_skills_command_shows_the_catalog(project):
    write_skill(project, "write-python", "House style for Python")
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("/", "s", "k", "i", "l", "enter")
        await pilot.pause()
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, InformationScreen)
        )
        assert "$write-python" in app.screen.content
        assert "House style for Python" in app.screen.content
        assert str(project / ".agents" / "skills") in app.screen.content


async def test_skills_command_explains_an_empty_catalog(project):
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("/", "s", "k", "i", "l", "enter")
        await pilot.pause()
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, InformationScreen)
        )
        assert "No skills found" in app.screen.content
        assert "skill-creation" in app.screen.content
