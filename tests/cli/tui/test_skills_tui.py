from pathlib import Path

import pytest

from tests.cli._app_fakes import FakeHITL
from tests.tui.utils import eventually
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.widgets import (
    HotlistScreen,
    InformationScreen,
    PromptArea,
)


def write_skill(root, name, description):
    path = root / ".ursa" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nDo {name}.\n",
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
        assert await eventually(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        assert [
            c
            for c in app.screen.candidates
            if not c.startswith("skill-creation —")
        ] == ["write-python — House style for Python"]

        await pilot.press("p", "y", "enter")
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await eventually(pilot, lambda: prompt.text == "$write-python ")
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
        await eventually(
            pilot, lambda: prompt.text == "tidy this up $write-python "
        )

        await pilot.press("enter")
        await eventually(
            pilot,
            lambda: hitl.calls == [("chat", "tidy this up $write-python")],
        )
        assert hitl.calls == [("chat", "tidy this up $write-python")]


async def test_escaping_the_dollar_picker_leaves_a_literal_dollar(project):
    write_skill(project, "write-python", "House style for Python")
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("$")
        assert await eventually(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        await pilot.press("escape")
        assert await eventually(
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
        assert await eventually(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        assert [
            c
            for c in app.screen.candidates
            if not c.startswith("skill-creation —")
        ] == [
            "project-skill — From the project root",
            "user-skill — From the user root",
        ]


async def test_skills_command_shows_the_catalog(project):
    write_skill(project, "write-python", "House style for Python")
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("/", "s", "k", "i", "l", "enter")
        await pilot.pause()
        assert await eventually(
            pilot, lambda: isinstance(app.screen, InformationScreen)
        )
        assert "$write-python" in app.screen.content
        assert "House style for Python" in app.screen.content
        assert str(project / ".ursa" / "skills") in app.screen.content


async def test_skills_command_lists_bundled_skills(project):
    app = UrsaTextualApp(FakeHITL(project))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("/", "s", "k", "i", "l", "enter")
        await pilot.pause()
        assert await eventually(
            pilot, lambda: isinstance(app.screen, InformationScreen)
        )
        assert "bundled" in app.screen.content
        assert "skill-creation" in app.screen.content


async def test_picker_and_catalog_use_workspace_and_share_agent_cache(
    project, chat_model, monkeypatch
):
    from ursa.agents.chat_agent import ChatAgent
    from ursa.skills import discovery

    write_skill(project, "launch-only", "Not in workspace")
    workspace = project / "workspace"
    write_skill(workspace, "workspace-only", "Workspace skill")
    agent = ChatAgent(llm=chat_model, workspace=workspace)
    app = UrsaTextualApp(FakeHITL(workspace))
    assert app.skill_catalog is agent.skill_catalog

    def no_scan(*args, **kwargs):
        pytest.fail("The picker and page must reuse the agent catalog")

    monkeypatch.setattr(discovery, "discover_skills_in", no_scan)
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("$")
        await eventually(pilot, lambda: isinstance(app.screen, HotlistScreen))
        assert "workspace-only — Workspace skill" in app.screen.candidates
        assert all("launch-only" not in item for item in app.screen.candidates)
        await pilot.press("escape")
        await app._show_command("skills")
        await eventually(
            pilot, lambda: isinstance(app.screen, InformationScreen)
        )
        assert "workspace-only" in app.screen.content
        assert "launch-only" not in app.screen.content


async def test_long_descriptions_wrap_in_catalog(project):
    from textual.widgets import Markdown

    description = "A long description with useful details. " * 25
    write_skill(project, "long-description", description)
    app = UrsaTextualApp(FakeHITL(project))
    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("skills")
        await eventually(
            pilot, lambda: isinstance(app.screen, InformationScreen)
        )
        await pilot.pause()
        markdown = app.screen.query_one(Markdown)
        paragraphs = list(markdown.query("MarkdownParagraph"))
        assert any(paragraph.size.height > 5 for paragraph in paragraphs)
        assert description.strip() in app.screen.content
        assert "Source:" in app.screen.content
        assert "Search locations" in app.screen.content


@pytest.mark.parametrize("surface", ["picker", "page"])
async def test_slow_discovery_keeps_the_prompt_responsive(
    project, monkeypatch, surface
):
    import threading

    from tests.tui.utils import await_event

    app = UrsaTextualApp(FakeHITL(project))
    started, release = threading.Event(), threading.Event()
    load = app.skill_catalog.load

    def slow_load():
        started.set()
        assert release.wait(10), "Test did not release discovery"
        return load()

    monkeypatch.setattr(app.skill_catalog, "load", slow_load)
    async with app.run_test(size=(80, 30)) as pilot:
        try:
            if surface == "picker":
                app._open_hotlist("$")
            else:
                await app._show_command("skills")
            await await_event(pilot, started)
            await pilot.press("x")
            assert app.query_one(PromptArea).text == "x"
        finally:
            release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
