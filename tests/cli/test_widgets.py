import os
from pathlib import Path
from types import SimpleNamespace

from textual import events
from textual.binding import Binding
from textual.widgets import Collapsible, Markdown, Static, Tab, TabPane

from tests.cli._app_fakes import FakeHITL
from ursa.cli.app import UrsaTextualApp
from ursa.cli.tips import TIPS
from ursa.cli.widgets import (
    AgentsScreen,
    HotlistScreen,
    InformationScreen,
    PromptArea,
    WelcomeBanner,
)


class FakeToolArgs:
    @classmethod
    def model_json_schema(cls):
        return {
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                }
            },
            "required": ["path"],
        }


class FakeConfiguredTool:
    name = "read_file"
    description = "Read a file from the configured workspace."
    args_schema = FakeToolArgs
    return_direct = False
    metadata = None


class FakeMcpTool(FakeConfiguredTool):
    name = "remote_read"


async def test_agent_hotlist_routes_selected_agent(tmp_path):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("#")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)

        options = app.screen.query_one("#hotlist-options")
        assert options.highlighted == 0
        await pilot.press("p", "l")
        assert app.screen.matches == ["plan"]
        assert options.highlighted == 0
        await pilot.press("enter")
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        assert prompt.text == "#plan "

        prompt.insert("make a plan")
        await pilot.press("enter")
        await pilot.pause()
        assert hitl.calls == [("plan", "make a plan")]


async def test_agent_selection_moves_to_front_replaces_and_preserves_cursor(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("Review docs carefully")
        prompt.move_cursor((0, 6))

        await pilot.press("#", "p", "l", "enter")
        await pilot.pause()
        assert prompt.text == "#plan Review docs carefully"
        assert prompt.cursor_location == (0, 12)

        await pilot.press("#", "c", "h", "enter")
        await pilot.pause()
        assert prompt.text == "#chat Review docs carefully"
        assert prompt.cursor_location == (0, 12)


async def test_macro_selectors_close_with_escape(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("Review docs carefully")
        prompt.move_cursor((0, 6))

        await pilot.press("#")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        await pilot.press("escape")
        await pilot.pause()

        assert prompt.text == "Review# docs carefully"
        assert prompt.cursor_location == (0, 7)
        assert prompt.has_focus

        await pilot.press("ctrl+z")
        assert prompt.text == "Review docs carefully"
        await pilot.press("ctrl+y")
        await pilot.pause()
        assert prompt.text == "Review# docs carefully"
        assert not isinstance(app.screen, HotlistScreen)

        prompt.load_text("")
        await pilot.press("@")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert prompt.text == "@"
        assert prompt.has_focus


async def test_escaping_command_picker_preserves_multiline_draft_and_undo(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("alpha\nbeta")
        prompt.move_cursor((0, 0))

        await pilot.press("/")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        await pilot.press("escape")
        await pilot.pause()

        assert prompt.text == "/alpha\nbeta"
        await pilot.press("ctrl+z")
        assert prompt.text == "alpha\nbeta"
        await pilot.press("ctrl+y")
        await pilot.pause()
        assert prompt.text == "/alpha\nbeta"
        assert not isinstance(app.screen, HotlistScreen)


async def test_macro_choice_is_undoable_without_reopening_picker(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("Review docs")
        prompt.move_cursor((0, 6))

        await pilot.press("#", "p", "l", "enter")
        await pilot.pause()
        assert prompt.text == "#plan Review docs"

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert prompt.text == "Review# docs"
        assert not isinstance(app.screen, HotlistScreen)

        await pilot.press("ctrl+y")
        await pilot.pause()
        assert prompt.text == "#plan Review docs"
        assert not isinstance(app.screen, HotlistScreen)


async def test_programmatic_and_pasted_macro_characters_do_not_open_picker(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("#plan programmatic")
        await pilot.pause()
        assert not isinstance(app.screen, HotlistScreen)

        prompt.load_text("")
        app.post_message(events.Paste("@notes.md /status"))
        await pilot.pause()
        assert prompt.text == "@notes.md /status"
        assert not isinstance(app.screen, HotlistScreen)


async def test_file_hotlist_uses_at_trigger(tmp_path):
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide")
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("@")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        assert app.screen.candidates == [
            f"{Path('docs')}{os.sep}",
            str(Path("docs/guide.md")),
            "notes.md",
        ]
        options = app.screen.query_one("#hotlist-options")
        assert options.highlighted == 0

        await pilot.press("n", "o")
        assert app.screen.matches == ["notes.md"]
        assert options.highlighted == 0
        await pilot.press("enter")
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        assert prompt.text == "@notes.md "

        prompt.load_text("")
        await pilot.press("@")
        await pilot.pause()
        await pilot.press("d", "o", "c", "s", "enter")
        await pilot.pause()
        assert prompt.text == f"@{Path('docs')}{os.sep} "


async def test_shift_enter_adds_a_prompt_newline(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        await pilot.press("a", "shift+enter", "b")
        assert app.query_one(PromptArea).text == "a\nb"


async def test_prompt_has_markdown_highlighting_paste_undo_and_redo(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        assert prompt.language == "markdown"

        app.post_message(events.Paste("# heading\nbody"))
        await pilot.pause()
        assert prompt.text == "# heading\nbody"

        await pilot.press("ctrl+z")
        assert prompt.text == ""
        await pilot.press("ctrl+y")
        assert prompt.text == "# heading\nbody"


async def test_prompt_supports_option_arrow_word_navigation(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("alpha beta")
        prompt.move_cursor((0, len(prompt.text)))

        await pilot.press("alt+left")
        assert prompt.cursor_location == (0, 6)
        await pilot.press("alt+right")
        assert prompt.cursor_location == (0, 10)


async def test_ctrl_c_clears_prompt_and_adds_it_to_history(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        await pilot.press("o", "l", "d", "enter")
        await pilot.pause()
        await pilot.press("d", "r", "a", "f", "t", "ctrl+c")
        prompt = app.query_one(PromptArea)
        assert prompt.text == ""

        await pilot.press("up")
        assert prompt.text == "draft"
        await pilot.press("up")
        assert prompt.text == "old"
        await pilot.press("down")
        assert prompt.text == "draft"


async def test_prompt_caps_at_thirty_percent_of_terminal_height(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        assert prompt.region.height == 3  # One content row plus the border.

        prompt.load_text("\n".join(str(index) for index in range(30)))
        for _ in range(3):
            await pilot.pause()
            if prompt.region.height == 13:
                break
        assert prompt.region.height == 13  # ceil(36 * 0.3) plus the border.

        await pilot.resize_terminal(100, 20)
        for _ in range(3):
            await pilot.pause()
            if prompt.region.height == 8:
                break
        assert prompt.region.height == 8  # ceil(20 * 0.3) plus the border.


async def test_prompt_grows_for_soft_wrapped_lines(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(40, 24)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("word " * 40)
        for _ in range(3):
            await pilot.pause()
            if prompt.region.height > 3:
                break

        assert prompt.virtual_size.height > 1
        assert prompt.region.height == min(8, prompt.virtual_size.height) + 2


async def test_welcome_banner_and_endpoint_status_are_visible(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.model = SimpleNamespace(
        model_name="test-model", base_url="https://llm.test/v1"
    )
    hitl.embedding = SimpleNamespace(
        model="embed-model", base_url="https://embed.test/v1"
    )
    hitl.group = "research"
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        banner = app.query_one(WelcomeBanner)
        snapshot = str(
            banner.query_one("#welcome-config-values", Static).content
        )
        workspace = banner.query_one("#welcome-workspace", Static)
        assert str(workspace.content).endswith(tmp_path.name[-12:])
        assert "test-model (https://llm.test/v1)" in snapshot
        assert "embed-model (https://embed.test/v1)" in snapshot
        assert "research" in snapshot
        assert "test-model (https://llm.test/v1)" in str(
            app.query_one("#status", Static).content
        )
        assert "Ctrl+" not in str(app.query_one("#status", Static).content)


async def test_named_agent_appears_in_statusline_and_status_command(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.agent_name = "lab-assistant"
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)):
        assert "lab-assistant" in str(app.query_one("#status", Static).content)
        assert "lab-assistant" in app._status_markdown()


async def test_welcome_chooses_one_tip_from_the_catalog(tmp_path, monkeypatch):
    calls = []

    def choose(candidates):
        calls.append(candidates)
        return candidates[-1]

    monkeypatch.setattr("ursa.cli.tips.random.choice", choose)
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)):
        banner = app.query_one(WelcomeBanner)
        assert calls == [TIPS]
        assert banner.tip == TIPS[-1]
        assert TIPS[-1] in str(app.query_one("#welcome-tip", Static).content)


async def test_welcome_version_and_workspace_align_at_narrow_width(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        logo = app.query_one("#welcome-logo")
        config = app.query_one("#welcome-config")
        art = app.query_one("#welcome-logo-art", Static)
        version = app.query_one("#welcome-version", Static)
        workspace_row = app.query_one("#welcome-workspace-row")
        label = app.query_one("#welcome-workspace-label", Static)
        workspace = app.query_one("#welcome-workspace", Static)

        assert version.styles.content_align_horizontal == "right"
        assert version.region.right == logo.content_region.right
        assert version.region.y == art.region.bottom
        assert len(str(version.content)) <= version.content_region.width
        assert workspace_row.has_class("workspace-stacked")
        assert workspace.styles.content_align_horizontal == "right"
        assert workspace.styles.text_overflow == "ellipsis"
        assert workspace.region.right == config.content_region.right
        assert workspace.region.y > label.region.y
        assert len(str(workspace.content)) <= workspace.content_region.width
        assert str(workspace.content).endswith(tmp_path.name[-12:])


async def test_workspace_uses_one_borderless_row_when_it_fits(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.workspace = Path("/tmp/ursa")
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        row = app.query_one("#welcome-workspace-row")
        label = app.query_one("#welcome-workspace-label", Static)
        workspace = app.query_one("#welcome-workspace", Static)
        values = app.query_one("#welcome-config-values", Static)

        assert row.has_class("workspace-inline")
        assert label.region.y == workspace.region.y
        assert workspace.styles.content_align_horizontal == "left"
        assert workspace.region.x == label.region.right
        assert values.region.y == row.region.bottom
        assert str(workspace.content) == str(Path("/tmp/ursa").resolve())


async def test_picker_header_shares_the_top_row_with_exit_hint(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(HotlistScreen("Workspace paths", ["src/"]))
        await pilot.pause()
        hotlist = app.screen.query_one("#hotlist")
        header = app.screen.query_one("#hotlist-header")
        title = app.screen.query_one("#hotlist-title", Static)
        exit_hint = app.screen.query_one("#hotlist-exit-hint", Static)

        assert header.region.y == hotlist.content_region.y
        assert title.region.y == exit_hint.region.y
        assert str(exit_hint.content) == "Esc to Exit"


async def test_slash_picker_opens_status_inside_textual(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.agent_name = "lab-assistant"
    hitl.config = SimpleNamespace(
        mcp_servers={
            "local": {"transport": "stdio", "command": "ursa-mcp"},
            "remote": {
                "transport": "streamable-http",
                "url": "https://example.test/mcp",
            },
        }
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        hotlist = app.screen.query_one("#hotlist")
        options = app.screen.query_one("#hotlist-options")
        assert hotlist.region.width == 80
        assert options.region.height >= 3
        assert options.region.bottom <= hotlist.region.bottom
        screenshot = app.export_screenshot()
        assert all(
            choice in screenshot for choice in ("agents", "status", "keymap")
        )
        assert [
            candidate.partition(" — ")[0] for candidate in app.screen.candidates
        ] == ["agents", "status", "keymap"]

        await pilot.press("s", "t", "a", "t", "u", "s", "enter")
        await pilot.pause()
        assert isinstance(app.screen, InformationScreen)
        assert "LLM endpoint" in app.screen.content
        assert "lab-assistant" in app.screen.content
        assert "MCP servers" in app.screen.content
        assert "ursa-mcp" in app.screen.content
        assert "https://example.test/mcp" in app.screen.content


async def test_command_picker_prioritizes_command_name_over_description(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", "k", "e")
        await pilot.pause()

        assert isinstance(app.screen, HotlistScreen)
        assert app.screen.matches[0].startswith("keymap —")
        assert any(match.startswith("status —") for match in app.screen.matches)
        assert app.screen.query_one("#hotlist-options").highlighted == 0


async def test_agents_command_uses_tabs_and_collapsed_tool_details(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.agents = {
        "plan": hitl.agents["plan"],
        "chat": hitl.agents["chat"],
    }
    for agent in hitl.agents.values():
        agent.tools = {
            "read_file": FakeConfiguredTool(),
            "remote_read": FakeMcpTool(),
        }
        agent.tool_sources = {"remote_read": "laboratory"}
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("/", "a", "g", "e", "n", "t", "s", "enter")
        await pilot.pause()

        assert isinstance(app.screen, AgentsScreen)
        panes = list(app.screen.query(TabPane))
        assert len(panes) == 2
        assert [tab.label_text for tab in app.screen.query(Tab)] == [
            "#plan",
            "#chat",
        ]
        tools = list(app.screen.query(Collapsible))
        assert len(tools) == 4
        assert all(tool.collapsed for tool in tools)
        assert [str(tool.title) for tool in tools[:2]] == [
            "read_file",
            "remote_read (mcp: laboratory)",
        ]

        tools[0].collapsed = False
        await pilot.pause()
        detail = str(tools[0].query_one(Markdown).source)
        assert "Read a file from the configured workspace." in detail
        assert "FakeConfiguredTool" in detail
        assert "FakeToolArgs" in detail
        assert "Workspace-relative file path." in detail


def test_command_details_and_keymap_come_from_live_bindings(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    app.total_tokens = 1234
    app.input_tokens = 1000
    app.output_tokens = 234
    app.cached_tokens = 456
    monkeypatch.setattr(
        PromptArea,
        "BINDINGS",
        [
            *PromptArea.BINDINGS,
            Binding("f12", "diagnostics", "Open diagnostics", show=False),
        ],
    )

    status = app._status_markdown()
    keymap = app._keymap_markdown()

    assert "1,234" in status
    assert "1,000" in status
    assert "234" in status
    assert "456" in status
    assert "test-model" in status
    for expected in (
        "## Application",
        "## Prompt editor",
        "## Picker",
        "## Information screen",
        "Submit prompt",
        "Choose workspace path",
        "Previous choice",
        "Scroll to bottom",
        "Open diagnostics",
    ):
        assert expected in keymap
