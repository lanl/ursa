import asyncio
import os
import random
import threading
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import yaml
from mcp import StdioServerParameters
from mcp.client.session_group import StreamableHttpParameters
from pydantic import SecretStr
from textual import events
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.theme import BUILTIN_THEMES
from textual.widgets import (
    Collapsible,
    Input,
    Markdown,
    Select,
    Static,
    Tab,
    TabbedContent,
    TabPane,
    TextArea,
)

import ursa.util.crossplatform as crossplatform
from tests.cli._app_fakes import FakeHITL, wait_for
from ursa.agents.base import AgentWithTools
from ursa.agents.execution_agent import ExecutionAgent
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
)
from ursa.cli.runtime import AgentHITL
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.tips import TIPS, random_tip, runtime_keymap
from ursa.cli.tui.widgets import (
    AgentsScreen,
    FuzzySelectOverlay,
    HotlistScreen,
    InformationScreen,
    ModelScreen,
    ModelSelection,
    PromptArea,
    ThemeScreen,
    ToolMessage,
    WelcomeBanner,
)
from ursa.util.inference_providers import ProviderModel


async def wait_for_yaml_debounce(pilot) -> None:
    """Let the YAML validation timer fire, then flush resulting messages."""
    await asyncio.sleep(ModelScreen.YAML_VALIDATION_DELAY + 0.1)
    await pilot.pause()


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


def test_model_select_type_search_is_fuzzy():
    overlay = FuzzySelectOverlay()
    overlay.add_options(["gpt-4", "gpt-5.4", "text-embedding-3-large"])

    assert overlay._find_search_match("g54") == 1


class FakeMcpTool(FakeConfiguredTool):
    name = "remote_read"


async def test_agent_hotlist_routes_selected_agent(tmp_path):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("#")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )

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
        await wait_for(pilot, lambda: hitl.calls == [("plan", "make a plan")])
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
        await wait_for(
            pilot, lambda: prompt.text == "#plan Review docs carefully"
        )
        assert prompt.text == "#plan Review docs carefully"
        assert prompt.cursor_location == (0, 12)

        await pilot.press("#", "c", "h", "enter")
        await pilot.pause()
        await wait_for(
            pilot, lambda: prompt.text == "#chat Review docs carefully"
        )
        assert prompt.text == "#chat Review docs carefully"
        assert prompt.cursor_location == (0, 12)


async def test_macro_selectors_close_with_escape(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("Review docs carefully")
        prompt.move_cursor((0, 6))

        await pilot.press("#")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        await pilot.press("escape")
        assert await wait_for(
            pilot, lambda: not isinstance(app.screen, HotlistScreen)
        )

        assert prompt.text == "Review# docs carefully"
        assert prompt.cursor_location == (0, 7)
        assert prompt.has_focus

        await pilot.press("ctrl+z")
        assert prompt.text == "Review docs carefully"
        await pilot.press("ctrl+y")
        await pilot.pause()
        await wait_for(pilot, lambda: prompt.text == "Review# docs carefully")
        assert prompt.text == "Review# docs carefully"
        assert not isinstance(app.screen, HotlistScreen)

        prompt.load_text("")
        await pilot.press("@")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        await pilot.press("escape")
        assert await wait_for(
            pilot, lambda: not isinstance(app.screen, HotlistScreen)
        )
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
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        await pilot.press("escape")
        await pilot.pause()

        await wait_for(pilot, lambda: prompt.text == "/alpha\nbeta")
        assert prompt.text == "/alpha\nbeta"
        await pilot.press("ctrl+z")
        assert prompt.text == "alpha\nbeta"
        await pilot.press("ctrl+y")
        await pilot.pause()
        await wait_for(pilot, lambda: prompt.text == "/alpha\nbeta")
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
        await wait_for(pilot, lambda: prompt.text == "#plan Review docs")
        assert prompt.text == "#plan Review docs"

        await pilot.press("ctrl+z")
        await pilot.pause()
        await wait_for(pilot, lambda: prompt.text == "Review# docs")
        assert prompt.text == "Review# docs"
        assert not isinstance(app.screen, HotlistScreen)

        await pilot.press("ctrl+y")
        await pilot.pause()
        await wait_for(pilot, lambda: prompt.text == "#plan Review docs")
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
        await wait_for(pilot, lambda: not isinstance(app.screen, HotlistScreen))
        assert not isinstance(app.screen, HotlistScreen)

        prompt.load_text("")
        app.post_message(events.Paste("@notes.md /status"))
        await pilot.pause()
        await wait_for(pilot, lambda: prompt.text == "@notes.md /status")
        assert prompt.text == "@notes.md /status"
        assert not isinstance(app.screen, HotlistScreen)


async def test_file_hotlist_uses_at_trigger(tmp_path):
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide")
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("@")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
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
        await wait_for(
            pilot, lambda: prompt.text == f"@{Path('docs')}{os.sep} "
        )
        assert prompt.text == f"@{Path('docs')}{os.sep} "


async def test_shift_enter_adds_a_prompt_newline(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        await pilot.press("a", "shift+enter", "b")
        assert app.query_one(PromptArea).text == "a\nb"


async def test_ctrl_j_adds_a_prompt_newline(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        await pilot.press("a", "ctrl+j", "b")
        assert app.query_one(PromptArea).text == "a\nb"


def test_newline_key_prefers_shift_enter_when_protocol_is_detected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(crossplatform, "expects_kitty_keyboard", lambda: True)
    app = UrsaTextualApp(FakeHITL(tmp_path))

    assert app.preferred_newline_key == "shift+enter"


def test_newline_key_falls_back_to_ctrl_j(tmp_path, monkeypatch):
    monkeypatch.setattr(crossplatform, "expects_kitty_keyboard", lambda: False)
    app = UrsaTextualApp(FakeHITL(tmp_path))

    assert app.preferred_newline_key == "ctrl+j"


async def test_prompt_has_markdown_highlighting_paste_undo_and_redo(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        assert prompt.language == "markdown"

        app.post_message(events.Paste("# heading\nbody"))
        await pilot.pause()
        await wait_for(pilot, lambda: prompt.text == "# heading\nbody")
        assert prompt.text == "# heading\nbody"

        await pilot.press("ctrl+z")
        assert prompt.text == ""
        await pilot.press("ctrl+y")
        assert prompt.text == "# heading\nbody"


async def test_prompt_copy_shortcut_preserves_selected_text(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("selected text")
        prompt.action_select_all()

        await pilot.press("super+c")

        assert copied == ["selected text"]
        assert prompt.text == "selected text"


async def test_prompt_copy_shortcut_delegates_to_screen_selection(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    delegated = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app.screen, "action_copy_text", lambda: delegated.append(True)
        )

        await pilot.press("ctrl+shift+c")

        assert delegated == [True]


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


async def test_welcome_banner_and_provider_status_are_visible(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.config.inference_providers.update({
        "hosted-chat": InferenceProviderConfig(base_url="https://llm.test/v1"),
        "hosted-embedding": InferenceProviderConfig(
            base_url="https://embed.test/v1"
        ),
    })
    hitl.config.llm_model = ChatModelConfig(
        model="test-model", inference_provider="hosted-chat"
    ).resolve_inference_provider(hitl.config.inference_providers)
    hitl.config.emb_model = EmbModelConfig(
        model="embed-model", inference_provider="hosted-embedding"
    ).resolve_inference_provider(hitl.config.inference_providers)
    hitl.inference_provider = "hosted-chat"
    hitl.embedding_inference_provider = "hosted-embedding"
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
        assert "test-model (hosted-chat - https://llm.test/v1)" in snapshot
        assert (
            "embed-model (hosted-embedding - https://embed.test/v1)" in snapshot
        )
        assert "research" in snapshot
        assert "test-model (hosted-chat)" in str(
            app.query_one("#status", Static).content
        )
        assert "Ctrl+" not in str(app.query_one("#status", Static).content)


async def test_named_agent_appears_in_statusline_and_status_command(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.agent_name = "lab-assistant"
    hitl.config.agent_name = "lab-assistant"
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)):
        assert "lab-assistant" in str(app.query_one("#status", Static).content)
        assert "lab-assistant" in app._status_markdown()


async def test_welcome_tips_vary_and_keymaps_resolve(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)):
        owners = (type(app), PromptArea, HotlistScreen)
        keymap = runtime_keymap(app, owners)
        assert all(tip.format_map(keymap) for tip in TIPS)
        assert len({random_tip(app, owners) for _ in range(100)}) > 1


async def test_welcome_tip_border_uses_visible_theme_border(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        tip = app.query_one("#welcome-tip", Static)
        for theme_name in ("ursa-dark", "ursa-light"):
            app.theme = theme_name
            await pilot.pause()

            theme_colors = (
                app.get_theme(theme_name).to_color_system().generate()
            )
            border_style, border_color = tip.styles.border.top
            assert border_style == "round"
            assert border_color.hex == theme_colors["border"]
            assert border_color != app.screen.styles.background


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
    hitl.config.agent_name = "lab-assistant"
    hitl.config.llm_model.api_key = SecretStr("actual-secret")
    hitl.config.mcp_servers = {
        "local": StdioServerParameters(command="ursa-mcp", args=[]),
        "remote": StreamableHttpParameters(url="https://example.test/mcp"),
        **{
            f"extra-{index}": StdioServerParameters(
                command=f"ursa-mcp-{index}", args=[]
            )
            for index in range(20)
        },
    }
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")
        assert await wait_for(
            pilot, lambda: isinstance(app.screen, HotlistScreen)
        )
        hotlist = app.screen.query_one("#hotlist")
        options = app.screen.query_one("#hotlist-options")
        assert hotlist.region.width == 80
        assert options.region.height >= 3
        assert options.region.bottom <= hotlist.region.bottom
        screenshot = app.export_screenshot()
        assert "agents" in screenshot
        assert [
            candidate.partition(" — ")[0] for candidate in app.screen.candidates
        ] == [
            "agents",
            "exit",
            "status",
            "terms",
            "keymap",
            "models",
            "theme",
        ]

        await pilot.press("s", "t", "a", "t", "u", "s", "enter")
        await pilot.pause()
        await wait_for(pilot, lambda: isinstance(app.screen, InformationScreen))
        assert isinstance(app.screen, InformationScreen)
        assert "LLM Endpoint" in app.screen.content
        assert "lab-assistant" in app.screen.content
        assert "MCP servers" in app.screen.content
        assert "ursa-mcp" in app.screen.content
        assert "https://example.test/mcp" in app.screen.content

        tabs = {str(tab.label): tab for tab in app.screen.query(Tab)}
        await pilot.click(f"#{tabs['Config'].id}")
        await pilot.press("tab")
        await pilot.pause()
        editor = app.screen.query_one("#status-config-yaml", TextArea)
        assert app.focused is editor
        assert editor.read_only
        assert editor.language == "yaml"
        assert type(editor.document).__name__ == "SyntaxAwareDocument"
        assert editor._highlight_query is not None
        assert "llm_model:" in editor.text
        assert "model: test-model" in editor.text
        assert "env: OPENAI_API_KEY" in editor.text
        assert "**********" in editor.text
        assert "actual-secret" not in editor.text
        assert str(
            app.screen.query_one("#status-config-readonly", Static).content
        ).startswith("Read only")
        await pilot.press("down", "shift+down")
        assert editor.cursor_location[0] == 2
        assert not editor.selection.is_empty

        await pilot.click(f"#{tabs['Status'].id}")
        await pilot.press("tab")
        body = app.screen.query_one("#information-body", VerticalScroll)
        assert app.focused is body
        assert body.scroll_y == 0
        await pilot.press("end")
        await pilot.pause()
        await wait_for(pilot, lambda: body.scroll_y > 0)
        assert body.scroll_y > 0

        await pilot.press("escape")
        await pilot.pause()
        await wait_for(
            pilot, lambda: not isinstance(app.screen, InformationScreen)
        )
        assert not isinstance(app.screen, InformationScreen)


async def test_exit_command_quits_the_app(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", "e", "x", "i", "t", "enter")
        await pilot.pause()

        await wait_for(pilot, lambda: app._exit)
        assert app._exit


async def test_model_command_switches_provider_and_model(tmp_path, monkeypatch):
    hitl = FakeHITL(tmp_path)
    hitl.config.llm_model.max_completion_tokens = 4096
    hitl.inference_provider = "stale-provider"
    hitl.config.inference_providers["stale-provider"] = InferenceProviderConfig(
        base_url="https://stale.example/v1"
    )

    def provider_models(config):
        if config.base_url == "https://stale.example/v1":
            return [ProviderModel("claude-stale", "anthropic")]
        return [
            ProviderModel("gpt-5.4", "openai"),
            ProviderModel("text-embedding-3-large", "openai", type="embedding"),
        ]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models",
        provider_models,
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        assert (
            app.screen.query_one("#chat-inference-provider", Select).value
            == "openai"
        )
        assert app.screen.query_one(
            "#chat-inference-provider", Select
        )._options == [
            ("None (direct model config)", ModelScreen.NONE_VALUE),
            ("openai (https://api.openai.com/v1)", "openai"),
            ("stale-provider (https://stale.example/v1)", "stale-provider"),
        ]
        assert (
            app.screen.query_one("#chat-model-label").tooltip
            == (ModelScreen.FIELD_HELP["model"])
        )
        assert (
            app.screen.query_one("#chat-model-provider-label").tooltip
            == ModelScreen.FIELD_HELP["model-provider"]
        )
        assert (
            app.screen.query_one("#chat-inference-provider-label").tooltip
            == ModelScreen.FIELD_HELP["inference-provider"]
        )
        assert app.screen.query_one("#chat-model-name", Select)._options == [
            ("None", ModelScreen.NONE_VALUE),
            ("gpt-5.4", "gpt-5.4"),
            ("text-embedding-3-large", "text-embedding-3-large"),
            ("Not found: test-model", "test-model"),
            ("Other…", ModelScreen.CUSTOM_VALUE),
        ]
        model_select = app.screen.query_one("#chat-model-name", Select)
        model_select.focus()
        await pilot.press("enter", "g")
        await asyncio.sleep(0.8)
        await pilot.press("5", "4")
        fuzzy_options = model_select.query_one(FuzzySelectOverlay)
        assert str(model_select.query_one("#label", Static).content) == "g54"
        assert fuzzy_options.border_title is None
        assert fuzzy_options.option_count == 1
        await pilot.press("escape")
        assert isinstance(app.screen, ModelScreen)
        assert not model_select.expanded
        model_select.focus()
        await pilot.press("enter")
        assert model_select.expanded
        await pilot.click("#chat-model-provider")
        assert not model_select.expanded
        model_select.focus()
        await pilot.press("enter", "g", "5", "4")
        assert fuzzy_options.option_count == 1
        assert fuzzy_options.highlighted == 0
        assert fuzzy_options.get_option_at_index(0).id == "1"
        await pilot.press("enter")
        assert model_select.value == "gpt-5.4"
        app.screen.query_one(
            "#chat-inference-provider", Select
        ).value = "stale-provider"
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.screen.query_one("#chat-model-name", Select)._options == [
            ("None", ModelScreen.NONE_VALUE),
            ("claude-stale", "claude-stale"),
            ("Not found: gpt-5.4", "gpt-5.4"),
            ("Other…", ModelScreen.CUSTOM_VALUE),
        ]
        assert (
            app.screen.query_one("#chat-model-name", Select).value == "gpt-5.4"
        )
        assert app.screen.query_one("#chat-model-name-custom", Input).has_class(
            "hidden"
        )
        app.screen.query_one("#chat-model-name", Select).value = "claude-stale"
        await pilot.pause()
        await wait_for(
            pilot,
            lambda: app.screen.query_one("#chat-model-provider", Select).value
            == "anthropic",
        )
        assert (
            app.screen.query_one("#chat-model-provider", Select).value
            == "anthropic"
        )
        assert app.screen.query_one(
            "#embedding-model-provider", Select
        )._options[0] == ("None", ModelScreen.NONE_VALUE)
        app.screen.query_one(
            "#chat-inference-provider", Select
        ).value = "openai"
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.screen.query_one("#chat-model-name", Select).value = "gpt-5.4"
        await pilot.pause()
        app.screen.query_one(
            "#embedding-model-name", Select
        ).value = ModelScreen.CUSTOM_VALUE
        app.screen.query_one(
            "#embedding-model-name-custom", Input
        ).value = "private-embedding"
        app.screen.query_one(
            "#embedding-model-provider", Select
        ).value = ModelScreen.NONE_VALUE
        await pilot.press("ctrl+enter")
        await pilot.pause()
        await app.workers.wait_for_complete()

        messages = list(app.query(ToolMessage))
        assert [message.content for message in messages[-2:]] == [
            "Changed the chat model to gpt-5.4 "
            "(openai - https://api.openai.com/v1)",
            "Changed the embedding model to private-embedding "
            "(openai - https://api.openai.com/v1)",
        ]
        welcome = str(app.query_one("#welcome-config-values", Static).content)
        assert (
            "LLM        gpt-5.4 (openai - https://api.openai.com/v1)" in welcome
        )
        assert (
            "Embedding  private-embedding "
            "(openai - https://api.openai.com/v1)" in welcome
        )

    assert len(hitl.model_changes) == 1
    chat_config, embedding_config = hitl.model_changes[0]
    assert isinstance(chat_config, ChatModelConfig)
    assert chat_config.model == "gpt-5.4"
    assert chat_config.model_provider == "openai"
    assert chat_config.inference_provider == "openai"
    assert chat_config.max_completion_tokens == 4096
    assert isinstance(embedding_config, EmbModelConfig)
    assert embedding_config.model == "private-embedding"
    assert embedding_config.model_provider is None
    assert embedding_config.inference_provider == "openai"


async def test_invalid_model_selection_notifies_without_closing_modal(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    notifications = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )

    def invalid_settings(*_args):
        raise ValueError("invalid model configuration")

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        notifications.clear()
        assert isinstance(app.screen, ModelScreen)
        monkeypatch.setattr(
            app.screen,
            "_settings",
            invalid_settings,
        )

        app.screen.action_apply()

        assert isinstance(app.screen, ModelScreen)
        assert notifications == [
            (
                "invalid model configuration",
                {
                    "title": "Model not changed",
                    "severity": "error",
                    "timeout": 10,
                    "markup": False,
                },
            )
        ]


async def test_model_yaml_round_trips_extras_and_updates_controls(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models",
        lambda _config: [ProviderModel("yaml-model", "openai")],
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        assert type(editor.document).__name__ == "SyntaxAwareDocument"
        assert editor._highlight_query is not None
        editor.text = """\
temperature: 0.25
model: yaml-model
model_provider: openai
inference_provider: openai
provider_options:
  reasoning: high
"""
        await pilot.pause()
        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await wait_for_yaml_debounce(pilot)

        assert editor.has_class("yaml-valid")
        assert (
            app.screen.query_one("#chat-model-name", Select).value
            == "yaml-model"
        )
        assert app.screen.drafts["chat"].model_extra == {
            "temperature": 0.25,
            "provider_options": {"reasoning": "high"},
        }

        app.screen.query_one(
            "#chat-model-name", Select
        ).value = ModelScreen.CUSTOM_VALUE
        app.screen.query_one(
            "#chat-model-name-custom", Input
        ).value = "changed-model"
        await pilot.pause()

        await wait_for(pilot, lambda: "model: changed-model" in editor.text)
        assert "model: changed-model" in editor.text
        assert "temperature: 0.25" in editor.text
        assert "reasoning: high" in editor.text
        assert editor.text.index("temperature:") < editor.text.index("model:")


async def test_valid_yaml_is_committed_only_when_apply_is_pressed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    hitl = FakeHITL(tmp_path)
    original = hitl.config.llm_model.model_copy(deep=True)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = """\
model: applied-model
model_provider: openai
inference_provider: openai
temperature: 0.35
provider_options:
  reasoning: high
"""
        assert hitl.config.llm_model == original

        app.screen.action_apply()
        await app.workers.wait_for_complete()
        await pilot.pause()

        await wait_for(
            pilot, lambda: hitl.config.llm_model.model == "applied-model"
        )
        assert hitl.config.llm_model.model == "applied-model"
        assert hitl.config.llm_model.model_extra == {
            "temperature": 0.35,
            "provider_options": {"reasoning": "high"},
        }


async def test_yaml_validation_keeps_unavailable_model_as_not_found(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models",
        lambda _config: [ProviderModel("available-model", "openai")],
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = """\
model: unavailable-model
model_provider: openai
inference_provider: openai
"""

        await pilot.pause()
        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await wait_for_yaml_debounce(pilot)

        model_select = app.screen.query_one("#chat-model-name", Select)
        assert model_select.value == "unavailable-model"
        assert (
            "Not found: unavailable-model",
            "unavailable-model",
        ) in model_select._options
        assert app.screen.query_one("#chat-model-name-custom", Input).has_class(
            "hidden"
        )


async def test_yaml_keeps_unavailable_model_provider_visible(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = """\
model: custom-model
model_provider: future_provider
inference_provider: openai
"""

        await pilot.pause()
        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await wait_for_yaml_debounce(pilot)

        provider = app.screen.query_one("#chat-model-provider", Select)
        assert provider.value == "future_provider"
        assert (
            "Not found: future_provider",
            "future_provider",
        ) in provider._options
        assert app.screen.drafts["chat"].model_provider == "future_provider"

        editor.text = """\
model: custom-model
model_provider: another_future_provider
inference_provider: openai
"""
        await pilot.pause()
        await wait_for_yaml_debounce(pilot)

        assert provider.value == "another_future_provider"
        assert (
            "Not found: another_future_provider",
            "another_future_provider",
        ) in provider._options
        assert (
            "Not found: future_provider",
            "future_provider",
        ) not in provider._options

        editor.text = """\
model: custom-model
model_provider: openai
inference_provider: openai
"""
        await pilot.pause()
        await wait_for_yaml_debounce(pilot)

        assert provider.value == "openai"
        assert not any(
            label.startswith("Not found:") for label, _ in provider._options
        )


async def test_whitespace_chat_model_yaml_is_rejected_before_apply(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = "model: '   '\nmodel_provider: openai\n"

        app.screen.action_apply()

        assert isinstance(app.screen, ModelScreen)
        assert editor.has_class("yaml-invalid")
        assert "Chat model must not be blank" in str(
            app.screen.query_one("#chat-yaml-error", Static).content
        )


async def test_whitespace_embedding_model_yaml_is_rejected_before_apply(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#embedding-config-yaml", TextArea)
        editor.text = "model: '   '\nmodel_provider: openai\n"

        app.screen.action_apply()

        assert isinstance(app.screen, ModelScreen)
        assert editor.has_class("yaml-invalid")
        assert "Embedding model must not be blank" in str(
            app.screen.query_one("#embedding-yaml-error", Static).content
        )

        editor.text = "model: ''\nmodel_provider: openai\n"
        app.screen.action_apply()
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert hitl.config.emb_model is None


async def test_structured_chat_none_updates_yaml_and_blocks_apply(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    hitl = FakeHITL(tmp_path)
    original = hitl.config.llm_model.model_copy(deep=True)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one(
            "#chat-model-name", Select
        ).value = ModelScreen.NONE_VALUE
        await pilot.pause()

        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        assert yaml.safe_load(editor.text)["model"] == ""
        assert app.screen.drafts["chat"].model == ""

        app.screen.action_apply()

        assert isinstance(app.screen, ModelScreen)
        assert editor.has_class("yaml-invalid")
        assert hitl.config.llm_model == original


async def test_structured_embedding_none_removes_configured_embedding(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one(
            "#embedding-model-name", Select
        ).value = ModelScreen.NONE_VALUE
        await pilot.pause()

        editor = app.screen.query_one("#embedding-config-yaml", TextArea)
        assert yaml.safe_load(editor.text)["model"] == ""
        assert app.screen.drafts["embedding"].model == ""
        assert editor.has_class("yaml-valid")

        app.screen.action_apply()
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert hitl.config.emb_model is None


async def test_yaml_validation_names_unknown_inference_provider(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = "model: test-model\ninference_provider: missing\n"

        app.screen.action_apply()

        assert isinstance(app.screen, ModelScreen)
        assert editor.has_class("yaml-invalid")
        assert (
            str(app.screen.query_one("#chat-yaml-error", Static).content)
            == "Unknown inference_provider 'missing'"
        )


async def test_model_option_refresh_preserves_sync_guard_and_empty_choice(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)

        app.screen._syncing_controls = True
        app.screen._set_model_options("embedding", {}, "")

        options = app.screen.query_one("#embedding-model-name", Select)._options
        assert app.screen._syncing_controls is True
        assert not any(label.startswith("Not found:") for label, _ in options)


async def test_programmatic_control_sync_suppresses_queued_events(
    tmp_path, monkeypatch
):
    discoveries = []

    def provider_models(config):
        discoveries.append(config)
        return []

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        discoveries.clear()
        updated = app.screen.drafts["chat"].model_copy(
            update={"model": "programmatic-model"}
        )
        app.screen.drafts["chat"] = updated
        app.screen._yaml_values["chat"] = app.screen._configured_values(updated)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = app.screen._yaml_text("chat")

        app.screen._update_controls_from_config("chat", updated)
        await pilot.pause()

        await wait_for(pilot, lambda: app.screen.drafts["chat"] is updated)
        assert app.screen.drafts["chat"] is updated
        assert editor.text == app.screen._yaml_text("chat")
        assert discoveries == []


async def test_masked_yaml_api_key_preserves_secret_value(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _: []
    )
    hitl = FakeHITL(tmp_path)
    hitl.config.llm_model = ChatModelConfig(
        model="secret-model",
        model_provider="openai",
        base_url="https://secret.example/v1",
        api_key=SecretStr("actual-secret"),
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        assert "actual-secret" not in editor.text
        assert "api_key: '**********'" in editor.text

        validated = app.screen._validate_yaml("chat", update_controls=True)

        assert isinstance(validated, ChatModelConfig)
        assert isinstance(validated.api_key, SecretStr)
        assert validated.api_key.get_secret_value() == "actual-secret"
        assert app.screen._yaml_values["chat"]["api_key"] == "**********"


async def test_direct_provider_selection_refreshes_model_catalog(
    tmp_path, monkeypatch
):
    calls = []

    def provider_models(config):
        calls.append(config)
        if isinstance(config, ChatModelConfig):
            return [ProviderModel("direct-only", "openai")]
        return [ProviderModel("provider-only", "openai")]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)

        app.screen.query_one(
            "#chat-inference-provider", Select
        ).value = ModelScreen.NONE_VALUE
        await pilot.pause()
        await app.workers.wait_for_complete()

        options = app.screen.query_one("#chat-model-name", Select)._options
        assert ("direct-only", "direct-only") in options
        assert ("provider-only", "provider-only") not in options
        assert any(
            isinstance(config, ChatModelConfig)
            and config.inference_provider is None
            for config in calls
        )


async def test_yaml_provider_change_refreshes_model_catalog(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.inference_providers["fast"] = InferenceProviderConfig(
        base_url="https://fast.example/v1"
    )

    def provider_models(config):
        name = (
            "fast-only"
            if config.base_url == "https://fast.example/v1"
            else "old-only"
        )
        return [ProviderModel(name, "openai")]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = """\
model: test-model
model_provider: openai
inference_provider: fast
"""

        await pilot.pause()
        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await wait_for_yaml_debounce(pilot)
        await app.workers.wait_for_complete()

        assert (
            app.screen.query_one("#chat-inference-provider", Select).value
            == "fast"
        )
        assert "fast-only" in app.screen.model_catalogs["chat"]
        assert "old-only" not in app.screen.model_catalogs["chat"]


async def test_provider_discovery_failure_clears_previous_catalog(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.inference_providers["broken"] = InferenceProviderConfig(
        base_url="https://broken.example/v1"
    )

    def provider_models(config):
        if config.base_url == "https://broken.example/v1":
            raise RuntimeError("discovery failed")
        return [ProviderModel("old-only", "openai")]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        assert "old-only" in app.screen.model_catalogs["chat"]

        app.screen.query_one(
            "#chat-inference-provider", Select
        ).value = "broken"
        await pilot.pause()
        await app.workers.wait_for_complete()

        select = app.screen.query_one("#chat-model-name", Select)
        assert app.screen.model_catalogs["chat"] == {}
        assert ("old-only", "old-only") not in select._options
        assert ("Not found: test-model", "test-model") in select._options


async def test_expanded_advanced_modal_is_scrollable_on_short_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one("#chat-advanced", Collapsible).collapsed = False
        await pilot.pause()

        dialog = app.screen.query_one(".settings-dialog")
        assert dialog.region.y >= 0
        assert dialog.region.bottom <= app.screen.size.height
        assert dialog.max_scroll_y > 0

        dialog.scroll_end(animate=False)
        await pilot.pause()
        actions = app.screen.query_one(".settings-actions")
        assert actions.region.y >= 0
        assert actions.region.bottom <= app.screen.size.height


async def test_advanced_yaml_seeded_fuzz_never_mutates_running_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    hitl = FakeHITL(tmp_path)
    original = hitl.config.llm_model.model_copy(deep=True)
    app = UrsaTextualApp(hitl)
    cases = [
        ("model: valid\nmodel_provider: openai\n", True),
        ("model: [broken", False),
        ("- model\n- list\n", False),
        ("null\n", False),
        ("model: valid\ninference_provider: missing\n", False),
        (
            "model: valid\nbase_url: https://example.test\n"
            "inference_provider: openai\n",
            False,
        ),
        ("model: valid\napi_key:\n  env:\n", False),
        (
            "advanced_first:\n  nested: [one, two]\n"
            "model: valid\ntemperature: 0.4\n",
            True,
        ),
        ("model: ''\n", False),
    ]
    random.Random(20260826).shuffle(cases)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        advanced = app.screen.query_one("#chat-advanced", Collapsible)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        assert advanced.collapsed
        assert editor.language == "yaml"

        for document, expected_valid in cases:
            editor.text = document
            await pilot.pause()
            assert not editor.has_class("yaml-valid", "yaml-invalid")
            app.screen._yaml_timers["chat"].stop()

            result = app.screen._validate_yaml(
                "chat", update_controls=expected_valid
            )

            assert (result is not None) is expected_valid
            assert editor.has_class(
                "yaml-valid" if expected_valid else "yaml-invalid"
            )
            assert hitl.config.llm_model == original
            if result is not None:
                expected = yaml.safe_load(document)
                assert result.model == expected["model"]
                assert app.screen.drafts["chat"] == result
                expected_choice = result.model or ModelScreen.NONE_VALUE
                assert (
                    app.screen.query_one("#chat-model-name", Select).value
                    == expected_choice
                )
                assert result.model_extra == {
                    key: value
                    for key, value in expected.items()
                    if key
                    not in {
                        "model",
                        "model_provider",
                        "base_url",
                        "api_key",
                        "inference_provider",
                        "ssl_verify",
                        "max_completion_tokens",
                    }
                }
            await pilot.pause()


async def test_cancel_discards_yaml_with_pending_validation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ModelScreen, "YAML_VALIDATION_DELAY", 30.0)
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    hitl = FakeHITL(tmp_path)
    original = hitl.config.llm_model.model_copy(deep=True)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one(
            "#chat-config-yaml", TextArea
        ).text = "model: changed-before-cancel\n"
        await pilot.pause()
        # Flush the Changed event after the editor has emitted it to the screen.
        await pilot.pause()
        timer = app.screen._yaml_timers["chat"]
        assert timer._task is not None
        assert not timer._task.done()

        app.screen.action_cancel()
        await pilot.pause()

        await wait_for(pilot, lambda: not isinstance(app.screen, ModelScreen))
        assert not isinstance(app.screen, ModelScreen)
        assert hitl.config.llm_model == original
        assert timer._task is None


async def test_yaml_debounce_restarts_from_latest_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = "model: first-edit\n"
        await pilot.pause()
        await asyncio.sleep(ModelScreen.YAML_VALIDATION_DELAY / 2)

        editor.text = "model: second-edit\n"
        await pilot.pause()
        await asyncio.sleep(ModelScreen.YAML_VALIDATION_DELAY / 2 + 0.05)
        await pilot.pause()

        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await asyncio.sleep(ModelScreen.YAML_VALIDATION_DELAY / 2 + 0.1)
        await pilot.pause()
        await wait_for(pilot, lambda: editor.has_class("yaml-valid"))
        assert editor.has_class("yaml-valid")
        assert app.screen.drafts["chat"].model == "second-edit"


async def test_invalid_yaml_blocks_model_accept(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))
    notifications = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = "model: [not valid"

        app.screen.action_apply()

        assert isinstance(app.screen, ModelScreen)
        assert editor.has_class("yaml-invalid")
        assert notifications[-1][1]["severity"] == "error"
        assert app.screen.query_one("#chat-yaml-error", Static).content


async def test_yaml_validation_error_with_rich_markup_is_plain_text(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = """\
model: gpt-5.4
model_provider: openai
base_url: http://foo
api_key:
  env:
"""

        await pilot.pause()
        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await wait_for_yaml_debounce(pilot)

        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one("#chat-advanced", Collapsible).collapsed = False
        await pilot.pause()
        await wait_for(pilot, lambda: editor.has_class("yaml-invalid"))
        assert editor.has_class("yaml-invalid")
        error = app.screen.query_one("#chat-yaml-error", Static)
        assert "validation errors for ChatModelConfig" in str(error.content)
        assert "api_key:" in str(error.content)
        assert "errors.pydantic.dev" not in str(error.content)
        assert error.styles.text_wrap == "wrap"
        assert error.region.height > 3
        assert error.virtual_size.width <= error.content_region.width


async def test_yaml_validation_lists_all_errors_in_bounded_scroll_area(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one("#chat-advanced", Collapsible).collapsed = False
        editor = app.screen.query_one("#chat-config-yaml", TextArea)
        editor.text = """\
model: gpt-5.4
model_provider: openai
inference_provider: openai
max_completion_tokens: nope
ssl_verify: also-nope
api_key:
  env:
"""

        await pilot.pause()
        assert not editor.has_class("yaml-valid", "yaml-invalid")
        await wait_for_yaml_debounce(pilot)

        error = app.screen.query_one("#chat-yaml-error", Static)
        message = str(error.content)
        assert "max_completion_tokens:" in message
        assert "ssl_verify:" in message
        assert "api_key:" in message
        assert message.startswith("4 validation errors for ChatModelConfig")
        assert len(message.splitlines()) == 5
        assert error.region.height == 6
        assert error.styles.overflow_y == "auto"


async def test_model_change_only_reports_changed_embedding(tmp_path):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)
    new_embedding = hitl.config.emb_model.model_copy(
        update={"model": "text-embedding-3-small"}
    )

    async with app.run_test(size=(80, 24)):
        app._select_model(
            ModelSelection(
                chat=hitl.config.llm_model.model_copy(),
                embedding=new_embedding,
            )
        )
        await app.workers.wait_for_complete()

        messages = [message.content for message in app.query(ToolMessage)]
        assert not any(
            "Changed the chat model" in message for message in messages
        )
        assert messages[-1].startswith(
            "Changed the embedding model to text-embedding-3-small"
        )


async def test_model_modal_preserves_direct_embedding_endpoint(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.emb_model = EmbModelConfig(
        model="old-embedding",
        model_provider="openai",
        base_url="https://embeddings.example/v1",
        check_embedding_ctx_length=False,
    )
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        assert (
            app.screen.query_one("#embedding-inference-provider", Select).value
            == ModelScreen.NONE_VALUE
        )
        app.screen.query_one(
            "#embedding-model-name", Select
        ).value = ModelScreen.CUSTOM_VALUE
        app.screen.query_one(
            "#embedding-model-name-custom", Input
        ).value = "new-embedding"

        embedding = app.screen._settings("embedding", app.screen.embedding)

        assert isinstance(embedding, EmbModelConfig)
        assert embedding.model == "new-embedding"
        assert embedding.inference_provider is None
        assert embedding.base_url == "https://embeddings.example/v1"
        assert embedding.check_embedding_ctx_length is False


async def test_switching_direct_endpoint_to_named_provider_stays_in_sync(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.llm_model = ChatModelConfig(
        model="direct-model",
        model_provider="openai",
        base_url="https://direct.example/v1",
    )
    discovered_with = []

    def provider_models(config):
        discovered_with.append(config)
        return [ProviderModel("provider-model", "openai")]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)

        provider = app.screen.query_one("#chat-inference-provider", Select)
        provider.value = "openai"
        await pilot.pause()
        await app.workers.wait_for_complete()

        draft = app.screen.drafts["chat"]
        yaml_values = yaml.safe_load(
            app.screen.query_one("#chat-config-yaml", TextArea).text
        )
        assert provider.value == "openai"
        assert draft.inference_provider == "openai"
        assert draft.base_url is None
        assert yaml_values["inference_provider"] == "openai"
        assert "base_url" not in yaml_values
        assert any(
            config.base_url
            == hitl.config.inference_providers["openai"].base_url
            for config in discovered_with
        )
        assert any(
            model.name == "provider-model"
            for model in app.screen.model_catalogs["chat"].values()
        )


async def test_named_provider_control_removes_direct_url_in_one_sync(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.llm_model = ChatModelConfig(
        model="direct-model",
        base_url="https://direct.example/v1",
    )
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        provider = app.screen.query_one("#chat-inference-provider", Select)
        initial_yaml = yaml.safe_load(
            app.screen.query_one("#chat-config-yaml", TextArea).text
        )
        assert initial_yaml["base_url"] == "https://direct.example/v1"
        app.screen._syncing_controls = True
        try:
            provider.value = "openai"
            await pilot.pause()
        finally:
            app.screen._syncing_controls = False

        app.screen._structured_controls_changed("chat")

        yaml_values = yaml.safe_load(
            app.screen.query_one("#chat-config-yaml", TextArea).text
        )
        assert app.screen.drafts["chat"].inference_provider == "openai"
        assert "base_url" not in yaml_values


async def test_immediate_apply_clears_direct_url_for_named_provider(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.llm_model = ChatModelConfig(
        model="direct-model",
        base_url="https://direct.example/v1",
        temperature=0.2,
    )
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", lambda _config: []
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one(
            "#chat-inference-provider", Select
        ).value = "openai"

        app.screen.action_apply()
        await pilot.pause()

        await wait_for(pilot, lambda: not isinstance(app.screen, ModelScreen))
        assert not isinstance(app.screen, ModelScreen)
        assert hitl.config.llm_model.inference_provider == "openai"
        assert (
            hitl.config.llm_model.base_url
            == hitl.config.inference_providers["openai"].base_url
        )
        assert hitl.config.llm_model.base_url != "https://direct.example/v1"
        assert hitl.config.llm_model.model_extra == {"temperature": 0.2}


async def test_model_modal_preserves_only_explicit_overrides_when_switching_provider(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.inference_providers["other"] = InferenceProviderConfig(
        base_url="https://other.example/v1",
        ssl_verify=False,
        timeout=20,
    )
    configured = ChatModelConfig(
        model="gpt-test",
        inference_provider="openai",
        ssl_verify=False,
        temperature=0.2,
    )
    openai = hitl.config.inference_providers["openai"]
    hitl.config.inference_providers["openai"] = InferenceProviderConfig(
        base_url=openai.base_url,
        api_key=openai.api_key,
        timeout=10,
    )
    hitl.config.llm_model = configured.resolve_inference_provider(
        hitl.config.inference_providers
    )
    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models",
        lambda _config: [ProviderModel("gpt-test", "openai")],
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one("#chat-inference-provider", Select).value = "other"

        chat = app.screen._settings("chat", app.screen.chat)

        assert isinstance(chat, ChatModelConfig)
        assert chat.inference_provider == "other"
        assert chat.ssl_verify is False
        assert chat.model_extra == {"temperature": 0.2}


async def test_model_modal_uses_default_provider_for_new_embedding(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.emb_model = None
    discovered_with = []

    def provider_models(config):
        discovered_with.append(config)
        return [ProviderModel("text-embedding-test", "openai", "embedding")]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)
        assert any(
            config.model_provider == "openai"
            and config.api_key
            == hitl.config.inference_providers["openai"].api_key
            for config in discovered_with
        )
        assert (
            app.screen.query_one("#embedding-inference-provider", Select).value
            == "openai"
        )
        app.screen.query_one(
            "#embedding-model-name", Select
        ).value = "text-embedding-test"
        app.screen.query_one(
            "#embedding-model-provider", Select
        ).value = ModelScreen.NONE_VALUE

        embedding = app.screen._settings("embedding", app.screen.embedding)

        assert isinstance(embedding, EmbModelConfig)
        assert embedding.model_provider == "openai"
        assert embedding.inference_provider == "openai"


async def test_stale_model_discovery_cannot_replace_new_provider_catalog(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.config.inference_providers["fast"] = InferenceProviderConfig(
        base_url="https://fast.example/v1"
    )
    slow_started = Event()
    release_slow = Event()
    replacement_requested = Event()
    publications_after_request = []

    def provider_models(config):
        if config.base_url == "https://api.openai.com/v1":
            slow_started.set()
            assert release_slow.wait(timeout=5)
            return [ProviderModel("gpt-5.4", "openai")]
        return [ProviderModel("fast-only", "openai")]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        assert await asyncio.to_thread(slow_started.wait, 5)
        assert isinstance(app.screen, ModelScreen)
        initial_workers = [
            worker
            for worker in app.workers
            if worker.group == "model-discovery"
        ]
        original_set_options = app.screen._set_model_options

        def record_options(prefix, catalog, current):
            if replacement_requested.is_set() and prefix == "chat":
                publications_after_request.append(set(catalog))
            original_set_options(prefix, catalog, current)

        monkeypatch.setattr(app.screen, "_set_model_options", record_options)
        replacement_requested.set()
        replacement_worker = app.screen._request_model_load(
            "chat", hitl.config.inference_providers["fast"]
        )
        release_slow.set()
        await replacement_worker.wait()
        await asyncio.gather(*(worker.wait() for worker in initial_workers))
        await pilot.pause()

        model_select = app.screen.query_one("#chat-model-name", Select)
        assert app.screen.model_catalogs["chat"] == {
            "fast-only": ProviderModel("fast-only", "openai")
        }
        assert model_select.value == "test-model"
        assert ("Not found: test-model", "test-model") in model_select._options
        assert ("gpt-5.4", "gpt-5.4") not in model_select._options
        assert all(
            "gpt-5.4" not in catalog for catalog in publications_after_request
        )
        assert app.screen.query_one("#chat-model-name-custom", Input).has_class(
            "hidden"
        )


async def test_model_modal_seeded_provider_fuzz_preserves_invariants(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    for index in range(3):
        hitl.config.inference_providers[f"provider-{index}"] = (
            InferenceProviderConfig(
                base_url=f"https://provider-{index}.example/v1"
            )
        )

    def provider_models(config):
        host = config.base_url or ""
        provider_index = next(
            (index for index in range(3) if f"provider-{index}" in host),
            9,
        )
        return [
            ProviderModel(f"model-{provider_index}-{model_index}", "openai")
            for model_index in range(3)
        ]

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.list_provider_models", provider_models
    )
    app = UrsaTextualApp(hitl)
    choices = [f"provider-{index}" for index in range(3)]
    rng = random.Random(20260826)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._show_command("models")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ModelScreen)

        for iteration in range(30):
            provider_select = app.screen.query_one(
                "#chat-inference-provider", Select
            )
            provider = rng.choice([
                choice for choice in choices if choice != provider_select.value
            ])
            provider_index = int(provider.rsplit("-", 1)[1])
            model_select = app.screen.query_one("#chat-model-name", Select)
            custom = app.screen.query_one("#chat-model-name-custom", Input)
            if rng.random() < 0.25:
                model_select.value = ModelScreen.CUSTOM_VALUE
                custom.value = f"custom-{iteration}"
                await pilot.pause()
                await wait_for(pilot, lambda: not custom.has_class("hidden"))
                assert not custom.has_class("hidden")
            elif rng.random() < 0.5:
                model_select.value = rng.choice([
                    value
                    for _label, value in model_select._options
                    if isinstance(value, str)
                    and value
                    not in {ModelScreen.NONE_VALUE, ModelScreen.CUSTOM_VALUE}
                ])
            provider_select.value = provider
            await pilot.pause()
            await app.workers.wait_for_complete()

            selected = str(model_select.value)
            options = dict(
                (value, label) for label, value in model_select._options
            )
            available = {
                f"model-{provider_index}-{model_index}"
                for model_index in range(3)
            }
            assert selected in options
            if selected not in available:
                assert options[selected] == f"Not found: {selected}"
            assert custom.has_class("hidden")
            assert app.screen.drafts["chat"].model == selected
            assert app.screen.drafts["chat"].inference_provider == provider
            yaml_values = yaml.safe_load(
                app.screen.query_one("#chat-config-yaml", TextArea).text
            )
            assert yaml_values["inference_provider"] == provider


def test_model_modal_new_embedding_inherits_direct_chat_settings(tmp_path):
    hitl = FakeHITL(tmp_path)
    chat = ChatModelConfig(
        model="chat-model",
        model_provider="openai",
        base_url="https://gateway.example/v1",
        api_key={"env": "GATEWAY_API_KEY"},
        ssl_verify=False,
    )

    screen = ModelScreen(hitl.config.inference_providers, chat, None)

    assert screen.embedding.model_provider == "openai"
    assert screen.embedding.inference_provider is None
    assert screen.embedding.base_url == "https://gateway.example/v1"
    assert screen.embedding.api_key == chat.api_key
    assert screen.embedding.ssl_verify is False


async def test_command_picker_prioritizes_command_name_over_description(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", "k", "e")
        await pilot.pause()

        await wait_for(pilot, lambda: isinstance(app.screen, HotlistScreen))
        assert isinstance(app.screen, HotlistScreen)
        assert app.screen.matches[0].startswith("keymap —")
        assert any(match.startswith("status —") for match in app.screen.matches)
        assert app.screen.query_one("#hotlist-options").highlighted == 0


async def test_theme_command_selects_theme_and_escape_preserves_it(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        status = app.query_one("#status", Static)
        dark_background = status.styles.background
        assert app.theme == "ursa-dark"
        assert app.get_theme("ursa-dark").dark
        assert not app.get_theme("ursa-light").dark

        await pilot.press("/")
        await pilot.pause()
        await pilot.press("t", "h", "e", "m", "e", "enter")
        await pilot.pause()

        await wait_for(pilot, lambda: isinstance(app.screen, ThemeScreen))
        assert isinstance(app.screen, ThemeScreen)
        assert app.screen.styles.background.a == 0
        assert app.screen.picker_title == "Themes"
        assert app.screen.candidates[:2] == ["ursa-dark", "ursa-light"]
        assert set(BUILTIN_THEMES) <= set(app.screen.candidates)
        await pilot.press("down")
        await pilot.pause()

        await wait_for(pilot, lambda: app.theme == "ursa-light")
        assert app.theme == "ursa-light"
        assert status.styles.background != dark_background
        await pilot.press("up")
        await pilot.pause()
        await wait_for(pilot, lambda: app.theme == "ursa-dark")
        assert app.theme == "ursa-dark"
        assert status.styles.background == dark_background

        await pilot.press("down", "enter")
        await pilot.pause()

        await wait_for(pilot, lambda: app.theme == "ursa-light")
        assert app.theme == "ursa-light"
        assert status.styles.background != dark_background
        assert "| Theme | `ursa-light` |" in app._status_markdown()

        await app._show_command("theme")
        await pilot.pause()
        await wait_for(
            pilot,
            lambda: app.screen.candidates[:2] == ["ursa-light", "ursa-dark"],
        )
        assert app.screen.candidates[:2] == ["ursa-light", "ursa-dark"]
        await pilot.press("down")
        await pilot.pause()
        await wait_for(pilot, lambda: app.theme == "ursa-dark")
        assert app.theme == "ursa-dark"
        await pilot.press("escape")
        await pilot.pause()

        await wait_for(pilot, lambda: app.theme == "ursa-light")
        assert app.theme == "ursa-light"
        assert app.query_one(PromptArea).has_focus

        await app._show_command("theme")
        await pilot.pause()
        app.screen.query_one(Input).value = "nord"
        await pilot.pause()
        await wait_for(pilot, lambda: app.theme == "nord")
        assert app.theme == "nord"
        await pilot.press("escape")
        await pilot.pause()
        await wait_for(pilot, lambda: app.theme == "ursa-light")
        assert app.theme == "ursa-light"


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

    async def get_agent(name):
        return hitl.agents[name]

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("/", "a", "g", "e", "n", "t", "s", "enter")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        await wait_for(pilot, lambda: isinstance(app.screen, AgentsScreen))
        assert isinstance(app.screen, AgentsScreen)
        panes = list(app.screen.query(TabPane))
        assert len(panes) == 2
        assert [tab.label_text for tab in app.screen.query(Tab)] == [
            "#plan",
            "#chat",
        ]
        tools = list(app.screen.query(Collapsible))
        assert len(tools) == 2
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
        tools[0].collapsed = True
        await pilot.pause()
        await wait_for(pilot, lambda: tools[0].collapsed)
        assert tools[0].collapsed
        tools[0].collapsed = False
        await pilot.pause()
        await wait_for(pilot, lambda: len(tools[0].query(Markdown)) == 1)
        assert len(tools[0].query(Markdown)) == 1

        await pilot.press("right")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await wait_for(pilot, lambda: len(app.screen.query(Collapsible)) == 4)
        assert len(app.screen.query(Collapsible)) == 4


async def test_agents_lazily_load_tools_and_only_once(tmp_path):
    hitl = FakeHITL(tmp_path)
    ready = asyncio.Event()
    calls = []
    wrappers = {
        name: SimpleNamespace(
            description=f"{name} agent",
            config={},
            tool_sources={},
            _agent=None,
        )
        for name in ("plan", "chat")
    }
    hitl.agents = wrappers

    async def get_agent(name):
        calls.append(name)
        if name == "plan":
            await ready.wait()
        wrapper = wrappers[name]
        wrapper._agent = SimpleNamespace(
            tools={"read_file": FakeConfiguredTool()}
        )
        return wrapper

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await app._show_command("agents")
        await pilot.pause()

        await wait_for(pilot, lambda: isinstance(app.screen, AgentsScreen))
        assert isinstance(app.screen, AgentsScreen)
        assert calls == ["plan"]
        # A callback already queued when hydration stops must tolerate the
        # frame state being gone while the loading node is still mounted.
        app.screen._stop_tool_loading(0)
        app.screen._advance_tool_loading(0)
        # Tool discovery is suspended, but tabs remain interactive.
        await pilot.press("right")
        await pilot.pause()
        assert calls == ["plan", "chat"]
        assert len(app.screen.query("#agent-tools-1 .agent-tool")) == 1

        await pilot.press("left", "right")
        await pilot.pause()
        assert calls == ["plan", "chat"]

        ready.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await wait_for(pilot, lambda: app.screen.agents[0].tools_loaded)
        assert app.screen.agents[0].tools_loaded
        # Hydration of the hidden plan tab updates state without mounting its
        # potentially expensive Markdown tool cards on the UI thread.
        assert not app.screen.query("#agent-tools-0 .agent-tool")
        assert app.screen._tool_panes_pending_render == {0}

        await pilot.press("left")
        await pilot.pause()
        await wait_for(
            pilot,
            lambda: len(app.screen.query("#agent-tools-0 .agent-tool")) == 1,
        )
        assert len(app.screen.query("#agent-tools-0 .agent-tool")) == 1
        assert app.screen._tool_panes_pending_render == set()

        await pilot.press("right", "left")
        await pilot.pause()
        assert calls == ["plan", "chat"]


async def test_immediate_switch_preempts_large_tool_render(
    tmp_path, monkeypatch
):
    hydration_started = asyncio.Event()
    hydration_release = asyncio.Event()
    construction_started = asyncio.Event()
    loading_seen_during_construction = False
    construction_count = 0
    safe = SimpleNamespace(
        description="safe agent",
        config={},
        tool_sources={},
        _agent=SimpleNamespace(tools={}),
    )
    execute = SimpleNamespace(
        description="execution agent",
        config={},
        tool_sources={},
        _agent=None,
    )
    hitl = FakeHITL(tmp_path)
    hitl.agents = {"safe": safe, "execute": execute}

    async def get_agent(name):
        wrapper = hitl.agents[name]
        if name == "execute":
            hydration_started.set()
            await hydration_release.wait()
            wrapper._agent = SimpleNamespace(
                tools={
                    f"tool_{index}": SimpleNamespace(
                        name=f"tool_{index}",
                        description="A detailed configured tool. " * 20,
                        args_schema=FakeToolArgs,
                        return_direct=False,
                    )
                    for index in range(100)
                }
            )
        return wrapper

    from ursa.cli.tui.widgets import AgentToolDetails

    def deliberately_slow_card(tool):
        nonlocal construction_count, loading_seen_during_construction
        construction_count += 1
        construction_started.set()
        loading_seen_during_construction = bool(
            app.screen.query("#agent-tools-1 .agent-tools-loading")
        )
        time.sleep(0.01)
        return AgentToolDetails(tool)

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.AgentToolDetails", deliberately_slow_card
    )
    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await app._show_command("agents")
        await app.workers.wait_for_complete()
        await pilot.press("right")
        await hydration_started.wait()

        ticks = 0

        def heartbeat():
            nonlocal ticks
            ticks += 1

        timer = app.screen.set_interval(0.01, heartbeat)
        ticks_before_release = ticks
        hydration_release.set()
        await construction_started.wait()
        assert loading_seen_during_construction
        # This is the reported ordering: completion becomes runnable just as
        # the user asks to leave the expensive tab.
        switch_task = asyncio.create_task(pilot.press("left"))
        tabs = app.screen.query_one("#agents-tabs", TabbedContent)
        async with asyncio.timeout(0.2):
            while tabs.active != "agent-tab-0":
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        timer.stop()

        assert tabs.active == "agent-tab-0"
        assert ticks > ticks_before_release

        await app.workers.wait_for_complete()
        await switch_task
        constructions_after_switch = construction_count
        assert constructions_after_switch <= 5
        await asyncio.sleep(0.05)
        assert construction_count == constructions_after_switch
        assert len(app.screen.query("#agent-tools-1 .agent-tool")) < 100
        assert not app.screen.query("#agent-tools-1 .agent-tool Markdown")

        await pilot.press("right")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await wait_for(
            pilot,
            lambda: len(app.screen.query("#agent-tools-1 .agent-tool")) == 100,
        )
        assert len(app.screen.query("#agent-tools-1 .agent-tool")) == 100
        assert not app.screen.query("#agent-tools-1 .agent-tool Markdown")
        assert not app.screen.query("#agent-tools-1 .agent-tools-loading")


async def test_agent_tool_render_failure_is_displayed(tmp_path, monkeypatch):
    construction_attempts = 0
    safe = SimpleNamespace(
        description="safe agent",
        config={},
        tool_sources={},
        _agent=SimpleNamespace(tools={}),
    )
    broken_tool = FakeConfiguredTool()
    broken = SimpleNamespace(
        description="broken agent",
        config={},
        tool_sources={},
        _agent=SimpleNamespace(tools={broken_tool.name: broken_tool}),
    )
    hitl = FakeHITL(tmp_path)
    hitl.agents = {"safe": safe, "broken": broken}

    async def get_agent(name):
        return hitl.agents[name]

    def fail_to_build_card(_tool):
        nonlocal construction_attempts
        construction_attempts += 1
        raise RuntimeError("tool card could not be built")

    monkeypatch.setattr(
        "ursa.cli.tui.widgets.AgentToolDetails", fail_to_build_card
    )
    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await app._show_command("agents")
        await app.workers.wait_for_complete()
        await pilot.press("right")
        await pilot.pause()
        await app.workers.wait_for_complete()

        error = app.screen.query_one(
            "#agent-tools-1 .agent-tools-error", Static
        )
        assert "tool card could not be built" in str(error.render())
        assert not app.screen.query("#agent-tools-1 .agent-tools-loading")
        assert construction_attempts == 1
        assert 1 not in app.screen._tool_panes_pending_render

        await pilot.press("left")
        await pilot.pause()
        await wait_for(
            pilot,
            lambda: app.screen.query_one("#agents-tabs", TabbedContent).active
            == "agent-tab-0",
        )
        assert app.screen.query_one("#agents-tabs", TabbedContent).active == (
            "agent-tab-0"
        )

        await pilot.press("right")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert construction_attempts == 1
        assert "tool card could not be built" in str(error.render())


async def test_initialized_tools_render_while_schema_hydration_is_pending(
    tmp_path,
):
    schema_started = threading.Event()
    schema_release = threading.Event()

    class BlockingSchema:
        @classmethod
        def model_json_schema(cls):
            schema_started.set()
            schema_release.wait(timeout=5)
            return {"properties": {}}

    configured_tool = FakeConfiguredTool()
    configured_tool.args_schema = BlockingSchema
    wrapper = SimpleNamespace(
        description="ready agent",
        config={},
        tool_sources={},
        _agent=SimpleNamespace(tools={"read_file": configured_tool}),
    )
    hitl = FakeHITL(tmp_path)
    hitl.agents = {"ready": wrapper}

    async def get_agent(_name):
        return wrapper

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    try:
        async with app.run_test(size=(100, 36)) as pilot:
            await app._show_command("agents")
            assert await asyncio.to_thread(schema_started.wait, 2)
            assert app.screen.query(".agent-tools-loading")
            tools = app.screen.query("#agent-tools-0 .agent-tool")
            assert len(tools) == 1
            assert "read_file" in str(tools.first(Collapsible).title)

            schema_release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await wait_for(
                pilot,
                lambda: len(app.screen.query("#agent-tools-0 .agent-tool"))
                == 1,
            )
            assert len(app.screen.query("#agent-tools-0 .agent-tool")) == 1
    finally:
        schema_release.set()


async def test_agents_display_tool_load_failure(tmp_path):
    hitl = FakeHITL(tmp_path)
    wrapper = SimpleNamespace(
        description="broken agent",
        config={},
        tool_sources={},
        _agent=None,
    )
    hitl.agents = {"broken": wrapper}

    async def get_agent(name):
        raise RuntimeError("MCP server unavailable")

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await app._show_command("agents")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        error = app.screen.query_one(".agent-tools-error", Static)
        assert "MCP server unavailable" in str(error.render())
        assert not app.screen.query(".agent-tools-loading")
        assert app.screen._tool_loading_timers == {}
        assert app.screen._tool_loading_frames == {}
        assert isinstance(app.screen, AgentsScreen)


async def test_agents_remain_responsive_during_blocking_initialization(
    tmp_path, chat_model
):
    constructor_started = threading.Event()
    constructor_release = threading.Event()
    mcp_started = threading.Event()
    mcp_release = threading.Event()
    schema_started = threading.Event()
    schema_release = threading.Event()

    class BlockingSchema:
        @classmethod
        def model_json_schema(cls):
            schema_started.set()
            schema_release.wait(timeout=5)
            return {"properties": {}}

    configured_tool = FakeConfiguredTool()
    configured_tool.args_schema = BlockingSchema

    class PhasedAgent(AgentWithTools):
        """A deliberately long description used to make this pane scroll.

        The remaining text creates enough vertical content to exercise scroll
        input while constructor, MCP, and schema phases are independently held.
        """

        def __init__(self, **_kwargs):
            constructor_started.set()
            constructor_release.wait(timeout=5)
            self._test_tools = {}

        @property
        def tools(self):
            return self._test_tools

        async def add_mcp_tools(self, _client):
            mcp_started.set()
            await asyncio.to_thread(mcp_release.wait, 5)
            self._test_tools = {configured_tool.name: configured_tool}
            return {configured_tool.name: "laboratory"}

    hitl = FakeHITL(tmp_path)
    execute = AgentHITL(agent_class=PhasedAgent)
    execute.config.update({f"option_{index}": index for index in range(20)})
    initialized = AgentHITL(agent_class=ExecutionAgent)
    initialized._agent = SimpleNamespace(tools={})
    hitl.agents = {"execute": execute, "ready": initialized}

    async def get_agent(name):
        wrapper = hitl.agents[name]
        if wrapper._agent is None:
            await wrapper.instantiate(
                llm=chat_model,
                workspace=tmp_path,
                agent_name="persistent",
                group="default",
                mcp_client=object(),
                thread_id="test",
            )
        return wrapper

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async def assert_ui_is_live(pilot, screen):
        loading = screen.query_one(".agent-tools-loading", Static)
        first_frame = str(loading.render())
        assert await wait_for(
            pilot, lambda: str(loading.render()) != first_frame
        )
        await pilot.press("right")
        assert screen.query_one("#agents-tabs", TabbedContent).active == (
            "agent-tab-1"
        )
        await pilot.press("left")
        scroll = screen._scroll_view()
        await pilot.press("end")
        await pilot.pause()
        bottom = scroll.scroll_y
        assert bottom > 0
        await pilot.press("home")
        await pilot.pause()
        await wait_for(pilot, lambda: scroll.scroll_y < bottom)
        assert scroll.scroll_y < bottom

    try:
        async with app.run_test(size=(100, 36)) as pilot:
            await app._show_command("agents")
            assert await asyncio.to_thread(constructor_started.wait, 2)
            screen = app.screen
            await assert_ui_is_live(pilot, screen)

            constructor_release.set()
            assert await asyncio.to_thread(mcp_started.wait, 2)
            await assert_ui_is_live(pilot, screen)

            mcp_release.set()
            assert await asyncio.to_thread(schema_started.wait, 2)
            await assert_ui_is_live(pilot, screen)

            schema_release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await wait_for(
                pilot, lambda: screen.query("#agent-tools-0 .agent-tool")
            )
            assert screen.query("#agent-tools-0 .agent-tool")
            assert screen._tool_loading_timers == {}
            assert screen._tool_loading_frames == {}
    finally:
        constructor_release.set()
        mcp_release.set()
        schema_release.set()


async def test_dismissing_agents_during_loading_cleans_up_and_publishes(
    tmp_path,
):
    schema_started = threading.Event()
    schema_release = threading.Event()

    class BlockingSchema:
        @classmethod
        def model_json_schema(cls):
            schema_started.set()
            schema_release.wait(timeout=5)
            return {"properties": {}}

    configured_tool = FakeConfiguredTool()
    configured_tool.args_schema = BlockingSchema

    class SlowAgent:
        description = "Slow agent"

        def __init__(self, **_kwargs):
            self.tools = {"read_file": configured_tool}

    hitl = FakeHITL(tmp_path)
    wrapper = AgentHITL(agent_class=SlowAgent)
    hitl.agents = {"slow": wrapper}

    async def get_agent(_name):
        await wrapper.instantiate()
        return wrapper

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    try:
        async with app.run_test(size=(100, 36)) as pilot:
            await app._show_command("agents")
            assert await asyncio.to_thread(schema_started.wait, 2)
            loading_screen = app.screen
            await pilot.press("escape")
            await pilot.pause()
            await wait_for(
                pilot, lambda: not isinstance(app.screen, AgentsScreen)
            )
            assert not isinstance(app.screen, AgentsScreen)
            assert loading_screen._tool_loading_timers == {}
            assert loading_screen._tool_loading_frames == {}

            schema_release.set()
            await wrapper.wait_until_initialized()
            await app._show_command("agents")
            await pilot.pause()
            await wait_for(
                pilot, lambda: not app.screen.query(".agent-tools-loading")
            )
            assert not app.screen.query(".agent-tools-loading")
            assert app.screen.query("#agent-tools-0 .agent-tool")
    finally:
        schema_release.set()


async def test_agents_lazily_render_execution_agent_tools(tmp_path, chat_model):
    hitl = FakeHITL(tmp_path)
    wrapper = AgentHITL(agent_class=ExecutionAgent)
    hitl.agents = {"execute": wrapper}

    async def get_agent(name):
        if wrapper._agent is None:
            await wrapper.instantiate(
                llm=chat_model,
                workspace=tmp_path,
                agent_name=None,
                group="default",
                mcp_client=None,
                thread_id="test",
            )
        return wrapper

    hitl.get_agent = get_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await app._show_command("agents")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        rendered_tools = app.screen.query("#agent-tools-0 .agent-tool")
        assert len(rendered_tools) == len(wrapper._agent.tools)
        assert len(rendered_tools) > 0
        first_tool = rendered_tools.first()
        tools_container = app.screen.query_one("#agent-tools-0", Vertical)
        details = app.screen.query_one(".agent-details", VerticalScroll)
        children = list(details.children)
        tools_title = app.screen.query_one(".agent-tools-title", Static)
        title_index = children.index(tools_title)
        assert all(
            children.index(markdown) < title_index
            for markdown in details.query(Markdown)
            if markdown.parent is details
        )
        assert children.index(tools_title) < children.index(tools_container)
        app.screen.refresh(layout=True)
        await pilot.pause()
        await wait_for(pilot, lambda: first_tool.region.height > 0)
        assert first_tool.region.height > 0
        assert tools_container.region.contains_region(first_tool.region)
        assert first_tool.region.y < app.screen.region.bottom


def test_command_details_and_keymap_come_from_live_bindings(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(crossplatform, "expects_kitty_keyboard", lambda: False)
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
    assert "| Terminal backend |" in status
    assert "Ghostty" in status or "Process" in status
    assert "Kitty keyboard support" in keymap
    assert "not identified" in keymap
    assert "shift+⏎ / ^j" in keymap
    assert "^c" in keymap
    assert "Clear prompt" in keymap
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


def test_keymap_omits_compatibility_warning_when_kitty_is_expected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(crossplatform, "expects_kitty_keyboard", lambda: True)
    app = UrsaTextualApp(FakeHITL(tmp_path))

    keymap = app._keymap_markdown()

    assert "Kitty keyboard support expected" in keymap
    assert "may not work" not in keymap


def test_hotlist_mount_survives_absent_children():
    # The app can begin tearing down while this screen is still mounting;
    # the Mount dispatch then runs with children absent, and a bare
    # query_one crashed the whole app from inside the event handler,
    # surfacing at run_test exit and masking the test's real failure.
    screen = HotlistScreen("pick", ["one"])

    screen.on_mount()


async def test_hotlist_mount_race_with_teardown_does_not_crash(tmp_path):
    # Real-machinery pin for the mount race: exiting right after the push
    # makes mount_all skip composing children while Mount still
    # dispatches, and the unguarded on_mount crashed the app (NoMatches
    # raised at run_test exit).
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)):
        app.push_screen(HotlistScreen("pick", ["one"]))
        app.exit()
