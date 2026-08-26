import asyncio
import os
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client.session_group import StreamableHttpParameters
from textual import events
from textual.binding import Binding
from textual.theme import BUILTIN_THEMES
from textual.widgets import (
    Collapsible,
    Input,
    Markdown,
    Select,
    Static,
    Tab,
    TabPane,
)

import ursa.util.crossplatform as crossplatform
from tests.cli._app_fakes import FakeHITL
from ursa.cli.app import UrsaTextualApp
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
)
from ursa.cli.tips import TIPS, random_tip, runtime_keymap
from ursa.cli.widgets import (
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
    hitl.config.mcp_servers = {
        "local": StdioServerParameters(command="ursa-mcp", args=[]),
        "remote": StreamableHttpParameters(url="https://example.test/mcp"),
    }
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
        assert "agents" in screenshot
        assert [
            candidate.partition(" — ")[0] for candidate in app.screen.candidates
        ] == ["agents", "exit", "status", "keymap", "models", "theme"]

        await pilot.press("s", "t", "a", "t", "u", "s", "enter")
        await pilot.pause()
        assert isinstance(app.screen, InformationScreen)
        assert "LLM Endpoint" in app.screen.content
        assert "lab-assistant" in app.screen.content
        assert "MCP servers" in app.screen.content
        assert "ursa-mcp" in app.screen.content
        assert "https://example.test/mcp" in app.screen.content


async def test_exit_command_quits_the_app(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", "e", "x", "i", "t", "enter")
        await pilot.pause()

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
        "ursa.cli.widgets.list_provider_models",
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
            ("Other…", ModelScreen.CUSTOM_VALUE),
        ]
        app.screen.query_one("#chat-model-name", Select).value = "claude-stale"
        await pilot.pause()
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
        "ursa.cli.widgets.list_provider_models", lambda _config: []
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
        "ursa.cli.widgets.list_provider_models",
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
        "ursa.cli.widgets.list_provider_models", provider_models
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

        assert isinstance(app.screen, ThemeScreen)
        assert app.screen.styles.background.a == 0
        assert app.screen.picker_title == "Themes"
        assert app.screen.candidates[:2] == ["ursa-dark", "ursa-light"]
        assert set(BUILTIN_THEMES) <= set(app.screen.candidates)
        await pilot.press("down")
        await pilot.pause()

        assert app.theme == "ursa-light"
        assert status.styles.background != dark_background
        await pilot.press("up")
        await pilot.pause()
        assert app.theme == "ursa-dark"
        assert status.styles.background == dark_background

        await pilot.press("down", "enter")
        await pilot.pause()

        assert app.theme == "ursa-light"
        assert status.styles.background != dark_background
        assert "| Theme | `ursa-light` |" in app._status_markdown()

        await app._show_command("theme")
        await pilot.pause()
        assert app.screen.candidates[:2] == ["ursa-light", "ursa-dark"]
        await pilot.press("down")
        await pilot.pause()
        assert app.theme == "ursa-dark"
        await pilot.press("escape")
        await pilot.pause()

        assert app.theme == "ursa-light"
        assert app.query_one(PromptArea).has_focus

        await app._show_command("theme")
        await pilot.pause()
        app.screen.query_one(Input).value = "nord"
        await pilot.pause()
        assert app.theme == "nord"
        await pilot.press("escape")
        await pilot.pause()
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
