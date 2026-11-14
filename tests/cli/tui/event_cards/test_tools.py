import pytest
from langchain_core.messages import ToolMessage
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from tests.cli._app_fakes import FakeHITL
from tests.tui.utils import eventually
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import TermCard, ToolCallCard
from ursa.cli.tui.event_cards.term import terminal_id
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn


async def test_default_tool_card_switches_from_input_to_output(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("call it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "lookup_widget"},
            "",
            run_id="tool-one",
            inputs={"query": "bear", "limit": 5},
        )
        await pilot.pause()
        card = turn.query_one(ToolCallCard)
        assert (
            str(card.query_one(".tool-call-title", Static).content)
            == "🛠️ lookup_widget"
        )
        assert card.tool_input == {"query": "bear", "limit": 5}
        assert not card.completed
        card.set_expanded(True)
        assert (
            card.query_one(".tool-input-json", Static).content.code
            == '{\n  "limit": 5,\n  "query": "bear"\n}'
        )
        await handler.on_tool_end(
            {"matches": ["polar", "grizzly"]}, run_id="tool-one"
        )
        await eventually(pilot, lambda: card.completed)
        assert (
            card.query_one(".tool-output-json", Static).content.code.find(
                '"grizzly"'
            )
            >= 0
        )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("Terminal ID: Ab12Cd34", "Ab12Cd34"),
        ("Terminal ID: Ab12Cd34\n", "Ab12Cd34"),
        ({"term_id": "Ab12Cd34"}, "Ab12Cd34"),
        ("result Terminal ID: Ab12Cd34", None),
        ({"line": "Terminal ID: Ab12Cd34"}, None),
    ],
)
def test_terminal_id_only_accepts_documented_session_results(result, expected):
    assert terminal_id(result) == expected


@pytest.mark.parametrize(
    "tool",
    [
        "term_send_text",
        "term_paste_text",
        "term_wait_screen",
        "term_screenshot",
    ],
)
async def test_terminal_session_tools_share_one_live_card(tmp_path, tool):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("use a terminal", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "term"},
            "",
            run_id="launch",
            inputs={"cmd": "python", "session": True},
        )
        await handler.on_tool_end("Terminal ID: Ab12Cd34", run_id="launch")
        await handler.on_tool_start(
            {"name": tool},
            "",
            run_id="operation",
            inputs={"term_id": "Ab12Cd34"},
        )
        await pilot.pause()
        cards = list(turn.query(TermCard))
        assert len(cards) == 1
        assert cards[0].term_id == "Ab12Cd34"
        assert cards[0].call_count == 1


async def test_short_terminal_result_stays_a_generic_tool_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("quick terminal", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "term"}, "", run_id="quick", inputs={"cmd": "echo"}
        )
        await handler.on_tool_end(
            "Terminal contents:\none\ntwo\n", run_id="quick"
        )
        await pilot.pause()
        assert not list(turn.query(TermCard))
        assert "two" in str(turn.query_one(ToolCallCard).output)


async def test_default_tool_card_shows_failure_output(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("call it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "lookup_widget"}, "", run_id="bad", inputs={}
        )
        await handler.on_tool_error(ValueError("bad filter"), run_id="bad")
        await pilot.pause()
        card = turn.query_one(ToolCallCard)
        assert card.failed and card.output == "bad filter"


async def test_tool_call_preview_renders_code_brackets_as_plain_text(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    output = "[patch={'old_code': 'ax.set_xticklabels([labels])'}]"
    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("edit it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "edit_plot"}, "", run_id="code", inputs={}
        )
        await handler.on_tool_end(output, run_id="code")
        await pilot.pause()
        assert turn.query_one(ToolCallCard).output == output


async def test_tool_message_prefers_structured_content_as_json(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("call it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "lookup_widget"}, "", run_id="structured", inputs={}
        )
        await handler.on_tool_end(
            ToolMessage(
                content="fallback",
                artifact={"structured_content": {"matches": ["grizzly"]}},
                tool_call_id="structured",
            ),
            run_id="structured",
        )
        await pilot.pause()
        card = turn.query_one(ToolCallCard)
        card.set_expanded(True)
        assert (
            '"grizzly"'
            in card.query_one(".tool-output-json", Static).content.code
        )


async def test_text_only_tool_message_renders_as_markdown(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("call it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "lookup_widget"}, "", run_id="text", inputs={}
        )
        await handler.on_tool_end(
            ToolMessage(
                content=[
                    {"type": "text", "text": "## Result\n\n**Found it.**"}
                ],
                tool_call_id="text",
            ),
            run_id="text",
        )
        await pilot.pause()
        card = turn.query_one(ToolCallCard)
        card.set_expanded(True)
        assert (
            str(card.query_one(".tool-output-markdown", Markdown).source)
            == "## Result\n\n**Found it.**"
        )
