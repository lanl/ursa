from langchain_core.messages import ToolMessage
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from tests.cli._app_fakes import FakeHITL
from ursa.cli.app import UrsaTextualApp
from ursa.cli.event_cards import ToolCallCard
from ursa.cli.event_handler import TextualEventHandler
from ursa.cli.turn import Turn


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
        assert card.query_one(".tool-call-title", Static).content == (
            "🛠️ lookup_widget"
        )
        assert card.tool_input == {"query": "bear", "limit": 5}
        assert not card.completed
        assert "bear" in str(
            card.query_one(".tool-call-preview", Static).content
        )
        assert card.query_one(".tool-call-state", Static).content in (
            *card.app.query_one(".activity").FRAMES,
        )
        assert card.query_one(".tool-call-details").has_class("hidden")

        card.set_expanded(True)
        assert not card.query_one(".tool-call-details").has_class("hidden")
        assert card.query_one(".tool-output-pane").has_class("hidden")
        input_syntax = card.query_one(".tool-input-json", Static).content
        assert type(input_syntax).__name__ == "Syntax"
        assert input_syntax.code == '{\n  "limit": 5,\n  "query": "bear"\n}'

        await handler.on_tool_end(
            {"matches": ["polar", "grizzly"]}, run_id="tool-one"
        )
        await pilot.pause()

        assert card.completed
        assert card.query_one(".tool-call-state", Static).content == "✓"
        assert "polar" in str(
            card.query_one(".tool-call-preview", Static).content
        )
        assert not card.query_one(".tool-output-pane").has_class("hidden")
        output_syntax = card.query_one(".tool-output-json", Static).content
        assert type(output_syntax).__name__ == "Syntax"
        assert '"grizzly"' in output_syntax.code


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
        assert card.failed
        assert card.output == "bad filter"
        assert card.query_one(".tool-call-state", Static).content == "✗"


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
                content="Human-readable fallback",
                artifact={
                    "structured_content": {"matches": ["polar", "grizzly"]}
                },
                tool_call_id="structured",
            ),
            run_id="structured",
        )
        await pilot.pause()

        card = turn.query_one(ToolCallCard)
        card.set_expanded(True)
        output = card.query_one(".tool-output-json", Static)
        assert not output.has_class("hidden")
        assert '"grizzly"' in output.content.code
        assert card.query_one(".tool-output-markdown").has_class("hidden")


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
                    {
                        "type": "text",
                        "text": "## Result\n\n**Found it.**",
                    }
                ],
                tool_call_id="text",
            ),
            run_id="text",
        )
        await pilot.pause()

        card = turn.query_one(ToolCallCard)
        card.set_expanded(True)
        markdown = card.query_one(".tool-output-markdown", Markdown)
        assert not markdown.has_class("hidden")
        assert str(markdown.source) == "## Result\n\n**Found it.**"
        assert card.query_one(".tool-output-json").has_class("hidden")
