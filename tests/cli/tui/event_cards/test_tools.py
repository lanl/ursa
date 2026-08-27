import pytest
from langchain_core.messages import ToolMessage
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Markdown, Static

from tests.cli._app_fakes import FakeHITL, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import TermCard, ToolCallCard
from ursa.cli.tui.event_cards.term import TERM_TOOLS, terminal_id
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.terminal_view import TerminalView
from ursa.cli.tui.turn import Turn
from ursa.tools.terminal.base import TerminalRenderSnapshot, TerminalSpan


class _TermCardManager:
    def __init__(self) -> None:
        self.contents_value = ""
        self.missing = False
        self.snapshot_value = "live one"
        self.screen = False

    async def contents(self, term_id: str) -> str:
        assert term_id == "Ab12Cd34"
        if self.missing:
            raise KeyError(term_id)
        return self.contents_value

    async def render_snapshot(self, term_id: str) -> TerminalRenderSnapshot:
        return TerminalRenderSnapshot(
            term_id,
            (TerminalSpan(self.snapshot_value),),
            rows=24 if self.screen else None,
            cols=80 if self.screen else None,
            screen=self.screen,
        )


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

        await wait_for(pilot, lambda: card.completed)
        assert card.completed
        assert card.query_one(".tool-call-state", Static).content == "✓"
        assert "polar" in str(
            card.query_one(".tool-call-preview", Static).content
        )
        assert not card.query_one(".tool-output-pane").has_class("hidden")
        output_syntax = card.query_one(".tool-output-json", Static).content
        assert type(output_syntax).__name__ == "Syntax"
        assert '"grizzly"' in output_syntax.code


async def test_terminal_calls_for_same_session_share_live_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
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
            {"name": "term_send_line"},
            "",
            run_id="send",
            inputs={"term_id": "Ab12Cd34", "line": "print(42)"},
        )
        await handler.on_tool_end(
            "Sent line to terminal Ab12Cd34", run_id="send"
        )
        await pilot.pause()

        cards = list(turn.query(TermCard))
        assert len(cards) == 1
        card = cards[0]
        assert card.term_id == "Ab12Cd34"
        assert card.call_count == 2
        assert "2 calls" in str(
            card.query_one(".term-card-title", Static).content
        )

        card.set_expanded(True)
        assert card.query_one(".term-card-tail").has_class("hidden")
        assert not card.query_one(".term-card-live").has_class("hidden")
        assert card.query_one(TerminalView).term_id == "Ab12Cd34"


async def test_short_terminal_call_keeps_standard_tool_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("quick terminal", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "term"},
            "",
            run_id="quick",
            inputs={"cmd": "printf 'one\\ntwo\\n'"},
        )
        await handler.on_tool_end(
            "Terminal contents:\none\ntwo\n", run_id="quick"
        )
        await pilot.pause()

        assert not list(turn.query(TermCard))
        card = turn.query_one(ToolCallCard)
        assert card.completed
        assert "two" in str(card.output)


async def test_distinct_terminal_ids_get_distinct_cards(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("two terminals", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        for run_id, term_id in (("one", "Ab12Cd34"), ("two", "Ef56Gh78")):
            await handler.on_tool_start(
                {"name": "term_send_text"},
                "",
                run_id=run_id,
                inputs={"term_id": term_id, "text": "hello"},
            )
        await pilot.pause()

        cards = list(turn.query(TermCard))
        assert {card.term_id for card in cards} == {"Ab12Cd34", "Ef56Gh78"}

        turn.set_card_details_expanded(True)
        assert all(card.expanded for card in cards)
        assert all(
            not card.query_one(".term-card-live").has_class("hidden")
            for card in cards
        )


async def test_delayed_launch_result_merges_provisional_card_and_aliases(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("launch and use terminal", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "term"},
            "",
            run_id="launch",
            inputs={"cmd": "python", "session": True},
        )
        provisional = turn.query_one(TermCard)

        await handler.on_tool_start(
            {"name": "term_send_line"},
            "",
            run_id="send",
            inputs={"term_id": "Ab12Cd34", "line": "print(42)"},
        )
        await handler.on_tool_end(
            "Sent line to terminal Ab12Cd34", run_id="send"
        )
        await handler.on_tool_end("Terminal ID: Ab12Cd34", run_id="launch")
        await pilot.pause()

        cards = list(turn.query(TermCard))
        assert len(cards) == 1
        card = cards[0]
        assert card is not provisional
        assert card.term_id == "Ab12Cd34"
        assert card.call_count == 2
        assert turn._term_calls_by_id["launch"] is card
        assert turn._term_calls_by_id["send"] is card
        assert provisional.key not in turn.cards


async def test_custom_term_events_with_different_run_ids_do_not_duplicate(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    result = ToolMessage(
        content=[{"type": "text", "text": "Terminal ID: Ab12Cd34"}],
        tool_call_id="launch",
    )

    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("launch terminal", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "term"},
            "",
            run_id="launch",
            inputs={"cmd": "python", "session": True},
        )
        # This mirrors providers that surface an in-tool custom range without
        # the enclosing callback ID, followed by a child-ID result event.
        await turn.event({"tool": "term", "phase": "start"})
        await turn.event({
            "tool": "term",
            "phase": "start",
            "_run_id": "child",
        })
        await turn.event({
            "tool": "term",
            "phase": "end",
            "tool_message": result,
            "_run_id": "child",
        })
        await handler.on_tool_end(result, run_id="launch")
        await pilot.pause()

        cards = list(turn.query(TermCard))
        assert len(cards) == 1
        card = cards[0]
        assert card.term_id == "Ab12Cd34"
        assert card.call_count == 1
        assert turn._term_calls_by_id["launch"] is card
        assert turn._term_calls_by_id["child"] is card


async def test_distinct_parallel_launches_remain_separate(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("launch two terminals", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        for run_id in ("launch-one", "launch-two"):
            await handler.on_tool_start(
                {"name": "term"},
                "",
                run_id=run_id,
                inputs={"cmd": "python", "session": True},
            )
        assert len(list(turn.query(TermCard))) == 2

        await handler.on_tool_end("Terminal ID: Ab12Cd34", run_id="launch-one")
        await handler.on_tool_end("Terminal ID: Ef56Gh78", run_id="launch-two")
        await pilot.pause()

        cards = list(turn.query(TermCard))
        assert len(cards) == 2
        assert {card.term_id for card in cards} == {"Ab12Cd34", "Ef56Gh78"}
        assert all(card.call_count == 1 for card in cards)


def test_terminal_id_requires_documented_exact_result():
    message = ToolMessage(
        content=[{"type": "text", "text": "Terminal ID: Ab12Cd34"}],
        tool_call_id="launch",
    )
    assert terminal_id(message) == "Ab12Cd34"
    assert terminal_id("Terminal ID: Ab12Cd34\n") == "Ab12Cd34"
    assert terminal_id("result Terminal ID: Ab12Cd34") is None
    assert terminal_id("Ab12Cd34") is None
    assert terminal_id({"term_id": "Ab12Cd34"}) == "Ab12Cd34"
    assert terminal_id({"cmd": "Terminal ID: Ab12Cd34"}) is None
    assert terminal_id({"line": "Terminal ID: Ab12Cd34"}) is None


async def test_term_card_tail_and_live_view_refresh(tmp_path):
    manager = _TermCardManager()
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 30)) as pilot:
        card = TermCard("term:test", "Ab12Cd34", manager=manager)  # type: ignore[arg-type]
        await app.query_one("#conversation", VerticalScroll).mount(card)

        manager.contents_value = "first\nlast useful\n   \n"
        await card._refresh_tail()
        assert str(card.query_one(".term-card-tail", Static).content) == (
            "last useful"
        )
        manager.contents_value = "\n   \n"
        await card._refresh_tail()
        assert str(card.query_one(".term-card-tail", Static).content) == (
            "(terminal is empty)"
        )
        manager.missing = True
        await card._refresh_tail()
        assert "no longer exists" in str(
            card.query_one(".term-card-tail", Static).content
        )

        card.set_expanded(True)
        await pilot.pause()
        view = card.query_one(TerminalView)
        await view._update_snapshot()
        assert "live one" in str(view.content)
        manager.snapshot_value = "live two"
        await view._update_snapshot()
        await pilot.pause()
        assert "live two" in str(view.content)


async def test_term_card_shows_full_ghostty_grid_without_clipping(tmp_path):
    manager = _TermCardManager()
    manager.screen = True
    manager.snapshot_value = "\n".join(f"row {row}" for row in range(24))
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 30)) as pilot:
        card = TermCard("term:test", "Ab12Cd34", manager=manager)  # type: ignore[arg-type]
        await app.query_one("#conversation", VerticalScroll).mount(card)
        card.set_expanded(True)
        await pilot.pause()
        view = card.query_one(TerminalView)
        await view._update_snapshot()
        await pilot.pause()

        live = card.query_one(".term-card-live", Vertical)
        # The Ghostty child retains its exact 24x80 grid plus two border cells,
        # and its content-driven parent grows to expose the complete screen.
        assert view.outer_size.height == 26
        assert view.outer_size.width == 82
        assert live.size.height >= view.outer_size.height


async def test_term_card_full_ghostty_width_is_reachable_in_narrow_app(
    tmp_path,
):
    manager = _TermCardManager()
    manager.screen = True
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 30)) as pilot:
        card = TermCard("term:test", "Ab12Cd34", manager=manager)  # type: ignore[arg-type]
        await app.query_one("#conversation", VerticalScroll).mount(card)
        card.set_expanded(True)
        await pilot.pause()
        view = card.query_one(TerminalView)
        await view._update_snapshot()
        await pilot.pause()

        live = card.query_one(".term-card-live", Vertical)
        assert view.outer_size.width == 82
        assert live.virtual_size.width >= view.outer_size.width
        assert live.max_scroll_x > 0


@pytest.mark.parametrize("tool", sorted(TERM_TOOLS - {"term"}))
async def test_every_session_term_tool_routes_to_term_card(tmp_path, tool):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        turn = Turn("terminal operation", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "tool": tool,
            "phase": "start",
            "_run_id": tool,
            "term_id": "Ab12Cd34",
        })
        await pilot.pause()
        card = turn.query_one(TermCard)
        assert card.term_id == "Ab12Cd34"


async def test_arbitrary_eight_character_term_result_is_not_promoted(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        turn = Turn("quick output", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "term"}, "", run_id="quick", inputs={"cmd": "echo"}
        )
        await handler.on_tool_end("Ab12Cd34", run_id="quick")
        await pilot.pause()
        assert not list(turn.query(TermCard))
        assert turn.query_one(ToolCallCard).output == "Ab12Cd34"


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

        card = turn.query_one(ToolCallCard)
        assert card.completed
        assert card.output == output


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
