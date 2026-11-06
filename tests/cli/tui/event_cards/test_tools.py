import json

import pytest
from langchain_core.messages import ToolMessage
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from tests.cli._app_helpers import make_runtime, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import (
    AgentToolCard,
    CoordinationCard,
    ToolCallCard,
)
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn


async def test_default_tool_card_switches_from_input_to_output(tmp_path, app):
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


async def test_spawn_agent_card_shows_task_and_queue_acknowledgement(
    tmp_path, app
):
    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("delegate it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "spawn_agent"},
            "",
            run_id="spawn-reviewer",
            inputs={
                "name": "reviewer",
                "message": "Review the implementation for correctness.",
                "agent_type": "deep_review",
            },
        )
        await pilot.pause()

        group = turn.query_one(CoordinationCard)
        assert not turn.query(AgentToolCard)
        group.set_expanded(True)
        assert await wait_for(pilot, lambda: bool(turn.query(AgentToolCard)))
        card = turn.query_one(AgentToolCard)
        assert card.tool == "spawn_agent"
        assert str(card.query_one(".agent-tool-title", Static).content) == (
            "✦ Delegating work"
        )
        assert "reviewer" in str(
            card.query_one(".agent-tool-preview", Static).content
        )
        assert "deep_review" in str(
            card.query_one(".agent-tool-preview", Static).content
        )
        assert card.query_one(".agent-tool-details").has_class("hidden")

        card.set_expanded(True)
        assert await wait_for(
            pilot, lambda: card.task.startswith("Review the implementation")
        )
        assert "Review the implementation" in str(
            card.query_one(".agent-tool-task", Markdown).source
        )
        assert "reviewer" in str(
            card.query_one(".agent-tool-task", Markdown).source
        )

        await handler.on_tool_end(
            "Spawned agent root/reviewer\n\nMessage queued for root/reviewer.",
            run_id="spawn-reviewer",
        )
        await pilot.pause()

        assert card.completed
        assert card.query_one(".agent-tool-state", Static).content == "✓"
        assert "started" in str(
            card.query_one(".agent-tool-preview", Static).content
        )
        assert "Message queued" in str(
            card.query_one(".agent-tool-result", Markdown).source
        )


async def test_agent_message_and_listing_cards_render_specialized_outcomes(
    tmp_path,
    app,
):
    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("coordinate it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "send_message"},
            "",
            run_id="message-reviewer",
            inputs={"target": "root/reviewer", "message": "Check tests."},
        )
        await handler.on_tool_end(
            "Message queued for root/reviewer.",
            run_id="message-reviewer",
        )
        await handler.on_tool_start(
            {"name": "list_agents"}, "", run_id="list", inputs={}
        )
        await handler.on_tool_end("root (self)\nroot/reviewer", run_id="list")
        await pilot.pause()

        turn.query_one(CoordinationCard).set_expanded(True)
        assert await wait_for(
            pilot, lambda: len(turn.query(AgentToolCard)) == 2
        )
        cards = list(turn.query(AgentToolCard))
        assert len(cards) == 2
        message, listing = cards
        assert "Message queued for root/reviewer" in str(
            message.query_one(".agent-tool-preview", Static).content
        )
        assert "2 agents in the hierarchy" in str(
            listing.query_one(".agent-tool-preview", Static).content
        )
        listing.set_expanded(True)
        assert "root/reviewer" in str(
            listing.query_one(".agent-tool-result", Markdown).source
        )


async def test_agent_message_card_summarizes_broadcast_results(tmp_path, app):
    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("coordinate everyone", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "send_message"},
            "",
            run_id="broadcast",
            inputs={
                "target": ["root/analyst", "root/reviewer", "root/missing"],
                "message": "Compare findings.",
            },
        )
        await handler.on_tool_end(
            "root/analyst: Message queued for root/analyst.\n\n"
            "root/reviewer: Message queued for root/reviewer.\n\n"
            "root/critic: Message queued for root/critic.",
            run_id="broadcast",
        )
        await pilot.pause()

        turn.query_one(CoordinationCard).set_expanded(True)
        assert await wait_for(pilot, lambda: bool(turn.query(AgentToolCard)))
        card = turn.query_one(AgentToolCard)
        assert "Messages queued for 3 agents" in str(
            card.query_one(".agent-tool-preview", Static).content
        )


async def test_wait_card_renders_batched_agent_messages_as_plain_text(tmp_path):
    """Bracketed delivery JSON is content, never Textual markup."""
    app = UrsaTextualApp(make_runtime(tmp_path))
    output = (
        "Received 2 messages while waiting. Each entry retains its sender, "
        "recipient, and exact content:\n"
        '[{"sender": "root/p08", "recipient": "root", '
        '"content": "SETUP ACK: role=Villager"}, '
        '{"sender": "root/p20", "recipient": "root", '
        '"content": "SETUP ACK: identity p20"}]'
    )

    async with app.run_test(size=(125, 24)) as pilot:
        turn = Turn("coordinate", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "wait_for"},
            "",
            run_id="batched-wait",
            inputs={"target": ["root/p08", "root/p20"]},
        )
        await handler.on_tool_end(output, run_id="batched-wait")
        # Rendering the mounted card used to raise MarkupError here.
        await pilot.pause()

        turn.query_one(CoordinationCard).set_expanded(True)
        assert await wait_for(pilot, lambda: bool(turn.query(AgentToolCard)))
        card = turn.query_one(AgentToolCard)
        preview = card.query_one(".agent-tool-preview", Static)
        assert card.completed
        assert len(card.output) <= 241
        card.set_expanded(True)
        assert await wait_for(pilot, lambda: card.output == output)
        assert str(preview.content) == output
        assert '"sender": "root/p08"' in str(preview.content)


async def test_agent_tool_card_survives_unavailable_lazy_details(tmp_path, app):
    async def unavailable():
        raise LookupError("action was pruned")

    async with app.run_test(size=(100, 30)) as pilot:
        turn = Turn("coordinate", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        card = AgentToolCard(
            "missing-details",
            "send_message",
            {"target": "root/reviewer"},
            details_loader=unavailable,
        )
        await turn._replace_or_mount("missing-details", card)
        card.complete("Message queued for root/reviewer.")
        card.set_expanded(True)
        assert await wait_for(
            pilot,
            lambda: "Details unavailable"
            in card.query_one(".agent-tool-task", Markdown).source,
        )
        assert card.completed and not card.failed
        assert "Message queued" in str(
            card.query_one(".agent-tool-result", Markdown).source
        )


@pytest.mark.parametrize(
    ("tool", "output", "expected"),
    [
        pytest.param(
            "lookup_widget",
            ValueError("bad filter"),
            "bad filter",
            id="failure",
        ),
        pytest.param(
            "edit_plot",
            "[patch={'old_code': 'ax.set_xticklabels([labels])'}]",
            "[patch={'old_code': 'ax.set_xticklabels([labels])'}]",
            id="literal-code-brackets",
        ),
        pytest.param(
            "lookup_widget",
            ToolMessage(
                content="Human-readable fallback",
                artifact={
                    "structured_content": {"matches": ["polar", "grizzly"]}
                },
                tool_call_id="result",
            ),
            {"matches": ["polar", "grizzly"]},
            id="structured-tool-message",
        ),
        pytest.param(
            "lookup_widget",
            ToolMessage(
                content=[
                    {"type": "text", "text": "## Result\n\n**Found it.**"}
                ],
                tool_call_id="result",
            ),
            "## Result\n\n**Found it.**",
            id="text-block-tool-message",
        ),
    ],
)
async def test_tool_card_output_formats_and_failure_state(
    tmp_path, app, tool, output, expected
):
    async with app.run_test(size=(100, 24)) as pilot:
        turn = Turn("call it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": tool}, "", run_id="result", inputs={}
        )
        failed = isinstance(output, Exception)
        if failed:
            await handler.on_tool_error(output, run_id="result")
        else:
            await handler.on_tool_end(output, run_id="result")
        await pilot.pause()

        card = turn.query_one(ToolCallCard)
        assert card.completed and card.failed == failed
        assert card.query_one(".tool-call-state", Static).content == (
            "✗" if failed else "✓"
        )
        if failed or isinstance(output, str):
            assert card.output == expected
        card.set_expanded(True)
        json_output = card.query_one(".tool-output-json", Static)
        markdown = card.query_one(".tool-output-markdown", Markdown)
        structured = isinstance(expected, dict)
        assert json_output.has_class("hidden") != structured
        assert markdown.has_class("hidden") == structured
        if structured:
            assert json.loads(json_output.content.code) == expected
        else:
            assert markdown.source == expected
