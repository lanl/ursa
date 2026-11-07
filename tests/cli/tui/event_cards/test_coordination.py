import asyncio

import pytest
from langchain_core.messages import ToolMessage
from textual.containers import Horizontal, VerticalScroll
from textual.geometry import Region
from textual.widgets import Button, Markdown, Static

from tests.cli._app_helpers import emit_tool_start, static_text, wait_for
from ursa.cli.tui.event_cards import (
    AgentToolCard,
    CoordinationCard,
    ToolCallCard,
)
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import PromptArea


async def emit_action(turn, tool, run_id, *, phase="start", **details):
    """Feed a coordination action through the turn's real event handler."""
    await turn.event(dict(tool=tool, phase=phase, _run_id=run_id, **details))


async def expand_actions(pilot, group, count):
    """Expand coordination and wait for its expected number of action cards."""
    group.set_expanded(True)
    assert await wait_for(
        pilot, lambda: len(group.query(AgentToolCard)) == count
    )


async def test_parallel_coordination_calls_share_one_card_across_entire_turn(
    tmp_path,
    app,
):
    body_reads = []
    action_body = app.message_store.action_body

    async def tracked_body(turn_id, action_id):
        body_reads.append(action_id)
        return await action_body(turn_id, action_id)

    app.message_store.action_body = tracked_body
    async with app.run_test(size=(110, 40)) as pilot:
        turn = await app._mount_turn("coordinate")
        handler = TextualEventHandler(app, turn)
        tools = ("spawn_agent", "send_message", "wait_for", "stop_agent")
        message = "large exact task\n" * 5000
        await asyncio.gather(
            *(
                emit_tool_start(
                    handler,
                    tool,
                    tool,
                    name="a",
                    target="root/a",
                    message=message,
                )
                for tool in tools
            )
        )
        assert len(turn.query(CoordinationCard)) == 1
        group = turn.query_one(CoordinationCard)
        assert group.total == 4
        assert not turn.query(AgentToolCard)
        assert all(
            "message" not in pending for pending in handler.tools.values()
        )
        assert not turn._tool_calls_by_id
        assert "4 active" in static_text(group, ".coordination-summary")

        # An unrelated call, an error, and later coordination still use the
        # same card. The old summary grouping grace period doesn't apply.
        await handler.on_tool_start({"name": "other"}, "", run_id="other")
        await handler.on_tool_end("done", run_id="other")
        await handler.on_tool_error(
            ValueError("Unknown agent path: ../foo"), run_id="stop_agent"
        )
        for tool in tools[:-1]:
            await handler.on_tool_end("[role=Villager] accepted", run_id=tool)
        await emit_tool_start(
            handler, "send_message", "late", target="root/b", message="hello"
        )
        await handler.on_tool_end("queued", run_id="late")
        assert len(turn.query(CoordinationCard)) == 1
        assert len(turn.query(ToolCallCard)) == 1
        summary = static_text(group, ".coordination-summary")
        assert "5 actions" in summary and "1 action needs attention" in summary
        assert group.has_class("coordination-needs-attention")
        recent = "\n".join(
            str(row.content)
            for row in group.query(".coordination-action-preview")
        )
        assert len(recent.splitlines()) == 2
        assert "Needs attention · Stop agent" in recent and "root/b" in recent
        assert "Spawn agent" not in recent
        assert len(recent) < 800

        await expand_actions(pilot, group, 5)
        assert not body_reads
        assert all(
            "message" not in card.tool_input
            for card in group.query(AgentToolCard)
        )
        spawn = next(
            card
            for card in group.query(AgentToolCard)
            if card.tool == "spawn_agent"
        )
        spawn.set_expanded(True)
        assert await wait_for(pilot, lambda: spawn.task == message.strip())
        assert len(body_reads) == 1
        assert (
            "large exact task"
            in spawn.query_one(".agent-tool-task", Markdown).source
        )
        failed = next(
            card
            for card in group.query(AgentToolCard)
            if card.tool == "stop_agent"
        )
        assert failed.failed
        spawn.set_expanded(False)
        assert "message" not in spawn.tool_input
        assert not spawn.query_one(".agent-tool-task", Markdown).source
        group.set_expanded(False)
        assert await wait_for(pilot, lambda: not group.query(AgentToolCard))
        assert not group.query(Markdown)

        second_turn = Turn("next turn", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(second_turn)
        await emit_action(second_turn, "wait_for", "next", target="root/a")
        assert len(app.query(CoordinationCard)) == 2


async def test_coordination_actions_have_clear_status_and_keyboard_expansion(
    app,
):
    async with app.run_test(size=(100, 36)) as pilot:
        turn = await app._mount_turn("coordinate")
        handler = TextualEventHandler(app, turn)
        await emit_tool_start(
            handler,
            "wait_for",
            "wait",
            target=["root/a", "root/b"],
            timeout_seconds=120,
            wait_all=True,
        )
        group = turn.query_one(CoordinationCard)
        assert group.can_focus
        assert "1 active" in static_text(group, ".coordination-summary")
        assert "Wait for → 2 agents" in static_text(
            group, ".coordination-action-preview"
        )
        group.focus()
        await pilot.press("enter")
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == 1
        )
        card = group.query_one(AgentToolCard)
        assert card.can_focus
        assert "Click or Enter to collapse" in static_text(
            group, ".event-expand-hint"
        )
        card.focus()
        await pilot.press("space")
        assert await wait_for(
            pilot,
            lambda: "all agents finish"
            in card.query_one(".agent-tool-task", Markdown).source,
        )
        request = card.query_one(".agent-tool-task", Markdown).source
        assert "120 seconds" in request
        assert "`root/a`" in request and "`root/b`" in request
        # The nested action owns a hint too. Updating it must never replace
        # the coordination group's direct hint.
        hint = next(
            child
            for child in group.children
            if child.has_class("event-expand-hint")
        )
        assert "Click or Enter to collapse" in str(hint.content)
        group.set_expanded(False)
        assert "Click or Enter to expand" in str(hint.content)
        conversation = app.query_one("#conversation", VerticalScroll)
        scroll_y = conversation.scroll_y
        group._reveal_expanded_card()
        assert conversation.scroll_y == scroll_y
        group.set_expanded(True)
        assert "Click or Enter to collapse" in str(hint.content)

        await handler.on_tool_end(
            "All requested agents completed.", run_id="wait"
        )
        assert await wait_for(pilot, lambda: card.completed)
        assert "complete" in static_text(group, ".coordination-summary")
        assert "All requested agents completed." in static_text(
            card, ".agent-tool-preview"
        )


async def test_compact_recipient_count_survives_long_target_preview(app):
    targets = [
        f"root/reviewer-with-a-long-name-{index:03}" for index in range(80)
    ]
    async with app.run_test(size=(100, 30)):
        turn = await app._mount_turn("broadcast")
        await emit_action(
            turn,
            "send_message",
            "broadcast",
            target=targets,
            message="Coordinate findings.",
        )
        preview = turn.query_one(".coordination-action-preview", Static)
        assert "80 agents" in str(preview.content)
        assert targets[-1] not in str(preview.content)


async def test_group_target_expressions_are_never_presented_as_agent_counts(
    app,
):
    long_paths = [f"root/reviewer-{index:03}" for index in range(80)]
    async with app.run_test(size=(100, 30)) as pilot:
        turn = await app._mount_turn("broadcast")
        for run_id, target in (
            ("everyone", "@everyone"),
            ("mixed", ["@peers", "root/reviewer"]),
            ("long-mixed", [*long_paths, "@reports"]),
        ):
            await emit_action(
                turn,
                "send_message",
                run_id,
                target=target,
                message="Coordinate findings.",
            )
        group = turn.query_one(CoordinationCard)
        recent = "\n".join(
            str(row.content)
            for row in group.query(".coordination-action-preview")
        )
        assert "@peers + 1 explicit target" in recent
        assert "@reports + 80 explicit targets" in recent
        assert "agents" not in recent

        await expand_actions(pilot, group, 3)
        cards = {card.key: card for card in group.query(AgentToolCard)}
        assert cards["1"].target == "@everyone"
        assert cards["2"].target == "@peers + 1 explicit target"
        assert cards["3"].target == "@reports + 80 explicit targets"
        for run_id in ("everyone", "mixed", "long-mixed"):
            await emit_action(
                turn,
                "send_message",
                run_id,
                phase="end",
                result="Message queued for root/reviewer.",
            )
        previews = [
            static_text(card, ".agent-tool-preview") for card in cards.values()
        ]
        assert all("agents" not in preview for preview in previews)
        assert "@peers + 1 explicit target" in previews[1]


@pytest.mark.parametrize("size", [(40, 20), (60, 20), (80, 24), (120, 40)])
async def test_coordination_controls_and_recent_actions_fit_narrow_terminals(
    app, size
):
    async with app.run_test(size=size) as pilot:
        turn = await app._mount_turn("coordinate")
        await emit_action(
            turn,
            "stop_agent",
            "stop",
            target="root/reviewer-with-a-descriptive-name",
        )
        await emit_action(
            turn,
            "stop_agent",
            "stop",
            phase="error",
            target="root/reviewer-with-a-descriptive-name",
            error="Agent is stopping",
        )
        await emit_action(
            turn,
            "send_message",
            "message",
            target="root/researcher-with-a-descriptive-name",
            message="Please share your final result.",
        )
        await emit_action(
            turn,
            "send_message",
            "message",
            phase="end",
            result="Message queued for root/researcher-with-a-descriptive-name.",
        )
        group = turn.query_one(CoordinationCard)
        prompt = app.query_one(PromptArea)
        conversation = app.query_one("#conversation", VerticalScroll)
        # Simulate the ordinary live view, where the conversation follows the
        # bottom until a user deliberately opens this card.
        conversation.scroll_end(animate=False, immediate=True)
        conversation.anchor()
        assert conversation.is_anchored
        prompt.focus()
        await pilot.press("shift+tab")
        assert app.focused is group
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: bool(group.query(AgentToolCard)))
        await pilot.pause()

        navigation = group.query_one(".coordination-navigation", Horizontal)
        page = group.query_one(".coordination-page", Static)
        buttons = list(navigation.query(Button))
        assert page.region.width > 0
        assert page.region.y == navigation.region.y
        assert all(button.region.height == 1 for button in buttons)
        assert all(button.region.y == navigation.region.y for button in buttons)
        assert all(
            navigation.content_region.x <= button.region.x
            and button.region.right <= navigation.content_region.right
            for button in buttons
        )
        viewport = conversation.content_region
        header = group.query_one(".coordination-summary", Static)
        assert header.region.y >= viewport.y
        assert navigation.region.y >= viewport.y
        assert navigation.region.bottom <= viewport.bottom
        previews = list(group.query(".coordination-action-preview"))
        preview_text = "\n".join(str(preview.content) for preview in previews)
        assert "Needs attention" in preview_text
        assert "Send message" in preview_text
        # Later activity must not re-anchor the reader at the bottom and hide
        # the controls that were deliberately revealed above.
        await emit_action(
            turn, "wait_for", "background", target="root/another-agent"
        )
        await pilot.pause()
        assert header.region.y >= viewport.y
        assert navigation.region.y >= viewport.y
        assert navigation.region.bottom <= viewport.bottom
        previews = list(group.query(".coordination-action-preview"))
        assert all(
            preview.region.width <= group.content_region.width
            for preview in previews
        )
        assert group.has_class("coordination-needs-attention")
        assert "1 action needs attention" in static_text(
            group, ".coordination-summary"
        )


async def test_expanded_coordination_pages_bound_widget_count(app):
    async with app.run_test(size=(110, 40)) as pilot:
        turn = await app._mount_turn("many calls")
        for index in range(25):
            await emit_action(
                turn,
                "send_message",
                str(index),
                target=f"root/p{index}",
                message=f"task {index}",
            )
            await emit_action(
                turn, "send_message", str(index), phase="end", result="queued"
            )
        group = turn.query_one(CoordinationCard)
        assert not group.query(AgentToolCard)
        await expand_actions(pilot, group, group.PAGE_SIZE)
        assert group.query(AgentToolCard).first().target == "root/p5"
        group.query_one(".coordination-older", Button).press()
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == 5
        )
        assert group.query(AgentToolCard).first().target == "root/p0"
        # A new action must not displace a historical page being read.
        await emit_action(
            turn,
            "send_message",
            "newest",
            target="root/p25",
            message="new task",
        )
        await emit_action(
            turn, "send_message", "newest", phase="end", result="queued"
        )
        assert await wait_for(
            pilot,
            lambda: [card.target for card in group.query(AgentToolCard)]
            == [f"root/p{index}" for index in range(5)],
        )
        group.query_one(".coordination-newer", Button).press()
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == group.PAGE_SIZE
        )
        assert group.query(AgentToolCard).last().target == "root/p24"
        group.query_one(".coordination-newer", Button).press()
        assert await wait_for(
            pilot,
            lambda: group.query(AgentToolCard).last().target == "root/p25",
        )


async def test_coordination_errors_keep_exact_body_and_two_visible_rows(
    app,
):
    async with app.run_test(size=(85, 30)) as pilot:
        turn = await app._mount_turn("coordinate")
        handler = TextualEventHandler(app, turn)
        error = "[Unknown target=../foo] " + "explanation " * 1000
        await emit_tool_start(
            handler, "send_message", "bad", target="../foo", message="hello"
        )
        await handler.on_tool_end(
            ToolMessage(error, tool_call_id="bad", status="error"), run_id="bad"
        )
        await emit_tool_start(handler, "wait_for", "wait", target="root/a")
        group = turn.query_one(CoordinationCard)
        await pilot.pause()
        rows = list(group.query(".coordination-action-preview"))
        rendered = [
            row.render_lines(Region(0, 0, row.size.width, 1))[0] for row in rows
        ]
        assert len(rendered) == 2
        assert "Send message" in rendered[0].text
        assert "Wait for" in rendered[1].text
        assert rows[1].region.y == rows[0].region.y + 1
        await expand_actions(pilot, group, 2)
        card = group.query(AgentToolCard).first()
        assert card.failed and len(card.output) <= 241
        assert card.has_class("agent-tool-failed")
        assert "Failed: [Unknown target=../foo]" in static_text(
            card, ".agent-tool-preview"
        )
        card.set_expanded(True)
        assert await wait_for(pilot, lambda: card.output == error)
