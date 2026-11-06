import asyncio

from langchain_core.messages import ToolMessage
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widgets import Button, Markdown, Static

from tests.cli._app_helpers import make_runtime, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import (
    AgentToolCard,
    CoordinationCard,
    ToolCallCard,
)
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn


async def test_parallel_coordination_calls_share_one_card_across_entire_turn(
    tmp_path,
):
    app = UrsaTextualApp(make_runtime(tmp_path))
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
                handler.on_tool_start(
                    {"name": tool},
                    "",
                    run_id=tool,
                    inputs={
                        "name": "a",
                        "target": "root/a",
                        "message": message,
                    },
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
        assert "4 active" in str(
            group.query_one(".coordination-summary", Static).content
        )

        # An unrelated call, an error, and later coordination still use the
        # same card. The old summary grouping grace period doesn't apply.
        await handler.on_tool_start({"name": "other"}, "", run_id="other")
        await handler.on_tool_end("done", run_id="other")
        await handler.on_tool_error(
            ValueError("Unknown agent path: ../foo"), run_id="stop_agent"
        )
        for tool in tools[:-1]:
            await handler.on_tool_end("[role=Villager] accepted", run_id=tool)
        await handler.on_tool_start(
            {"name": "send_message"},
            "",
            run_id="late",
            inputs={"target": "root/b", "message": "hello"},
        )
        await handler.on_tool_end("queued", run_id="late")
        assert len(turn.query(CoordinationCard)) == 1
        assert len(turn.query(ToolCallCard)) == 1
        summary = str(group.query_one(".coordination-summary", Static).content)
        assert "5 actions" in summary and "1 need attention" in summary
        recent = "\n".join(
            str(row.content)
            for row in group.query(".coordination-action-preview")
        )
        assert len(recent.splitlines()) == 2
        assert "Stop agent" in recent and "root/b" in recent
        assert "Spawn agent" not in recent
        assert len(recent) < 800

        group.set_expanded(True)
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == 5
        )
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
        await second_turn.event({
            "tool": "wait_for",
            "phase": "start",
            "_run_id": "next",
            "target": "root/a",
        })
        assert len(app.query(CoordinationCard)) == 2


async def test_coordination_actions_have_clear_status_and_keyboard_expansion(
    tmp_path,
):
    app = UrsaTextualApp(make_runtime(tmp_path))
    async with app.run_test(size=(100, 36)) as pilot:
        turn = await app._mount_turn("coordinate")
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "wait_for"},
            "",
            run_id="wait",
            inputs={
                "target": ["root/a", "root/b"],
                "timeout_seconds": 120,
                "wait_all": True,
            },
        )
        group = turn.query_one(CoordinationCard)
        assert group.can_focus
        assert "1 active" in str(
            group.query_one(".coordination-summary", Static).content
        )
        assert "Wait for → 2 agents" in str(
            group.query_one(".coordination-action-preview", Static).content
        )
        group.focus()
        await pilot.press("enter")
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == 1
        )
        card = group.query_one(AgentToolCard)
        assert card.can_focus
        assert "Click or Enter to collapse" in str(
            group.query_one(".event-expand-hint", Static).content
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
        group.set_expanded(True)
        assert "Click or Enter to collapse" in str(hint.content)

        await handler.on_tool_end(
            "All requested agents completed.", run_id="wait"
        )
        assert await wait_for(pilot, lambda: card.completed)
        assert "complete" in str(
            group.query_one(".coordination-summary", Static).content
        )
        assert "All requested agents completed." in str(
            card.query_one(".agent-tool-preview", Static).content
        )


async def test_compact_recipient_count_survives_long_target_preview(tmp_path):
    app = UrsaTextualApp(make_runtime(tmp_path))
    targets = [
        f"root/reviewer-with-a-long-name-{index:03}" for index in range(80)
    ]
    async with app.run_test(size=(100, 30)):
        turn = await app._mount_turn("broadcast")
        await turn.event({
            "tool": "send_message",
            "phase": "start",
            "_run_id": "broadcast",
            "target": targets,
            "message": "Coordinate findings.",
        })
        preview = turn.query_one(".coordination-action-preview", Static)
        assert "80 agents" in str(preview.content)
        assert targets[-1] not in str(preview.content)


async def test_expanded_coordination_pages_bound_widget_count(tmp_path):
    app = UrsaTextualApp(make_runtime(tmp_path))
    async with app.run_test(size=(110, 40)) as pilot:
        turn = await app._mount_turn("many calls")
        for index in range(25):
            await turn.event({
                "tool": "send_message",
                "phase": "start",
                "_run_id": str(index),
                "target": f"root/p{index}",
                "message": f"task {index}",
            })
            await turn.event({
                "tool": "send_message",
                "phase": "end",
                "_run_id": str(index),
                "result": "queued",
            })
        group = turn.query_one(CoordinationCard)
        assert not group.query(AgentToolCard)
        group.set_expanded(True)
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == group.PAGE_SIZE
        )
        assert group.query(AgentToolCard).first().target == "root/p5"
        group.query_one(".coordination-older", Button).press()
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == 5
        )
        assert group.query(AgentToolCard).first().target == "root/p0"
        # A new action must not displace a historical page being read.
        await turn.event({
            "tool": "send_message",
            "phase": "start",
            "_run_id": "newest",
            "target": "root/p25",
            "message": "new task",
        })
        await turn.event({
            "tool": "send_message",
            "phase": "end",
            "_run_id": "newest",
            "result": "queued",
        })
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
    tmp_path,
):
    app = UrsaTextualApp(make_runtime(tmp_path))
    async with app.run_test(size=(85, 30)) as pilot:
        turn = await app._mount_turn("coordinate")
        handler = TextualEventHandler(app, turn)
        error = "[Unknown target=../foo] " + "explanation " * 1000
        await handler.on_tool_start(
            {"name": "send_message"},
            "",
            run_id="bad",
            inputs={"target": "../foo", "message": "hello"},
        )
        await handler.on_tool_end(
            ToolMessage(error, tool_call_id="bad", status="error"), run_id="bad"
        )
        await handler.on_tool_start(
            {"name": "wait_for"}, "", run_id="wait", inputs={"target": "root/a"}
        )
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
        group.set_expanded(True)
        assert await wait_for(
            pilot, lambda: len(group.query(AgentToolCard)) == 2
        )
        card = group.query(AgentToolCard).first()
        assert card.failed and len(card.output) <= 241
        card.set_expanded(True)
        assert await wait_for(pilot, lambda: card.output == error)
