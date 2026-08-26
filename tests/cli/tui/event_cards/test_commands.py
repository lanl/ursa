import asyncio

from textual.containers import VerticalScroll
from textual.widgets import Static

from tests.cli._app_fakes import FakeHITL, emit_event
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import CommandSafetyIndicator, RunCommandCard
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import ActivityIndicator
from ursa.util.events import DEFAULT_EVENT_NAME


def test_long_command_preview_keeps_top_and_bottom_eight_lines():
    command = "\n".join(f"line {index}" for index in range(1, 26))

    preview = RunCommandCard._preview_command(command)

    assert preview.splitlines() == [
        *(f"line {index}" for index in range(1, 9)),
        "… 9 lines omitted …",
        *(f"line {index}" for index in range(18, 26)),
    ]


async def test_overlapping_commands_stay_compact_and_complete_independently(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("run both", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        for command_id, query in (("one", "sleep 1"), ("two", "sleep 2")):
            await turn.event({
                "tool": "run_command",
                "phase": "start",
                "query": query,
                "_command_id": command_id,
            })
        await pilot.pause()

        cards = list(turn.query(RunCommandCard))
        assert len(cards) == 2
        assert all(card.multi_command for card in cards)
        assert all(
            not card.query_one(".command-compact").has_class("hidden")
            for card in cards
        )

        await turn.event({
            "tool": "run_command",
            "phase": "end",
            "query": "sleep 1",
            "_command_id": "one",
            "returncode": 0,
            "result": "",
        })
        await pilot.pause()
        assert cards[0].completed
        assert cards[0].returncode == 0
        assert not cards[1].completed
        assert (
            str(cards[1].query_one(".command-compact-state", Static).content)
            in ActivityIndicator.FRAMES
        )


async def test_identical_concurrent_commands_are_correlated_by_run_id(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("run both", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        for run_id in ("one", "two"):
            await handler.on_tool_start(
                {"name": "run_command"},
                "",
                run_id=run_id,
                inputs={"query": "echo same"},
            )
        await handler.on_tool_error(RuntimeError("first failed"), run_id="one")
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "query": "echo same",
                "safe": True,
            },
            run_id="child-event-run",
        )
        await handler.on_tool_end("second output", run_id="two")
        await pilot.pause()

        first, second = turn.query(RunCommandCard)
        assert first.query_one(CommandSafetyIndicator).status == "unavailable"
        assert first.query_one(".command-compact-state", Static).content == "✗"
        assert second.query_one(CommandSafetyIndicator).status == "passed"
        assert second.query_one(".command-compact-state", Static).content == "✓"
        assert (
            second.query_one(".command-output", Static).content.code
            == "second output"
        )


async def test_command_completion_finishes_pending_safety_indicator(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("run it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "tool": "run_command",
            "phase": "start",
            "query": "uptime",
            "_command_id": "uptime",
        })
        await turn.event({
            "tool": "run_command",
            "phase": "end",
            "query": "uptime",
            "_command_id": "uptime",
            "result": "up 10 days",
        })
        await pilot.pause()

        safety = turn.query_one(CommandSafetyIndicator)
        assert safety.status == "passed"


async def test_solitary_command_after_overlap_returns_to_detailed_layout(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("run commands", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        for command_id in ("one", "two"):
            await turn.event({
                "tool": "run_command",
                "phase": "start",
                "query": command_id,
                "_command_id": command_id,
            })
        for command_id in ("one", "two"):
            await turn.event({
                "tool": "run_command",
                "phase": "end",
                "query": command_id,
                "_command_id": command_id,
                "returncode": 0,
                "result": "done",
            })
        await turn.event({
            "tool": "run_command",
            "phase": "start",
            "query": "three",
            "_command_id": "three",
        })
        await pilot.pause()

        first, second, third = turn.query(RunCommandCard)
        assert first.multi_command
        assert second.multi_command
        assert not third.multi_command
        assert third.query_one(".command-compact").has_class("hidden")


async def test_run_command_card_tracks_safety_and_collapses_on_result(tmp_path):
    hitl = FakeHITL(tmp_path)
    pass_safety = asyncio.Event()
    return_result = asyncio.Event()
    return_second = asyncio.Event()
    command = "\n".join(f"echo line-{index}" for index in range(1, 7))

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="command-1",
            inputs={"query": command},
        )
        await pass_safety.wait()
        await emit_event(
            handler,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "message": "Command passed safety check",
                "query": command,
                "safe": True,
            },
        )
        await return_result.wait()
        await handler.on_tool_end(
            "STDOUT:\ncommand output\nSTDERR:\n", run_id="command-1"
        )
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="command-2",
            inputs={"query": "echo second"},
        )
        await emit_event(
            handler,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "message": "Command passed safety check",
                "query": "echo second",
                "safe": True,
            },
        )
        await return_second.wait()
        await handler.on_tool_end(
            "STDOUT:\nsecond output\nSTDERR:\n", run_id="command-2"
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        card = app.query_one(RunCommandCard)
        source = card.query_one(".command-source", Static)
        safety = card.query_one(CommandSafetyIndicator)
        assert len(source.content.code.splitlines()) == 6
        assert (
            str(safety.query_one(".activity-text", Static).content)
            == "Running safety check"
        )

        pass_safety.set()
        await pilot.pause()
        assert safety.status == "passed"

        return_result.set()
        await pilot.pause()
        assert source.content.code == "echo line-1 …"
        output = card.query_one(".command-output", Static)
        assert output.content.code == "command output"
        cards = list(app.query(RunCommandCard))
        assert len(cards) == 2
        assert [item.command for item in cards] == [command, "echo second"]
        assert output.has_class("hidden")
        assert not card.query_one(".command-compact").has_class("hidden")

        return_second.set()
        await pilot.pause()
        assert card.returncode is None
        assert card.query_one(".command-compact-state", Static).content == "✓"
        newest = cards[-1]
        assert not newest.query_one(".command-compact").has_class("hidden")
        assert newest.query_one(".command-output").has_class("hidden")

        await pilot.press("ctrl+o")
        assert not output.has_class("hidden")
        assert source.content.code == command

        await pilot.press("ctrl+o")
        assert output.has_class("hidden")
        assert source.content.code == "echo line-1 …"


async def test_single_command_output_preserves_top_and_bottom_until_expanded(
    tmp_path,
):
    hitl = FakeHITL(tmp_path)
    full_output = "\n".join(f"line {index}" for index in range(1, 31))

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="long-output",
            inputs={"query": "generate output"},
        )
        await emit_event(
            handler,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "query": "generate output",
                "safe": True,
            },
        )
        await handler.on_tool_end(full_output, run_id="long-output")
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        output = app.query_one(".command-output", Static)
        preview = output.content.code
        assert len(preview.splitlines()) == 9
        assert "line 1\n" in preview
        assert "line 4\n… 22 lines omitted …\nline 27" in preview
        assert preview.endswith("line 30")

        await pilot.press("ctrl+o")
        assert output.content.code == full_output


async def test_collapsed_commands_retain_execution_outcomes(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]

        async def command(
            run_id,
            query,
            *,
            safe=True,
            returncode=None,
            output="",
        ):
            await handler.on_tool_start(
                {"name": "run_command"},
                "",
                run_id=run_id,
                inputs={"query": query},
            )
            await emit_event(
                handler,
                {
                    "tool": "run_command",
                    "stage": "safety_check",
                    "query": query,
                    "safe": safe,
                    "reason": "Rejected" if not safe else "Allowed",
                },
            )
            if returncode is not None:
                await emit_event(
                    handler,
                    {
                        "tool": "run_command",
                        "stage": "execute",
                        "phase": "end",
                        "query": query,
                        "returncode": returncode,
                    },
                )
            await handler.on_tool_end(output, run_id=run_id)

        await command("empty", "true", returncode=0)
        await command("failed", "false", returncode=2, output="failed")
        await command("unsafe", "dangerous", safe=False, output="Rejected")
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        cards = list(app.query(RunCommandCard))
        assert [card.completed for card in cards] == [True, True, True]
        assert [card.returncode for card in cards] == [0, 2, None]
        assert [card.safety_failed for card in cards] == [False, False, True]
        assert [
            card.query_one(".command-compact-state", Static).content
            for card in cards
        ] == ["✓", "✗", "⚔️"]
        assert all(
            not card.query_one(".command-compact").has_class("hidden")
            for card in cards
        )
        assert cards[-1].query_one(CommandSafetyIndicator).status == "failed"
