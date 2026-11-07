import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static

import ursa.cli.tui.event_handler as event_handler_module
import ursa.cli.tui.turn as turn_module
from tests.cli._app_helpers import emit_event, emit_tool_start
from ursa.cli.tui.event_cards import EditCard, FileActivityCard, RunCommandCard
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import ActivityIndicator, MessageCard
from ursa.util.events import DEFAULT_EVENT_NAME


def _file_activity(path, *, tool="read_file", message="Reading", **details):
    """Build independent file events with the common activity fields."""
    return {"tool": tool, "path": path, "message": message, **details}


async def _emit_file_activity(handler, path, **details):
    await emit_event(handler, _file_activity(path, **details))


async def test_turn_spacing_is_one_row_with_or_without_events(tmp_path, app):
    async with app.run_test(size=(100, 40)) as pilot:
        conversation = app.query_one("#conversation", VerticalScroll)

        thinking = Turn("think about it", tmp_path)
        await conversation.mount(thinking)
        await pilot.pause()
        thinking_user = thinking.query_one(MessageCard)
        activity = thinking.query_one(ActivityIndicator)
        assert activity.region.y - thinking_user.region.bottom == 1

        no_tools = Turn("answer directly", tmp_path)
        await conversation.mount(no_tools)
        await no_tools.add_response("Direct answer")
        no_tools.finish_activity()

        with_tool = Turn("run a command", tmp_path)
        await conversation.mount(with_tool)
        await with_tool.event({
            "tool": "run_command",
            "phase": "start",
            "query": "pwd",
            "_command_id": "pwd",
        })
        await with_tool.event({
            "tool": "run_command",
            "phase": "end",
            "query": "pwd",
            "_command_id": "pwd",
            "result": str(tmp_path),
            "returncode": 0,
        })
        await with_tool.add_response("Command answer")
        with_tool.finish_activity()
        await pilot.pause()

        direct_messages = list(no_tools.query(MessageCard))
        assert (
            direct_messages[1].region.y - direct_messages[0].region.bottom == 1
        )

        tool_messages = list(with_tool.query(MessageCard))
        command = with_tool.query_one(RunCommandCard)
        assert command.region.y - tool_messages[0].region.bottom == 1
        assert tool_messages[1].region.y - command.region.bottom == 1


async def test_files_are_grouped_by_read_write_and_edit_operations(
    tmp_path, app
):
    async with app.run_test(size=(100, 36)) as pilot:
        async with running_turn(app, pilot) as handler:
            events = [
                _file_activity("src/read.py"),
                _file_activity("src/other.py"),
                _file_activity(
                    "src/new.py",
                    tool="write_code",
                    code="new file\n",
                    message="Writing",
                ),
                _file_activity(
                    "src/edit.py",
                    tool="edit_code",
                    old_code="old\n",
                    new_code="new\nmore\n",
                    message="Editing",
                ),
                _file_activity(
                    str(tmp_path / "src" / "edit.py"),
                    tool="edit_code",
                    message="Editing",
                ),
            ]
            for event in events:
                await handler.on_custom_event(DEFAULT_EVENT_NAME, event)
        file_groups = list(app.query(FileActivityCard))
        assert len(file_groups) == 1
        reading = file_groups[0]
        assert reading.files == {
            "Reading": {
                str(Path("src/read.py")): (None, None),
                str(Path("src/other.py")): (None, None),
            },
            "Editing": {},
        }
        reading_summary = reading.query_one(".file-summary", Static).content
        assert all(
            path in reading_summary.plain
            for path in (str(Path("src/read.py")), str(Path("src/other.py")))
        )
        writes = list(app.query(EditCard))
        assert [edit.path for edit in writes] == [
            str(Path("src/new.py")),
            str(Path("src/edit.py")),
        ]
        assert [(edit.additions, edit.deletions) for edit in writes] == [
            (1, 0),
            (2, 1),
        ]


async def test_event_summary_groups_follow_activity_order(app):
    async with app.run_test(size=(100, 36)) as pilot:
        async with running_turn(app, pilot) as handler:
            for path in ("fileA", "fileB", "fileC"):
                await _emit_file_activity(handler, path)
            await _emit_file_activity(
                handler,
                "fileB",
                tool="edit_code",
                message="Editing",
                additions=2,
                deletions=1,
            )
            await emit_tool_start(
                handler, "run_command", "ordered-command", query="pwd"
            )
            await handler.on_tool_end("", run_id="ordered-command")
            await _emit_file_activity(handler, "fileA")
        events = list(app.query_one(Turn).query_one(".events").children)
        assert [type(event) for event in events] == [
            FileActivityCard,
            FileActivityCard,
            RunCommandCard,
        ]
        assert list(events[0].files["Reading"]) == [
            "fileA",
            "fileB",
            "fileC",
        ]
        assert list(events[1].files["Editing"]) == ["fileB"]


@pytest.fixture
def event_clock(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(turn_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(event_handler_module, "monotonic", lambda: clock[0])
    return clock


@pytest.fixture
def summary_timers(monkeypatch):
    timers = []

    class CapturedTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.stopped = False

        def stop(self):
            self.stopped = True

        def fire(self):
            assert not self.stopped
            self.callback()

    def capture_timer(_turn, delay, callback):
        timer = CapturedTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(Turn, "set_timer", capture_timer)
    return timers


@asynccontextmanager
async def running_turn(app, pilot):
    """Expose real submission callbacks while the agent remains working."""
    ready = asyncio.get_running_loop().create_future()
    finish = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        ready.set_result(callbacks[0])
        await finish.wait()
        return "Finished"

    app.runtime.run_agent = run_agent
    await pilot.press("g", "o", "enter")
    try:
        yield await ready
    finally:
        finish.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


async def test_activity_kinds_keep_independent_cards_open(
    app, monkeypatch, event_clock, summary_timers
):
    monkeypatch.setattr(turn_module, "SUMMARY_GROUP_GRACE_SECONDS", 1.0)
    async with app.run_test(size=(100, 36)) as pilot:
        async with running_turn(app, pilot) as handler:
            await _emit_file_activity(handler, "fileA")
            await pilot.pause()
            reading = app.query_one(FileActivityCard)
            await _emit_file_activity(
                handler,
                "fileB",
                tool="edit_code",
                message="Editing",
                additions=2,
                deletions=1,
            )
            await pilot.pause()
            groups = list(app.query(FileActivityCard))
            assert len(groups) == 2
            editing = groups[1]
            assert not reading.done
            assert not editing.done

            await _emit_file_activity(handler, "fileC")
            await pilot.pause()
            assert list(reading.files["Reading"]) == ["fileA", "fileC"]
            assert len(app.query(FileActivityCard)) == 2
            assert not reading.done
            assert not editing.done

            old_reading, editing_timer, current_reading = summary_timers
            assert old_reading.stopped
            editing_timer.fire()
            await pilot.pause()
            assert not reading.done
            assert editing.done

            current_reading.fire()
            await pilot.pause()
            assert reading.done


async def test_event_group_grace_period_resets_after_each_update(
    app, event_clock
):
    async with app.run_test(size=(100, 36)) as pilot:
        async with running_turn(app, pilot) as handler:
            # C is past the original deadline, within B's renewed deadline;
            # D is past C's renewed deadline and must create a separate group.
            for path, elapsed in (
                ("fileA", 0),
                ("fileB", 2),
                ("fileC", 1.5),
                ("fileD", 3.001),
            ):
                event_clock[0] += elapsed
                await _emit_file_activity(handler, path)
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        assert list(groups[0].files["Reading"]) == ["fileA", "fileB", "fileC"]
        assert list(groups[1].files["Reading"]) == ["fileD"]


async def test_summary_card_mounts_immediately_updates_and_finalizes_after_idle(
    app, monkeypatch, event_clock, summary_timers
):
    monkeypatch.setattr(turn_module, "SUMMARY_GROUP_GRACE_SECONDS", 0.5)
    async with app.run_test(size=(100, 36)) as pilot:
        async with running_turn(app, pilot) as handler:
            for path in ("fileA", "fileB"):
                await _emit_file_activity(handler, path)
                await pilot.pause()
                groups = list(app.query(FileActivityCard))
                assert len(groups) == 1
                if path == "fileA":
                    first = groups[0]
                    assert list(first.files["Reading"]) == ["fileA"]
                else:
                    assert groups[0] is first
                    assert list(first.files["Reading"]) == ["fileA", "fileB"]
                assert not first.done
            assert summary_timers[0].stopped
            assert summary_timers[1].delay == 0.5

            summary_timers[1].fire()
            await pilot.pause()
            assert first.done

            await _emit_file_activity(handler, "fileC")
            await pilot.pause()
            groups = list(app.query(FileActivityCard))
            assert len(groups) == 2
            assert groups[0] is first
            assert list(groups[1].files["Reading"]) == ["fileC"]
            assert not groups[1].done

        assert groups[1].done


async def test_stale_summary_timer_cannot_finalize_replacement_card(event_turn):
    async with event_turn("read", size=(100, 36)) as (_pilot, turn):
        old = FileActivityCard("files:old")
        replacement = FileActivityCard("files:new")
        await turn.query_one(".events").mount(old, replacement)
        turn._summary_cards["files:Reading"] = replacement

        turn._finalize_summary("files:Reading", old)

        assert not replacement.done
        assert turn._summary_cards["files:Reading"] is replacement


async def test_parallel_read_callbacks_render_as_one_group(app):
    async with app.run_test(size=(100, 36)) as pilot:
        async with running_turn(app, pilot) as handler:

            async def read(run_id, path):
                await emit_tool_start(
                    handler, "read_file", run_id, filename=path
                )
                await _emit_file_activity(
                    handler, path, phase="start", message="Reading file"
                )

            await asyncio.gather(
                read("read-a", "README.md"),
                read("read-b", "tests/cli/test_config.py"),
                read("read-c", "pyproject.toml"),
            )
            await emit_tool_start(
                handler, "run_command", "command-after-reads", query="pwd"
            )
            await handler.on_tool_end("output", run_id="command-after-reads")
            # These delayed completions belong to the original Reading group and
            # must not create another Reading group after the command.
            for path in (
                "README.md",
                "tests/cli/test_config.py",
                "pyproject.toml",
            ):
                await _emit_file_activity(
                    handler, path, phase="end", message="File read"
                )
            await asyncio.gather(
                handler.on_tool_end("a", run_id="read-a"),
                handler.on_tool_end("b", run_id="read-b"),
                handler.on_tool_end("c", run_id="read-c"),
            )
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 1
        events = list(app.query_one(Turn).query_one(".events").children)
        assert [type(event) for event in events] == [
            FileActivityCard,
            RunCommandCard,
        ]
        assert list(groups[0].files["Reading"]) == [
            "README.md",
            str(Path("tests/cli/test_config.py")),
            "pyproject.toml",
        ]


async def test_file_failure_is_retained_on_its_activity_card(app, event_turn):
    async with event_turn("read it", size=(100, 36)) as (pilot, turn):
        handler = TextualEventHandler(app, turn)
        await emit_tool_start(
            handler, "read_file", "broken-read", filename="broken.txt"
        )
        await handler.on_tool_error(
            PermissionError("permission denied"), run_id="broken-read"
        )
        await pilot.pause()

        card = turn.query_one(FileActivityCard)
        assert card.outcomes[("Reading", "broken.txt")] == (
            "failed",
            "permission denied",
        )
        assert "permission denied" in str(
            card.query_one(".file-summary", Static).content
        )


async def test_unchanged_file_result_is_retained_on_activity_card(
    app, event_turn
):
    async with event_turn("edit it", size=(100, 36)) as (pilot, turn):
        handler = TextualEventHandler(app, turn)
        await emit_tool_start(
            handler, "edit_code", "unchanged-edit", filename="same.txt"
        )
        await handler.on_tool_end(
            "No changes made: content already matches",
            run_id="unchanged-edit",
        )
        await pilot.pause()

        card = turn.query_one(FileActivityCard)
        assert card.outcomes[("Editing", "same.txt")] == (
            "unchanged",
            "No changes made: content already matches",
        )


@pytest.mark.parametrize("outcome", ["failed", "unchanged"])
async def test_rich_edit_outcome_updates_existing_diff_row(
    app, event_turn, outcome
):
    async with event_turn("edit it", size=(100, 36)) as (pilot, turn):
        handler = TextualEventHandler(app, turn)
        await emit_tool_start(
            handler,
            "edit_code",
            "rich-edit",
            filename="same.txt",
            old_code="before",
            new_code="after",
        )
        if outcome == "failed":
            await handler.on_tool_error(
                RuntimeError("edit failed"), run_id="rich-edit"
            )
            expected = ("failed", "edit failed")
        else:
            message = "No changes made: content already matches"
            await handler.on_tool_end(message, run_id="rich-edit")
            expected = ("unchanged", message)
        await pilot.pause()

        cards = list(turn.query(EditCard))
        assert len(cards) == 1
        assert expected[1] in str(
            cards[0].query_one(".edit-outcome", Static).content
        )
