import asyncio

from textual.containers import VerticalScroll
from textual.widgets import Static

import ursa.cli.event_handler as event_handler_module
import ursa.cli.turn as turn_module
from tests.cli._app_fakes import FakeHITL, emit_event
from ursa.cli.app import UrsaTextualApp
from ursa.cli.event_cards import FileActivityCard, RunCommandCard
from ursa.cli.event_handler import TextualEventHandler
from ursa.cli.turn import Turn
from ursa.util.events import DEFAULT_EVENT_NAME


async def test_files_are_grouped_by_read_write_and_edit_operations(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        events = [
            {"tool": "read_file", "path": "src/read.py", "message": "Reading"},
            {
                "tool": "read_file",
                "path": "src/other.py",
                "message": "Reading",
            },
            {
                "tool": "write_code",
                "path": "src/new.py",
                "code": "new file\n",
                "message": "Writing",
            },
            {
                "tool": "edit_code",
                "path": "src/edit.py",
                "old_code": "old\n",
                "new_code": "new\nmore\n",
                "message": "Editing",
            },
            {
                "tool": "edit_code",
                "path": str(tmp_path / "src" / "edit.py"),
                "message": "Editing",
            },
        ]
        for event in events:
            await handler.on_custom_event(DEFAULT_EVENT_NAME, event)
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        file_groups = list(app.query(FileActivityCard))
        assert len(file_groups) == 2
        reading, editing = file_groups
        assert reading.files == {
            "Reading": {
                "src/read.py": (None, None),
                "src/other.py": (None, None),
            },
            "Editing": {},
        }
        assert editing.files == {
            "Reading": {},
            "Editing": {
                "src/new.py": (1, 0),
                "src/edit.py": (2, 1),
            },
        }
        reading_summary = reading.query_one(".file-summary", Static).content
        editing_summary = editing.query_one(".file-summary", Static).content
        assert all(
            path in reading_summary.plain
            for path in ("src/read.py", "src/other.py")
        )
        assert all(
            path in editing_summary.plain
            for path in ("src/new.py", "src/edit.py")
        )
        assert "+1 -0" in editing_summary.plain
        assert "+2 -1" in editing_summary.plain


async def test_event_summary_groups_follow_activity_order(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        for path in ("fileA", "fileB", "fileC"):
            await emit_event(
                handler,
                {"tool": "read_file", "path": path, "message": "Reading"},
            )
        await emit_event(
            handler,
            {
                "tool": "edit_code",
                "path": "fileB",
                "message": "Editing",
                "additions": 2,
                "deletions": 1,
            },
        )
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="ordered-command",
            inputs={"query": "pwd"},
        )
        await handler.on_tool_end("", run_id="ordered-command")
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await pilot.pause()
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


async def test_activity_kinds_keep_independent_cards_open(
    tmp_path, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(turn_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(event_handler_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        turn_module,
        "SUMMARY_GROUP_GRACE_SECONDS",
        1.0,
    )
    hitl = FakeHITL(tmp_path)
    reading_started = asyncio.Event()
    switch_to_editing = asyncio.Event()
    editing_started = asyncio.Event()
    return_to_reading = asyncio.Event()
    reading_updated = asyncio.Event()
    finish_agent = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        reading_started.set()
        await switch_to_editing.wait()
        await emit_event(
            handler,
            {
                "tool": "edit_code",
                "path": "fileB",
                "message": "Editing",
                "additions": 2,
                "deletions": 1,
            },
        )
        editing_started.set()
        await return_to_reading.wait()
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileC", "message": "Reading"},
        )
        reading_updated.set()
        await finish_agent.wait()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await reading_started.wait()
        await pilot.pause()
        reading = list(app.query(FileActivityCard))[0]

        switch_to_editing.set()
        await editing_started.wait()
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        editing = groups[1]
        assert not reading.done
        assert not editing.done

        await asyncio.sleep(0.6)
        return_to_reading.set()
        await reading_updated.wait()
        await pilot.pause()
        assert list(reading.files["Reading"]) == ["fileA", "fileC"]
        assert len(app.query(FileActivityCard)) == 2
        assert not reading.done
        assert not editing.done

        await asyncio.sleep(0.45)
        await pilot.pause()
        assert not reading.done
        assert editing.done

        await asyncio.sleep(0.6)
        await pilot.pause()
        assert reading.done

        finish_agent.set()
        await pilot.pause()


async def test_event_group_grace_period_resets_after_each_update(
    tmp_path, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(turn_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(event_handler_module, "monotonic", lambda: clock[0])
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        clock[0] += 2.0
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileB", "message": "Reading"},
        )
        # This is after the original deadline but before the deadline renewed
        # by fileB. A failure to reset the grace period creates a new card.
        clock[0] += 1.5
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileC", "message": "Reading"},
        )
        clock[0] += 3.001
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileD", "message": "Reading"},
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        assert list(groups[0].files["Reading"]) == [
            "fileA",
            "fileB",
            "fileC",
        ]
        assert list(groups[1].files["Reading"]) == ["fileD"]


async def test_summary_card_mounts_immediately_updates_and_finalizes_after_idle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        turn_module,
        "SUMMARY_GROUP_GRACE_SECONDS",
        0.5,
    )
    hitl = FakeHITL(tmp_path)
    first_emitted = asyncio.Event()
    emit_second = asyncio.Event()
    second_emitted = asyncio.Event()
    emit_third = asyncio.Event()
    third_emitted = asyncio.Event()
    finish_agent = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        first_emitted.set()
        await emit_second.wait()
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileB", "message": "Reading"},
        )
        second_emitted.set()
        await emit_third.wait()
        await emit_event(
            handler,
            {"tool": "read_file", "path": "fileC", "message": "Reading"},
        )
        third_emitted.set()
        await finish_agent.wait()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await first_emitted.wait()
        await pilot.pause()

        groups = list(app.query(FileActivityCard))
        assert len(groups) == 1
        first = groups[0]
        assert list(first.files["Reading"]) == ["fileA"]
        assert not first.done

        # Stay comfortably inside the 500 ms production grace period even on
        # a busy test runner.
        await asyncio.sleep(0.1)
        emit_second.set()
        await second_emitted.wait()
        await pilot.pause()
        assert list(first.files["Reading"]) == ["fileA", "fileB"]
        assert not first.done

        await asyncio.sleep(0.3)
        await pilot.pause()
        assert not first.done
        await asyncio.sleep(0.25)
        await pilot.pause()
        assert first.done

        emit_third.set()
        await third_emitted.wait()
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        assert groups[0] is first
        assert list(groups[1].files["Reading"]) == ["fileC"]
        assert not groups[1].done

        finish_agent.set()
        await pilot.pause()
        assert groups[1].done


async def test_stale_summary_timer_cannot_finalize_replacement_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)):
        turn = Turn("read", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        old = FileActivityCard("files:old")
        replacement = FileActivityCard("files:new")
        await turn.query_one(".events").mount(old, replacement)
        turn._summary_cards["files:Reading"] = replacement

        turn._finalize_summary("files:Reading", old)

        assert not replacement.done
        assert turn._summary_cards["files:Reading"] is replacement


async def test_parallel_read_callbacks_render_as_one_group(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]

        async def read(run_id, path):
            await handler.on_tool_start(
                {"name": "read_file"},
                "",
                run_id=run_id,
                inputs={"filename": path},
            )
            await emit_event(
                handler,
                {
                    "tool": "read_file",
                    "phase": "start",
                    "path": path,
                    "message": "Reading file",
                },
            )

        await asyncio.gather(
            read("read-a", "README.md"),
            read("read-b", "tests/cli/test_config.py"),
            read("read-c", "pyproject.toml"),
        )
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="command-after-reads",
            inputs={"query": "pwd"},
        )
        await handler.on_tool_end("output", run_id="command-after-reads")
        # These delayed completions belong to the original Reading group and
        # must not create another Reading group after the command.
        for path in (
            "README.md",
            "tests/cli/test_config.py",
            "pyproject.toml",
        ):
            await emit_event(
                handler,
                {
                    "tool": "read_file",
                    "phase": "end",
                    "path": path,
                    "message": "File read",
                },
            )
        await asyncio.gather(
            handler.on_tool_end("a", run_id="read-a"),
            handler.on_tool_end("b", run_id="read-b"),
            handler.on_tool_end("c", run_id="read-c"),
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "e", "a", "d", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 1
        events = list(app.query_one(Turn).query_one(".events").children)
        assert [type(event) for event in events] == [
            FileActivityCard,
            RunCommandCard,
        ]
        assert list(groups[0].files["Reading"]) == [
            "README.md",
            "tests/cli/test_config.py",
            "pyproject.toml",
        ]


async def test_file_failure_is_retained_on_its_activity_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("read it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "read_file"},
            "",
            run_id="broken-read",
            inputs={"filename": "broken.txt"},
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


async def test_unchanged_file_result_is_retained_on_activity_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("edit it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        handler = TextualEventHandler(app, turn)
        await handler.on_tool_start(
            {"name": "edit_code"},
            "",
            run_id="unchanged-edit",
            inputs={"filename": "same.txt"},
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
