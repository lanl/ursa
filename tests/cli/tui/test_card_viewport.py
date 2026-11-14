import asyncio

import pytest
from textual.containers import VerticalScroll
from textual.events import MouseScrollUp

from tests.cli._app_fakes import FakeHITL
from tests.cli.tui.test_card_store import seed_cards
from tests.tui.utils import eventually
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import EventCard, RunCommandCard
from ursa.cli.tui.event_cards.card_store import CardStore
from ursa.cli.tui.event_cards.card_stream import CardStream
from ursa.cli.tui.widgets import MessageCard, PromptArea


@pytest.mark.parametrize("count", [1000, 100000])
async def test_widget_window_and_layout_are_bounded_on_resume(tmp_path, count):
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = tmp_path / "session.db"
    store = CardStore(hitl.card_database_path)
    seed_cards(store, count)
    with store.transaction():
        for table in ("event_cards", "event_card_heads"):
            store.connection.execute(
                f"UPDATE {table} SET turn_id=(position-1)/10+1"
            )
    store.close()
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        viewport = app.card_viewport
        assert len(viewport.records) <= 5 * app.size.height
        assert len(app.query(EventCard)) == len(viewport.records)
        assert viewport.records[-1].position == count
        # DOM size is independent of the number of durable rows.
        assert len(app.query("*")) < 600
        await viewport.home()
        await pilot.pause()
        assert viewport.records[0].position == 1
        await viewport.page(1)
        await pilot.pause()
        assert viewport.records[0].position > 1
        assert [card.key for card in app.query(EventCard)] == [
            record.key for record in viewport.records
        ]
        await viewport.end()
        await pilot.pause()
        assert viewport.records[-1].position == count
        assert len(app.query(EventCard)) <= 5 * app.size.height
        await pilot.resize_terminal(60, 16)
        await eventually(
            pilot, lambda: len(viewport.records) <= 5 * app.size.height
        )
        assert len(app.query(EventCard)) == len(viewport.records)


async def test_resume_restores_messages_tokens_and_expansion(tmp_path):
    path = tmp_path / "session.db"
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = path
    app = UrsaTextualApp(hitl)
    async with app.run_test() as pilot:
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.total_tokens = 123
        app.action_toggle_card_details()
    resumed = FakeHITL(tmp_path)
    resumed.card_database_path = path
    app = UrsaTextualApp(resumed)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert [m.content for m in app.query(MessageCard)] == ["hi", "Finished"]
        assert app.total_tokens == 123
        assert app.card_details_expanded
        assert app.query_one(PromptArea).prompt_history == ["hi"]
        await pilot.press("n", "e", "w", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert [m.content for m in app.query(MessageCard)] == [
            "hi",
            "Finished",
            "new",
            "Finished",
        ]


async def test_reopen_incomplete_command_is_interrupted_without_replaying(
    tmp_path,
):
    path = tmp_path / "session.db"
    store = CardStore(path)
    stream = CardStream(store, tmp_path, "run it")
    await stream.start()
    await stream.event({
        "tool": "run_command",
        "tool_call_id": "running",
        "query": "sleep 100",
        "phase": "start",
    })
    store.close()
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = path
    app = UrsaTextualApp(hitl)
    async with app.run_test() as pilot:
        await pilot.pause()
        card = app.query_one(RunCommandCard)
        assert card.completed and card.execution_failed
        assert "Interrupted" in card._full_output
        assert hitl.calls == []


async def test_late_updates_while_reading_scrollback_keep_position(tmp_path):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)) as pilot:
        await asyncio.to_thread(seed_cards, app.card_store, 500)
        viewport = app.card_viewport
        await viewport.refresh()
        await pilot.pause()
        await viewport.home()
        await pilot.pause()
        conversation = app.query_one("#conversation", VerticalScroll)
        conversation.scroll_to(y=25, animate=False, immediate=True)
        await pilot.pause()
        anchor = viewport._anchor()
        assert anchor is not None
        position, y = anchor
        widget = viewport.widgets[position]
        stream = CardStream(app.card_store, tmp_path, "new activity")
        await stream.start()
        app.current_turn = stream
        for i in range(50):
            await stream.event({
                "tool": "custom",
                "_run_id": str(i),
                "phase": "end",
                "result": i,
            })
        await viewport.refresh()
        await pilot.pause()
        assert widget.is_mounted
        assert widget.region.y == y
        assert not viewport.follow_tail
        assert viewport.has_newer
        await viewport.end()
        await pilot.pause()
        assert viewport.follow_tail
        assert viewport.records[-1].turn_id == stream.turn_id


async def test_long_single_turn_does_not_accumulate_widgets_or_command_models(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(80, 20)) as pilot:
        stream = CardStream(app.card_store, tmp_path, "long run")
        await stream.start()
        app.current_turn = stream
        for i in range(150):
            await stream.event({
                "tool": "run_command",
                "tool_call_id": str(i),
                "phase": "end",
                "result": "output",
            })
            if i % 30 == 0:
                await app.card_viewport.refresh()
                await pilot.pause()
                assert len(app.query(EventCard)) <= 5 * app.size.height
                assert not stream.commands
        await stream.finish("done")
        await app.card_viewport.refresh()
        await pilot.pause()
        assert len(app.query(EventCard)) < 150
        assert len(app.card_store.window(1000)) == 153


async def test_individual_expansion_and_reading_position_survive_resume(
    tmp_path,
):
    path = tmp_path / "session.db"
    store = CardStore(path)
    seed_cards(store, 500)
    store.close()
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = path
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)) as pilot:
        await app.card_viewport.home()
        await pilot.pause()
        record = app.card_viewport.records[0]
        card = app.card_viewport.widgets[record.position]
        card.set_expanded(True)
        await app.card_viewport.save_expanded(record.position, True)
        await app.card_viewport.end()
        await app.card_viewport.home()
        await pilot.pause()
        assert app.card_viewport.widgets[record.position].expanded
        app.card_viewport.remember_position()
        position = app.card_viewport.resume_position
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = path
    resumed = UrsaTextualApp(hitl)
    async with resumed.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        assert not resumed.card_viewport.follow_tail
        assert resumed.card_viewport.records[0].position == position
        assert resumed.card_viewport.widgets[record.position].expanded
        # Global expansion overrides per-card preferences even after eviction.
        resumed.action_toggle_card_details()
        resumed.action_toggle_card_details()
        await resumed.card_viewport.end()
        await resumed.card_viewport.home()
        await pilot.pause()
        assert not resumed.card_viewport.widgets[record.position].expanded


async def test_clear_view_keeps_durable_history_and_resume_boundary(tmp_path):
    path = tmp_path / "session.db"
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = path
    app = UrsaTextualApp(hitl)
    async with app.run_test() as pilot:
        await pilot.press("a", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        count = len(app.card_store.window(100))
        await app.action_clear_conversation()
        assert len(app.card_store.window(100)) == count
        assert len(app.query(MessageCard)) == 0
        await pilot.press("b", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    hitl = FakeHITL(tmp_path)
    hitl.card_database_path = path
    app = UrsaTextualApp(hitl)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert [m.content for m in app.query(MessageCard)] == ["b", "Finished"]


async def test_mouse_can_page_when_already_at_window_edge(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 36)) as pilot:
        await asyncio.to_thread(seed_cards, app.card_store, 500)
        viewport = app.card_viewport
        await viewport.refresh()
        await pilot.pause()
        conversation = viewport.conversation
        viewport.follow_tail = False
        conversation.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        first = viewport.records[0].position
        # Scroll at y=0 cannot change the local scrollbar; it must still fetch.
        conversation.post_message(
            MouseScrollUp(
                conversation,
                x=5,
                y=5,
                delta_x=0,
                delta_y=-1,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        await eventually(pilot, lambda: viewport.records[0].position < first)


async def test_input_messages_are_processed_while_window_mount_is_waiting(
    tmp_path, monkeypatch
):
    from textual.containers import Vertical
    from textual.events import Key

    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(100, 36)) as pilot:
        await asyncio.to_thread(seed_cards, app.card_store, 500)
        entered = asyncio.Event()
        release = asyncio.Event()
        processed = asyncio.Event()
        original = Vertical.mount

        async def delayed_mount(container, *widgets, **kwargs):
            if any(isinstance(widget, EventCard) for widget in widgets):
                entered.set()
                await release.wait()
            return await original(container, *widgets, **kwargs)

        monkeypatch.setattr(Vertical, "mount", delayed_mount)
        app.card_viewport.request_refresh()
        await asyncio.wait_for(entered.wait(), 5)
        try:
            prompt = app.query_one(PromptArea)
            prompt.post_message(Key("x", "x"))
            # A viewport waiting on widget work must not hold the app's
            # message queue (the queue also handles input and global actions).
            app.call_later(processed.set)
            await asyncio.wait_for(processed.wait(), 1)
            assert prompt.text == "x"
        finally:
            release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
