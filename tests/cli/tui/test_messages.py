import pytest
from textual.widgets import Button, Markdown, Static

from tests.cli._app_helpers import make_runtime, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.widgets import HierarchyMessageCard, MessagesScreen


async def test_message_screen_pages_without_retaining_full_bodies(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(make_runtime(tmp_path))
    body = "[message key=value] 🐺\n" + "Detailed content.\n" * 5000
    for index in range(125):
        await app.record_hierarchy_message(
            "root", (f"root/p{index}",), f"{index}: {body}"
        )
    reads = []
    read_body = app.message_store.message_body

    async def tracked_read(message_id):
        reads.append(message_id)
        return await read_body(message_id)

    monkeypatch.setattr(app.message_store, "message_body", tracked_read)
    async with app.run_test(size=(120, 40)) as pilot:
        await app._show_command("messages")
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert await wait_for(
            pilot, lambda: len(screen.query(HierarchyMessageCard)) == 50
        )
        cards = list(screen.query(HierarchyMessageCard))
        assert cards[0].message.content.startswith("75:")
        assert cards[-1].message.content.startswith("124:")
        assert not reads
        assert all(len(card.content) <= 241 for card in cards)
        assert all(not card.query_one(Markdown).source for card in cards)
        assert not hasattr(app, "_hierarchy_messages")
        assert app.screen.query_one("#messages-latest", Button).disabled
        assert str(
            app.screen.query_one("#messages-footer", Static).content
        ).startswith("Live ·")

        card = cards[-1]
        card.set_expanded(True)
        assert await wait_for(
            pilot, lambda: card.query_one(Markdown).source == f"124: {body}"
        )
        assert reads == [card.message.id]
        card.set_expanded(False)
        assert await wait_for(
            pilot, lambda: not card.query_one(Markdown).source
        )

        await screen.action_older()
        cards = list(screen.query(HierarchyMessageCard))
        assert len(cards) == 50 and cards[0].message.content.startswith("25:")
        assert not screen.query_one("#messages-latest", Button).disabled
        # Traffic arriving during historical browsing must not move the page.
        await app.record_hierarchy_message(
            "root/p01", ("root",), "new live message"
        )
        await screen._sync_messages()
        assert (
            screen.query(HierarchyMessageCard)
            .first()
            .message.content.startswith("25:")
        )
        assert "newer available" in str(
            screen.query_one("#messages-footer", Static).content
        )
        await screen.action_scroll_home()
        assert (
            screen.query(HierarchyMessageCard)
            .first()
            .message.content.startswith("0:")
        )
        await screen.action_latest()
        assert len(screen.query(HierarchyMessageCard)) == 50
        assert (
            screen.query(HierarchyMessageCard).last().content
            == "new live message"
        )
        assert len(reads) == 1

        await pilot.pause()
        await app.record_hierarchy_message(
            "root", ("root/p01",), "next live message"
        )
        assert await wait_for(
            pilot,
            lambda: screen.query(HierarchyMessageCard).last().content
            == "next live message",
        )
        assert len(screen.query(HierarchyMessageCard)) == 50

        await pilot.press("escape")
        assert not app.query(HierarchyMessageCard)
        await app._show_command("messages")
        assert await wait_for(
            pilot, lambda: len(app.screen.query(HierarchyMessageCard)) == 50
        )
        assert (
            app.screen.query(HierarchyMessageCard).last().content
            == "next live message"
        )
        assert len(reads) == 1


async def test_message_log_pauses_on_scroll_or_expansion_and_keyboard_toggles(
    tmp_path,
):
    app = UrsaTextualApp(make_runtime(tmp_path))
    for index in range(16):
        await app.record_hierarchy_message(
            "root", ("root/peer",), f"message {index}\nfull detail"
        )

    async with app.run_test(size=(80, 14)) as pilot:
        await app._show_command("messages")
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert await wait_for(
            pilot, lambda: len(screen.query(HierarchyMessageCard)) == 16
        )
        body = screen.query_one("#messages-body")
        assert await wait_for(pilot, lambda: body.scroll_y == body.max_scroll_y)

        # A focused card is usable without a mouse and reading it pauses live
        # follow before the body grows or new traffic arrives.
        card = screen.query(HierarchyMessageCard).last()
        assert card.can_focus
        card.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: card.expanded)
        assert not screen._live
        assert not screen.query_one("#messages-latest", Button).disabled
        await pilot.press("space")
        assert await wait_for(pilot, lambda: not card.expanded)

        # A plain upward scroll pauses the latest page as well. Incoming
        # traffic remains in SQLite until the reader deliberately resumes.
        body.scroll_end(animate=False, immediate=True)
        screen._live = True
        body.action_scroll_up()
        await pilot.pause(0.25)
        await screen._sync_messages()
        paused_at = body.scroll_y
        assert not screen._live
        assert not screen.query_one("#messages-latest", Button).disabled
        assert str(
            screen.query_one("#messages-footer", Static).content
        ).startswith("Paused ·")
        await app.record_hierarchy_message(
            "root/peer", ("root",), "new traffic"
        )
        await screen._sync_messages()
        assert body.scroll_y == pytest.approx(paused_at)
        assert all(
            card.content != "new traffic"
            for card in screen.query(HierarchyMessageCard)
        )

        # End is the keyboard equivalent of Latest and resumes follow.
        await pilot.press("end")
        assert await wait_for(
            pilot,
            lambda: screen._live
            and screen.query(HierarchyMessageCard).last().content
            == "new traffic",
        )
        assert body.scroll_y == body.max_scroll_y
