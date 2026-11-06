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
        assert "Following live messages" in str(
            app.screen.query_one("#messages-footer", Static).content
        )

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
        assert "newer messages available" in str(
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
