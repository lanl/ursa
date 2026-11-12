import json
import sqlite3

import pytest

from ursa.cli.tui.event_cards.card_store import CardStore


def seed_cards(store, count, *, start=1, turn_id=1):
    """Bulk fixture setup; intentionally bypass rendering and agent callbacks."""
    with store.transaction():
        store.connection.executemany(
            "INSERT INTO event_cards (id,session,turn_id,position,record_type,card_json) "
            "VALUES (?, ?, ?, ?, 'card', ?)",
            (
                (
                    i,
                    store.session,
                    turn_id,
                    i,
                    json.dumps({
                        "version": 1,
                        "type": "EventCard",
                        "state": {
                            "key": f"card:{i}",
                            "label": f"Event {i}",
                            "lines": [f"Item {i}"],
                            "details": [],
                            "done": True,
                        },
                    }),
                )
                for i in range(start, start + count)
            ),
        )
        store.connection.executemany(
            "INSERT INTO event_card_heads VALUES (?, ?, ?, ?, ?)",
            (
                (store.session, i, i, turn_id, f"card:{i}")
                for i in range(start, start + count)
            ),
        )


def test_monotonic_ids_reopen_and_thread_isolation(tmp_path):
    path = tmp_path / "session.db"
    store = CardStore(path, "one")
    with store.transaction():
        first = store.append(
            1, "card", {"type": "future", "content_ref": {"id": "pty:1"}}
        )
        second = store.append(
            1,
            "card",
            {"type": "future", "content_ref": {"id": "pty:2"}},
            record_type="commit",
        )
    assert second > first
    assert store.window(10)[0].position == first
    assert store.window(10)[0].id == second
    store.save_state(expanded=True)
    store.close()
    store = CardStore(path, "two")
    assert store.window(10) == []
    assert store.state() == {}
    with store.transaction():
        third = store.append(1, "card", {"type": "anything"})
    assert third > second
    store.close()
    store = CardStore(path, "one")
    assert store.window(10)[0].card["content_ref"] == {"id": "pty:2"}
    assert store.state() == {"expanded": True}
    store.close()


def test_atomic_commit_and_late_correlations(tmp_path):
    store = CardStore(tmp_path / "session.db")
    with store.transaction():
        position = store.append(1, "bundle", {"members": [1]})
        store.link(1, "run:abc", "bundle")
    with pytest.raises(RuntimeError), store.transaction():
        store.append(1, "bundle", {"members": [1, 2]}, record_type="commit")
        store.append(1, "second", {})
        raise RuntimeError("interrupted transaction")
    assert store.linked(1, "run:abc").card == {"members": [1]}
    assert store.window(10)[0].position == position
    assert len(store.window(10)) == 1
    assert store.find(1, "second") is None
    store.close()


def test_keyset_pagination_handles_gaps_revisions_and_other_sessions(tmp_path):
    store = CardStore(tmp_path / "session.db")
    with store.transaction():
        for i in range(250):
            store.append(1, str(i), {"i": i})
            for _ in range(3):
                store.append(1, str(i), {"i": i}, visible=False)
    expected = list(range(250))
    seen, cursor = [], 0
    while page := store.window(13, after=cursor):
        seen.extend(item.card["i"] for item in page)
        cursor = page[-1].position
    assert seen == expected
    seen, cursor = [], None
    while page := store.window(13, before=cursor):
        seen[:0] = [item.card["i"] for item in page]
        cursor = page[0].position
    assert seen == expected
    assert store.window(0) == []
    with pytest.raises(ValueError):
        store.window(10, before=10, after=1)
    store.close()


def test_window_query_work_is_bounded_at_both_ends_and_in_middle(tmp_path):
    store = CardStore(tmp_path / "session.db")
    seed_cards(store, 1000)

    def instructions(**cursor):
        steps = 0

        def count():
            nonlocal steps
            steps += 1
            return 0

        store.connection.set_progress_handler(count, 1)
        try:
            assert len(store.window(50, **cursor)) == 50
        finally:
            store.connection.set_progress_handler(None, 0)
        return steps

    small = instructions()
    seed_cards(store, 99000, start=1001)
    for cursor in ({}, {"after": 0}, {"after": 50000}, {"before": 50000}):
        assert instructions(**cursor) < small * 1.2
    # Every read uses the session/position index and rowid lookups. There is no
    # sort, skip-through-history OFFSET, or revision filtering on the read path.
    plan = store.connection.execute(
        "EXPLAIN QUERY PLAN SELECT e.card_json FROM event_card_heads h "
        "JOIN event_cards e ON e.id=h.card_id "
        "WHERE h.session=? AND h.position>? ORDER BY h.position LIMIT 50",
        (store.session, 50000),
    ).fetchall()
    assert all("SEARCH" in row[3] for row in plan)
    store.close()


def test_coexists_with_checkpoint_tables(tmp_path):
    path = tmp_path / "checkpointer.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE checkpoints (id TEXT PRIMARY KEY, value BLOB)"
        )
        conn.execute("INSERT INTO checkpoints VALUES ('one', X'1234')")
    store = CardStore(path)
    with store.transaction():
        store.append(1, "hello", {"version": 1})
    assert (
        store.connection.execute("SELECT value FROM checkpoints").fetchone()[0]
        == b"\x12\x34"
    )
    store.close()
