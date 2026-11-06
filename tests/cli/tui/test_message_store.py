import asyncio
import json
import sqlite3

from ursa.cli.tui.message_store import MessageStore


async def test_messages_persist_exact_bodies_but_list_only_bounded_previews(
    tmp_path,
):
    path = tmp_path / "db" / "checkpointer.db"
    store = MessageStore(path, "game")
    content = "[SETUP role=Villager] 🐺\n" + "long message\n" * 20_000
    message_id = await store.append("root/p01", ("root", "root/p02"), content)
    # A fresh store can retrieve the exact content without an in-memory log.
    reopened = MessageStore(path, "game")
    rows = await reopened.messages()
    assert len(rows) == 1
    assert len(rows[0].content) <= 241
    assert "\n" not in rows[0].content
    assert rows[0].recipients == ("root", "root/p02")
    assert await reopened.message_body(message_id) == content
    assert await MessageStore(path, "other-thread").messages() == []
    assert not any(
        isinstance(value, (dict, list)) for value in vars(store).values()
    )


async def test_message_pagination_has_no_gaps_or_duplicates(tmp_path):
    store = MessageStore(tmp_path / "checkpointer.db", "thread")
    for index in range(125):
        await store.append("root", ("root/child",), str(index))
    latest = await store.messages()
    assert [int(row.content) for row in latest] == list(range(75, 125))
    earlier = await store.messages(before=latest[0].id)
    assert [int(row.content) for row in earlier] == list(range(25, 75))
    first = await store.messages(before=earlier[0].id)
    assert [int(row.content) for row in first] == list(range(25))
    assert await store.messages(after=earlier[-1].id) == latest


async def test_coordinated_actions_keep_input_on_completion_and_isolate_turns(
    tmp_path,
):
    store = MessageStore(tmp_path / "checkpointer.db", "thread")
    task = {"target": ["root/a", "root/b"], "message": "exact message\n" * 5000}
    await store.record_action("turn-a", "send", "send_message", task)
    await store.record_action(
        "turn-b", "send", "send_message", {"target": "other"}
    )
    await store.record_action(
        "turn-a", "wait", "wait_for", {"target": "root/a"}
    )
    result = '[{"sender":"root/a", "content":"role=Villager"}]' * 5000
    await store.record_action(
        "turn-a", "wait", "wait_for", {}, output=result, completed=True
    )
    await store.record_action(
        "turn-a",
        "send",
        "send_message",
        {},
        output="error",
        completed=True,
        failed=True,
    )
    counts, recent = await store.action_summary("turn-a")
    assert counts == {"total": 2, "running": 0, "failed": 1}
    assert len(recent) == 2
    assert all(len(row["preview"]) <= 241 for row in recent)
    assert all("input" not in row and "output" not in row for row in recent)
    actions = await store.actions("turn-a")
    assert all(
        "input" not in action and "output" not in action for action in actions
    )
    assert "message" not in json.loads(actions[1]["input_preview"])
    assert (await store.action_body("turn-a", actions[0]["id"]))[1] == result
    assert (await store.action_body("turn-a", actions[1]["id"]))[0] == task
    assert len(await store.actions("turn-b")) == 1


async def test_checkpoint_writer_lock_does_not_block_event_loop(tmp_path):
    store = MessageStore(tmp_path / "checkpointer.db", "thread")
    await store.append("root", ("root/a",), "first")
    db = sqlite3.connect(store.path)
    try:
        db.execute("BEGIN IMMEDIATE")
        write = asyncio.create_task(store.append("root", ("root/a",), "second"))
        await asyncio.sleep(0.05)
        assert not write.done()
        # Releasing a competing checkpoint transaction lets the background
        # writer finish. A synchronous UI write would block this coroutine.
        db.rollback()
        await asyncio.wait_for(write, timeout=2)
    finally:
        db.close()
    assert len(await store.messages()) == 2
