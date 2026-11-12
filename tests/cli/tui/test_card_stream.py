import json

import pytest
from langchain_core.messages import ToolMessage

from ursa.cli.tui.event_cards import (
    AgentEventCard,
    ArtifactCard,
    EditCard,
    EventCard,
    ExceptionCard,
    FileActivityCard,
    PlanCard,
    RunCommandCard,
    SearchEventCard,
    ToolCallCard,
)
from ursa.cli.tui.event_cards.card_codec import decode_card, encode_card
from ursa.cli.tui.event_cards.card_store import CardStore
from ursa.cli.tui.event_cards.card_stream import BUNDLE_LIMIT, CardStream


@pytest.fixture
def store(tmp_path):
    result = CardStore(tmp_path / "session.db")
    yield result
    result.close()


async def test_cards_are_durable_before_callback_returns_and_commit_defines_bundle(
    store, tmp_path
):
    stream = CardStream(store, tmp_path, "read files")
    await stream.start()
    for path in ("one", "two", "three"):
        await stream.event({"tool": "read_file", "path": path})
        # These are normalized cards, not the raw event payload.
        rows = store.connection.execute(
            "SELECT record_type,card_json FROM event_cards ORDER BY id"
        ).fetchall()
        assert json.loads(rows[-2][1])["type"] == "FileActivityCard"
        assert json.loads(rows[-1][1])["state"]["files"]["Reading"][path] == [
            None,
            None,
        ]
    await stream.finish("done")
    record = next(
        row
        for row in store.window(10)
        if row.card["type"] == "FileActivityCard"
    )
    assert record.card["state"]["done"]
    members = record.card["bundle"]["members"]
    assert len(members) == 3
    assert members == sorted(set(members))
    assert all(
        store.connection.execute(
            "SELECT 1 FROM event_cards WHERE id=?", (member,)
        ).fetchone()
        for member in members
    )
    assert stream.groups == {}


async def test_endless_magnetic_stream_is_split_into_bounded_bundles(
    store, tmp_path
):
    stream = CardStream(store, tmp_path, "read forever")
    await stream.start()
    for i in range(BUNDLE_LIMIT * 4 + 1):
        await stream.event({"tool": "read_file", "path": str(i)})
        assert len(stream.groups) == 1
        assert len(stream.groups["files:Reading"].members) <= BUNDLE_LIMIT
    await stream.finish("done")
    cards = [
        r for r in store.window(20) if r.card["type"] == "FileActivityCard"
    ]
    assert len(cards) == 5
    assert (
        sum(len(r.card["state"]["files"]["Reading"]) for r in cards)
        == BUNDLE_LIMIT * 4 + 1
    )


async def test_late_tool_and_edit_results_use_durable_identity(store, tmp_path):
    stream = CardStream(store, tmp_path, "many tools")
    await stream.start()
    for i in range(150):
        await stream.event({
            "tool": "my_tool",
            "_run_id": str(i),
            "phase": "start",
            "x": i,
        })
    await stream.event({
        "tool": "my_tool",
        "_run_id": "0",
        "phase": "end",
        "result": "old result",
    })
    assert (
        store.linked(stream.turn_id, "tool:0").card["state"]["output"]
        == "old result"
    )
    assert all(
        r.key != store.linked(stream.turn_id, "tool:0").key
        for r in store.window(10)
    )
    for run in ("a", "b"):
        await stream.event({
            "tool": "edit_code",
            "_run_id": run,
            "path": "same.py",
            "old_code": "before",
            "new_code": run,
        })
    await stream.event({
        "tool": "edit_code",
        "_run_id": "a",
        "path": "same.py",
        "phase": "error",
        "error": "failed a",
    })
    first = store.linked(stream.turn_id, "edit-run:a")
    second = store.linked(stream.turn_id, "edit-run:b")
    assert first.card["state"]["outcome"] == ["failed", "failed a"]
    assert second.card["state"]["outcome"] is None


async def test_command_commits_and_live_reference_do_not_copy_buffer(
    store, tmp_path
):
    stream = CardStream(store, tmp_path, "terminal")
    await stream.start()
    reference = {"kind": "terminal", "id": "terminal-123"}
    await stream.event({
        "tool": "run_command",
        "tool_call_id": "x",
        "query": "echo hi",
        "phase": "start",
        "content_ref": reference,
    })
    await stream.event({
        "tool": "run_command",
        "tool_call_id": "x",
        "phase": "end",
        "result": "large live buffer" * 1000,
        "content_ref": reference,
    })
    record = store.find(stream.turn_id, "command:x")
    assert record.card["state"]["content_ref"] == reference
    assert record.card["state"]["_full_output"] == ""
    assert record.card["bundle"]["position"] == record.position
    assert record.card["bundle"]["through"] < record.id
    assert stream.commands == {}


async def test_command_memory_contains_only_active_commands(store, tmp_path):
    stream = CardStream(store, tmp_path, "commands")
    await stream.start()
    for i in range(150):
        await stream.event({
            "tool": "run_command",
            "tool_call_id": str(i),
            "phase": "start",
            "query": "true",
        })
        await stream.event({
            "tool": "run_command",
            "tool_call_id": str(i),
            "phase": "end",
            "result": "",
        })
        assert stream.commands == {}
    assert len(store.window(200)) == 151


def test_all_card_types_round_trip_without_widget_runtime_state():
    file = FileActivityCard("files")
    file.add_file("Reading", "one")
    file.record_outcome("Reading", "one", "failed", "denied")
    edit = EditCard("one.py", "old", "new")
    edit.set_outcome("unchanged", "same content")
    plan = PlanCard("plan", 2)
    plan.set_plan(["do it"])
    tool = ToolCallCard("tool", "name", {"a": 1})
    tool.complete(
        ToolMessage(
            content="hello",
            tool_call_id="x",
            artifact={"structured_content": {"x": 2}},
        )
    )
    command = RunCommandCard("command", "echo hi")
    command.update_event({"stage": "safety_check", "safe": True})
    command.complete("hi", execution_confirmed=True)
    cards = [
        file,
        edit,
        plan,
        tool,
        command,
        EventCard("base", "label"),
        AgentEventCard("agent", "LammpsAgent"),
        SearchEventCard("search", "run_web_search"),
        ArtifactCard("artifact", [{"kind": "text", "content": "a"}]),
        ExceptionCard("error", ValueError("broken"), "traceback text"),
    ]
    for card in cards:
        data = json.loads(json.dumps(encode_card(card)))
        clone = decode_card(data)
        assert type(clone) is type(card)
        assert json.loads(json.dumps(encode_card(clone))) == data
        assert not clone.is_mounted
    unknown = decode_card({"version": 100, "type": "future"})
    assert "Unsupported" in unknown.label


async def test_read_after_edit_is_not_suppressed(store, tmp_path):
    stream = CardStream(store, tmp_path, "edit then check")
    await stream.start()
    await stream.event({
        "tool": "edit_code",
        "path": "same.py",
        "old_code": "a",
        "new_code": "b",
    })
    await stream.event({"tool": "read_file", "path": "same.py"})
    cards = store.window(10)
    assert [r.card["type"] for r in cards] == [
        "message",
        "EditCard",
        "FileActivityCard",
    ]


async def test_late_file_failure_preserves_committed_members(store, tmp_path):
    stream = CardStream(store, tmp_path, "read")
    await stream.start()
    await stream.event({
        "tool": "read_file",
        "path": "one",
        "_run_id": "first",
        "_received_at": 1.0,
    })
    await stream.event({
        "tool": "read_file",
        "path": "two",
        "_received_at": 100.0,
    })
    before = store.linked(stream.turn_id, "file-run:first")
    assert "bundle" in before.card
    await stream.event({
        "tool": "read_file",
        "path": "one",
        "_run_id": "first",
        "phase": "error",
        "error": "failed read",
    })
    after = store.linked(stream.turn_id, "file-run:first")
    assert after.card["bundle"] == before.card["bundle"]
    assert after.position == before.position
    assert after.card["state"]["outcomes"][0][-1] == "failed read"


async def test_finish_closes_outstanding_tools_and_commands(store, tmp_path):
    stream = CardStream(store, tmp_path, "unfinished tools")
    await stream.start()
    await stream.event({
        "tool": "run_command",
        "tool_call_id": "x",
        "phase": "start",
        "query": "sleep 100",
    })
    await stream.event({"tool": "other_tool", "_run_id": "y", "phase": "start"})
    await stream.finish("failed", succeeded=False)
    command = store.find(stream.turn_id, "command:x")
    tool = store.linked(stream.turn_id, "tool:y")
    assert command.card["state"]["completed"]
    assert command.card["state"]["execution_failed"]
    assert "bundle" in command.card
    assert tool.card["state"]["completed"]
    assert tool.card["state"]["failed"]
    assert not stream.commands and not stream.pending_tools
