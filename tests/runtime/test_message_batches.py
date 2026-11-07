from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from tests.runtime.fixtures import runtime_request
from ursa.runtime.agent_runtime import _BufferedAgentMessage


class LifecycleCallback:
    def __init__(self, events):
        self.events = events

    async def on_agent_message_start(self, path):
        self.events.append(("start", path))

    async def on_agent_message_end(self, path, *, succeeded):
        self.events.append(("end", path, succeeded))


@pytest.mark.parametrize("reset_query", [False, True])
async def test_process_delivers_batch_in_one_turn_with_separate_envelopes(
    reset_query,
    process_runner,
):
    runner = process_runner(
        "root/worker", result="processed", reset_query=reset_query
    )
    agent = runner.agent
    bodies = [
        "First question\n[brackets]",
        'Second question: "quoted"',
        "New phase",
    ]
    senders = ["root/first", "root/second", "root"]
    result = await runner.ainvoke(
        bodies[0],
        config={
            "metadata": {
                "ursa_message_sender": senders[0],
                "ursa_message_recipient": "root/worker",
            },
            "ursa_additional_messages": [
                {
                    "sender": sender,
                    "recipient": "root/worker",
                    "content": body,
                }
                for sender, body in zip(senders[1:], bodies[1:], strict=True)
            ],
        },
    )
    assert result == "processed"
    assert "ursa_additional_messages" not in agent.configs[0]
    assert "ursa_additional_messages" not in agent.configs[0]["metadata"]
    assert len(agent.seen) == 1
    assert len(agent.formatted) == 1
    if reset_query:
        entries = json.loads(agent.seen[0]["task"].split("\n", 1)[1])
        assert [entry["content"] for entry in entries] == bodies
        assert [entry["sender"] for entry in entries] == senders
    messages = agent.seen[0]["messages"]
    assert len(messages) == 7
    for index, (sender, body) in enumerate(zip(senders, bodies, strict=True)):
        envelope, message = messages[1 + index * 2 : 3 + index * 2]
        assert isinstance(envelope, SystemMessage)
        assert f"Sender: {sender}" in envelope.content
        assert "Recipient: root/worker" in envelope.content
        assert isinstance(message, HumanMessage)
        assert message.content == body


@pytest.fixture
async def runtime(runtime_factory, monkeypatch):
    runtime, manager = runtime_factory()
    monkeypatch.setattr(manager, "start_agent", AsyncMock())
    await runtime._ensure_registered("chat")
    root = runtime.agents["chat"].runtime_id
    for name in ("first", "second"):
        await runtime._spawn_runtime_subagent(root, name)
    return runtime, root


async def queue_peer_messages(runtime, root, callbacks=()):
    expected = []
    for sender, content in (
        ("first", "question one"),
        ("second", "question two"),
        ("first", "newer instruction"),
    ):
        sender_id = runtime._hierarchies[root][f"root/{sender}"]
        result = await runtime_request(
            runtime,
            sender_id,
            "invoke",
            callbacks=callbacks,
            target="root",
            message=content,
        )
        assert result == "Message queued for root."
        expected.append({
            "sender": f"root/{sender}",
            "recipient": "root",
            "content": content,
        })
    return expected


async def test_busy_recipient_runs_one_batch_with_one_activity(
    runtime, monkeypatch
):
    runtime, root = runtime
    runtime._active_root_id = root
    invocations = []
    lifecycle = []

    async def invoke(agent_id, prompt, *, config, **kwargs):
        invocations.append((agent_id, prompt, config))
        return "done"

    monkeypatch.setattr(runtime._manager, "invoke", invoke)
    runtime._mark_working("root", 1)
    expected = await queue_peer_messages(
        runtime, root, (LifecycleCallback(lifecycle),)
    )
    assert len(runtime._buffered_agent_messages["root"]) == 3
    assert not invocations
    runtime._mark_working("root", -1)
    await asyncio.gather(*runtime._background_agent_tasks)
    assert len(invocations) == 1
    agent_id, prompt, config = invocations[0]
    assert agent_id == root
    assert prompt == expected[0]["content"]
    assert config["metadata"]["ursa_message_sender"] == expected[0]["sender"]
    assert config["ursa_additional_messages"] == expected[1:]
    assert lifecycle == [("start", "root"), ("end", "root", True)]
    assert runtime._buffered_agent_messages == {}
    assert runtime._working_counts == {}


async def test_wait_consumes_entire_buffer_once_without_replaying_it(runtime):
    runtime, root = runtime
    runtime._mark_working("root", 1)
    expected = await queue_peer_messages(runtime, root)
    result = await runtime._wait_for_runtime_agent(root, ["first", "second"], 0)
    assert result.startswith("Received 3 messages while waiting.")
    assert json.loads(result.split("\n", 1)[1]) == expected
    assert runtime._buffered_agent_messages == {}
    runtime._mark_working("root", -1)
    assert not runtime._background_agent_tasks
    again = await runtime._wait_for_runtime_agent(root, ["first", "second"], 0)
    assert "Received" not in again


async def test_stopping_recipient_does_not_flush_a_new_batch(runtime):
    runtime, root = runtime
    runtime._mark_working("root", 1)
    await queue_peer_messages(runtime, root)
    runtime._stopping_subtrees.add((root, "root"))
    runtime._mark_working("root", -1)
    assert not runtime._background_agent_tasks
    assert runtime._buffered_agent_messages == {}


async def test_user_and_peer_messages_share_fifo_without_extra_working_count(
    runtime, monkeypatch
):
    runtime, root = runtime
    runtime._active_root_id = root
    invocations = []

    async def invoke(agent_id, prompt, *, config, **_kwargs):
        invocations.append((agent_id, prompt, config))
        return f"reply: {prompt}"

    monkeypatch.setattr(runtime._manager, "invoke", invoke)
    runtime._mark_working("root", 1)
    user = asyncio.create_task(runtime.run_active_agent("root", "user first"))
    await asyncio.sleep(0)
    assert runtime._working_counts == {"root": 1}
    assert len(runtime._buffered_agent_messages["root"]) == 1

    await queue_peer_messages(runtime, root)
    assert len(runtime._buffered_agent_messages["root"]) == 4
    runtime._mark_working("root", -1)

    assert await user == "reply: user first"
    await asyncio.gather(*tuple(runtime._background_agent_tasks))
    assert [prompt for _id, prompt, _config in invocations] == ["user first"]
    user_config = invocations[0][2]
    assert user_config["metadata"] == {
        "ursa_message_sender": "user",
        "ursa_message_recipient": "root",
    }
    assert user_config["ursa_additional_messages"][0]["content"] == (
        "question one"
    )


async def test_mixed_user_batch_promotes_every_turn_without_delegated_spinner(
    runtime, monkeypatch
):
    runtime, root = runtime
    runtime._active_root_id = root
    invocations = []
    starts = []
    lifecycle = []

    async def invoke(agent_id, prompt, *, config, callbacks, **_kwargs):
        invocations.append((agent_id, prompt, config, callbacks))
        return "one combined reply"

    monkeypatch.setattr(runtime._manager, "invoke", invoke)
    runtime._mark_working("root", 1)
    requests = []
    for name in ("first", "second"):
        requests.append(
            asyncio.create_task(
                runtime.run_active_agent(
                    "root",
                    f"{name} user",
                    callbacks=[LifecycleCallback(lifecycle)],
                    on_start=lambda _message, name=name: starts.append(name),
                )
            )
        )
        await asyncio.sleep(0)
    runtime._mark_working("root", -1)

    assert await requests[0] is None
    assert await requests[1] == "one combined reply"
    assert starts == ["first", "second"]
    assert lifecycle == []
    assert invocations[0][1] == "first user"
    assert invocations[0][2]["ursa_additional_messages"] == [
        {"sender": "user", "recipient": "root", "content": "second user"}
    ]


@pytest.mark.parametrize(
    ("operation", "target", "error"),
    [
        ("close", "root", "closed"),
        ("reconfigure", "root", "reconfigured"),
        ("stop", "root/first", "stopped"),
    ],
)
async def test_lifecycle_change_fails_a_buffered_user_delivery(
    runtime, operation, target, error
):
    runtime, root = runtime
    runtime._active_root_id = root
    runtime._mark_working(target, 1)
    queued = asyncio.create_task(runtime.run_active_agent(target, "later"))
    await asyncio.sleep(0)

    if operation == "close":
        await runtime.aclose()
    elif operation == "reconfigure":
        await runtime.reconfigure_models(
            runtime.config.llm_model, runtime.config.emb_model
        )
    else:
        assert await runtime._stop_runtime_subagent(root, "first") == (
            "Stopped agent root/first and its subagents."
        )
        await asyncio.gather(*tuple(runtime._background_teardown_tasks))
    with pytest.raises(RuntimeError, match=error):
        await queued


async def test_cancelled_before_start_settles_user_delivery_future(runtime):
    runtime, _root = runtime
    response = asyncio.get_running_loop().create_future()
    item = _BufferedAgentMessage(
        sender_path="user",
        target_id="unused",
        target_path="root",
        message="queued",
        callbacks=(),
        response=response,
    )
    runtime._schedule_message_target(item)
    task = next(iter(runtime._background_agent_tasks))
    task.cancel()
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="cancelled before start"):
        await response
    assert runtime._working_counts == {}
    assert runtime._background_agent_tasks == set()


@pytest.mark.parametrize(
    ("stage", "error_type", "message"),
    [
        ("format_query", ValueError, "cannot format this message"),
        ("snapshot", TypeError, "cannot snapshot root state"),
        ("on_start", ValueError, "cannot start request"),
        ("ainvoke", ValueError, "model failed"),
        ("ainvoke", asyncio.CancelledError, "Agent invocation was cancelled"),
        ("format_result", ValueError, "cannot format result"),
    ],
)
async def test_bad_message_does_not_kill_the_root_inbox_worker(
    process_runner, monkeypatch, stage, error_type, message
):
    runner = process_runner()

    def fail(*args, **kwargs):
        raise error_type(message)

    with monkeypatch.context() as patch:
        if stage == "snapshot":

            class UncopyableState:
                __deepcopy__ = fail

            runner.state = UncopyableState()
        elif stage != "on_start":
            patch.setattr(
                runner.agent,
                stage,
                AsyncMock(side_effect=fail) if stage == "ainvoke" else fail,
            )
        raised = (
            RuntimeError if error_type is asyncio.CancelledError else error_type
        )
        with pytest.raises(raised, match=message):
            await runner._submit(
                "bad message", on_start=fail if stage == "on_start" else None
            )
        if stage == "format_result":
            assert runner.state["messages"][-1].content == "bad message"
    runner.state = None
    assert await runner.ainvoke("root survives") == "root survives"
    assert runner._worker is not None and not runner._worker.done()


@pytest.mark.parametrize("shutdown", ["worker", "close"])
async def test_shutdown_cancels_the_active_root_message_response(
    process_runner, monkeypatch, shutdown
):
    runner = process_runner()
    started = asyncio.Event()

    async def invoke(query, config=None):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner.agent, "ainvoke", invoke)
    active = asyncio.create_task(runner.ainvoke("working"))
    await asyncio.wait_for(started.wait(), 1)
    assert runner._worker is not None
    if shutdown == "close":
        queued = asyncio.create_task(runner.ainvoke("queued"))
        await asyncio.sleep(0)
        assert runner._inbox.qsize() == 1
        await runner.aclose()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert runner._inbox.empty()
    else:
        runner._worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    assert runner._worker.done()
