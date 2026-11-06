from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ursa.cli.config import UrsaConfig
from ursa.runtime import AgentContext, AgentManager, AgentRuntime
from ursa.runtime.agent_runtime import _BufferedAgentMessage, _UrsaProcessAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_query", [False, True])
async def test_process_delivers_batch_in_one_turn_with_separate_envelopes(
    reset_query,
):
    class Context:
        active_request_id = "batch"

        def _enter_request(self, request_id):
            return request_id

        def _leave_request(self, token):
            pass

    class CaptureAgent:
        calls = []
        formatted = []

        @staticmethod
        def format_query(prompt, state=None):
            CaptureAgent.formatted.append(prompt)
            if reset_query:
                state = {
                    "messages": [SystemMessage(content="Instructions")],
                    "task": prompt,
                }
            state = state or {
                "messages": [SystemMessage(content="Instructions")]
            }
            state["messages"].append(HumanMessage(content=prompt))
            return state

        async def ainvoke(self, query, config=None):
            assert "ursa_additional_messages" not in config
            assert "ursa_additional_messages" not in config["metadata"]
            self.calls.append(query)
            return query

        @staticmethod
        def format_result(state):
            return "processed"

    agent = CaptureAgent()
    runner = _UrsaProcessAgent(
        agent, cast(AgentContext, Context()), "root/worker"
    )
    bodies = [
        "First question\n[brackets]",
        'Second question: "quoted"',
        "New phase",
    ]
    senders = ["root/first", "root/second", "root"]
    try:
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
                    for sender, body in zip(
                        senders[1:], bodies[1:], strict=True
                    )
                ],
            },
        )
        assert result == "processed"
        assert len(agent.calls) == 1
        assert len(agent.formatted) == 1
        if reset_query:
            entries = json.loads(agent.calls[0]["task"].split("\n", 1)[1])
            assert [entry["content"] for entry in entries] == bodies
            assert [entry["sender"] for entry in entries] == senders
        messages = agent.calls[0]["messages"]
        assert len(messages) == 7
        for index, (sender, body) in enumerate(
            zip(senders, bodies, strict=True)
        ):
            envelope, message = messages[1 + index * 2 : 3 + index * 2]
            assert isinstance(envelope, SystemMessage)
            assert f"Sender: {sender}" in envelope.content
            assert "Recipient: root/worker" in envelope.content
            assert isinstance(message, HumanMessage)
            assert message.content == body
    finally:
        await runner.aclose()


@pytest.fixture
async def runtime(tmp_path, monkeypatch):
    manager = AgentManager()
    runtime = AgentRuntime(
        UrsaConfig(workspace=tmp_path, llm_model={"model": "fake:chat"}),
        manager=manager,
    )

    async def start(_agent_id):
        pass

    monkeypatch.setattr(manager, "start_agent", start)
    await runtime._ensure_registered("chat")
    root = runtime.agents["chat"].runtime_id
    await runtime._spawn_runtime_subagent(root, "first")
    await runtime._spawn_runtime_subagent(root, "second")
    try:
        yield runtime, root
    finally:
        await runtime.aclose()
        await manager.aclose()


async def queue_peer_messages(runtime, root, callbacks=()):
    expected = []
    for sender, content in (
        ("first", "question one"),
        ("second", "question two"),
        ("first", "newer instruction"),
    ):
        sender_id = runtime._hierarchies[root][f"root/{sender}"]
        result = await runtime._handle_runtime_request(
            sender_id,
            "invoke",
            {"target": "root", "message": content},
            callbacks,
        )
        assert result == "Message queued for root."
        expected.append({
            "sender": f"root/{sender}",
            "recipient": "root",
            "content": content,
        })
    return expected


@pytest.mark.asyncio
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

    class Callback:
        async def on_agent_message_start(self, path):
            lifecycle.append(("start", path))

        async def on_agent_message_end(self, path, *, succeeded):
            lifecycle.append(("end", path, succeeded))

    monkeypatch.setattr(runtime._manager, "invoke", invoke)
    runtime._mark_working("root", 1)
    expected = await queue_peer_messages(runtime, root, (Callback(),))
    assert runtime._pending_counts == {"root": 3}
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
    assert runtime._pending_counts == {}
    assert runtime._working_counts == {}
    assert runtime._buffered_agent_messages == {}


@pytest.mark.asyncio
async def test_wait_consumes_entire_buffer_once_without_replaying_it(runtime):
    runtime, root = runtime
    runtime._mark_working("root", 1)
    expected = await queue_peer_messages(runtime, root)
    result = await runtime._wait_for_runtime_agent(root, ["first", "second"], 0)
    assert result.startswith("Received 3 messages while waiting.")
    assert json.loads(result.split("\n", 1)[1]) == expected
    assert runtime._pending_counts == {}
    assert runtime._buffered_agent_messages == {}
    runtime._mark_working("root", -1)
    assert not runtime._background_agent_tasks
    again = await runtime._wait_for_runtime_agent(root, ["first", "second"], 0)
    assert "Received" not in again


@pytest.mark.asyncio
async def test_stopping_recipient_does_not_flush_a_new_batch(runtime):
    runtime, root = runtime
    runtime._mark_working("root", 1)
    await queue_peer_messages(runtime, root)
    runtime._stopping_subtrees.add((root, "root"))
    runtime._mark_working("root", -1)
    assert not runtime._background_agent_tasks
    assert runtime._pending_counts == {}
    assert runtime._buffered_agent_messages == {}


@pytest.mark.asyncio
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
    assert runtime._pending_counts == {"root": 1}

    await queue_peer_messages(runtime, root)
    assert runtime._pending_counts == {"root": 4}
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


@pytest.mark.asyncio
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

    class Callback:
        async def on_agent_message_start(self, path):
            lifecycle.append(("start", path))

        async def on_agent_message_end(self, path, *, succeeded):
            lifecycle.append(("end", path, succeeded))

    monkeypatch.setattr(runtime._manager, "invoke", invoke)
    runtime._mark_working("root", 1)
    first = asyncio.create_task(
        runtime.run_active_agent(
            "root",
            "first user",
            callbacks=[Callback()],
            on_start=lambda _message: starts.append("first"),
        )
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        runtime.run_active_agent(
            "root",
            "second user",
            callbacks=[Callback()],
            on_start=lambda _message: starts.append("second"),
        )
    )
    await asyncio.sleep(0)
    runtime._mark_working("root", -1)

    assert await first is None
    assert await second == "one combined reply"
    assert starts == ["first", "second"]
    assert lifecycle == []
    assert invocations[0][1] == "first user"
    assert invocations[0][2]["ursa_additional_messages"] == [
        {"sender": "user", "recipient": "root", "content": "second user"}
    ]


@pytest.mark.asyncio
async def test_shutdown_fails_a_buffered_user_delivery(runtime):
    runtime, root = runtime
    runtime._active_root_id = root
    runtime._mark_working("root", 1)
    queued = asyncio.create_task(runtime.run_active_agent("root", "later"))
    await asyncio.sleep(0)

    await runtime.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await queued


@pytest.mark.asyncio
async def test_reconfigure_fails_a_buffered_user_delivery(runtime):
    runtime, root = runtime
    runtime._active_root_id = root
    runtime._mark_working("root", 1)
    queued = asyncio.create_task(runtime.run_active_agent("root", "later"))
    await asyncio.sleep(0)

    await runtime.reconfigure_models(
        runtime.config.llm_model, runtime.config.emb_model
    )
    with pytest.raises(RuntimeError, match="reconfigured"):
        await queued


@pytest.mark.asyncio
async def test_stop_fails_a_buffered_user_delivery(runtime):
    runtime, root = runtime
    runtime._active_root_id = root
    child_path = "root/first"
    runtime._mark_working(child_path, 1)
    queued = asyncio.create_task(runtime.run_active_agent(child_path, "later"))
    await asyncio.sleep(0)

    assert await runtime._stop_runtime_subagent(root, "first") == (
        "Stopped agent root/first and its subagents."
    )
    await asyncio.gather(*tuple(runtime._background_teardown_tasks))
    with pytest.raises(RuntimeError, match="stopped"):
        await queued


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_bad_message_format_does_not_kill_the_root_inbox_worker():
    class Context:
        active_request_id = "message"

        def _enter_request(self, request_id):
            return request_id

        def _leave_request(self, token):
            pass

    class Agent:
        def format_query(self, prompt, state=None):
            if prompt == "bad message":
                raise ValueError("cannot format this message")
            return {"messages": [HumanMessage(content=prompt)]}

        async def ainvoke(self, query, config=None):
            return query

        @staticmethod
        def format_result(result):
            return result["messages"][-1].content

    runner = _UrsaProcessAgent(Agent(), cast(AgentContext, Context()), "root")
    try:
        with pytest.raises(ValueError, match="cannot format"):
            await runner.ainvoke("bad message")
        assert await runner.ainvoke("root survives") == "root survives"
        assert runner._worker is not None and not runner._worker.done()
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_bad_root_state_snapshot_does_not_kill_the_inbox_worker():
    class Context:
        active_request_id = "message"

        def _enter_request(self, request_id):
            return request_id

        def _leave_request(self, token):
            pass

    class UncopyableState:
        def __deepcopy__(self, memo):
            raise TypeError("cannot snapshot root state")

    class Agent:
        @staticmethod
        def format_query(prompt, state=None):
            return {"messages": [HumanMessage(content=prompt)]}

        async def ainvoke(self, query, config=None):
            return query

        @staticmethod
        def format_result(result):
            return result["messages"][-1].content

    runner = _UrsaProcessAgent(Agent(), cast(AgentContext, Context()), "root")
    try:
        runner.state = UncopyableState()
        with pytest.raises(TypeError, match="cannot snapshot"):
            await runner.ainvoke("bad state")
        runner.state = None
        assert await runner.ainvoke("root survives") == "root survives"
        assert runner._worker is not None and not runner._worker.done()
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_shutdown_cancels_the_active_root_message_response():
    class Context:
        active_request_id = "message"

        def _enter_request(self, request_id):
            return request_id

        def _leave_request(self, token):
            pass

    started = asyncio.Event()

    class Agent:
        @staticmethod
        def format_query(prompt, state=None):
            return {"messages": [HumanMessage(content=prompt)]}

        async def ainvoke(self, query, config=None):
            started.set()
            await asyncio.Event().wait()
            return query

        @staticmethod
        def format_result(result):
            return result["messages"][-1].content

    runner = _UrsaProcessAgent(Agent(), cast(AgentContext, Context()), "root")
    try:
        active = asyncio.create_task(runner.ainvoke("working"))
        await asyncio.wait_for(started.wait(), 1)
        assert runner._worker is not None
        runner._worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert runner._worker.done()
    finally:
        await runner.aclose()
