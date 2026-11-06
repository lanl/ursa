import asyncio
from typing import Any

import pytest
from langchain.tools import ToolException

from ursa.agents.base import (
    AgentMessage,
    BaseAgent,
)
from ursa.tools.agent_tools import build_agent_tools


class MailboxAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.received: list[AgentMessage] = []
        kwargs.setdefault("enable_metrics", False)
        super().__init__(*args, **kwargs)

    def _noop(self, state):
        return state

    def _build_graph(self):
        self.add_node(self._noop, "noop")
        self.graph.set_entry_point("noop")
        self.graph.set_finish_point("noop")

    async def ahandle_message(self, message: AgentMessage) -> Any:
        self.received.append(message)
        return {
            "content": message.content,
            "sender": message.sender,
            "recipient": message.recipient,
        }


class BlockingMailboxAgent(MailboxAgent):
    """Mailbox test double whose current work is released by the test."""

    def __init__(
        self,
        *args,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        **kwargs,
    ):
        self.started = started or asyncio.Event()
        self.release = release or asyncio.Event()
        super().__init__(*args, **kwargs)

    async def ahandle_message(self, message: AgentMessage) -> Any:
        self.received.append(message)
        self.started.set()
        await self.release.wait()
        return {"content": message.content}


def make_agent(chat_model, tmp_path, **kwargs) -> MailboxAgent:
    return MailboxAgent(llm=chat_model, workspace=tmp_path, **kwargs)


def hierarchy_tool(agent: BaseAgent, name: str):
    return {tool.name: tool for tool in build_agent_tools(agent)}[name]


@pytest.mark.asyncio
async def test_user_can_post_and_request_messages(chat_model, tmp_path):
    agent = make_agent(chat_model, tmp_path)

    envelope = await agent.post_message("one-way")
    result = await agent.request("reply please")
    await agent.wait_for_messages()

    assert envelope.sender == "user"
    assert envelope.recipient == agent.agent_path
    assert [message.content for message in agent.received] == [
        "one-way",
        "reply please",
    ]
    assert result == {
        "content": "reply please",
        "sender": "user",
        "recipient": agent.agent_path,
    }

    await agent.stop_message_loop()


@pytest.mark.asyncio
async def test_agents_spawn_hierarchies_and_message_siblings(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    analyst = root.spawn_subagent(
        "analyst",
        MailboxAgent,
        enable_metrics=False,
    )
    reviewer = root.spawn_subagent(
        "reviewer",
        MailboxAgent,
        enable_metrics=False,
    )
    specialist = analyst.spawn_subagent(
        "specialist",
        MailboxAgent,
        enable_metrics=False,
    )

    assert root.hierarchy_name == "root"
    assert [agent.agent_path for agent in root.walk_agents()] == [
        "root",
        "root/analyst",
        "root/analyst/specialist",
        "root/reviewer",
    ]
    assert specialist.resolve_agent("../..") is root
    assert analyst.resolve_agent("reviewer") is reviewer

    result = await analyst.ask_agent("reviewer", "check this")
    assert result["sender"] == analyst.agent_path
    assert result["recipient"] == reviewer.agent_path

    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_wait_for_local_agent_completion_timeout_and_message_wake(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    worker = root.spawn_subagent(
        "worker",
        BlockingMailboxAgent,
        started=started,
        release=release,
        enable_metrics=False,
    )
    wait_for = hierarchy_tool(root, "wait_for")

    work = asyncio.create_task(worker.request("long task"))
    await asyncio.wait_for(started.wait(), 1)

    timed_out = await wait_for.ainvoke({
        "target": "worker",
        "timeout_seconds": 0,
    })
    assert timed_out == "Agent root/worker did not finish within 0 seconds."

    completed_wait = asyncio.create_task(
        wait_for.ainvoke({
            "target": "worker",
            "timeout_seconds": 1,
        })
    )
    await asyncio.sleep(0)
    release.set()
    assert await completed_wait == "Agent root/worker finished its work."
    await work

    # Start another piece of work, then deliver an ordinary inbox message to
    # the caller. The wait consumes the message so the model can react to its
    # content immediately without receiving a duplicate later turn.
    started.clear()
    release.clear()
    work = asyncio.create_task(worker.request("second long task"))
    await asyncio.wait_for(started.wait(), 1)
    wake_wait = asyncio.create_task(
        wait_for.ainvoke({
            "target": "worker",
            "timeout_seconds": 1,
        })
    )
    for _ in range(100):
        if root._message_waiters:
            break
        await asyncio.sleep(0)
    assert root._message_waiters
    await root.post_message("new direction")
    assert (
        await wake_wait == "Received a message while waiting.\n"
        "Sender: user\n"
        "Recipient: root\n"
        "Exact message content:\n"
        "new direction"
    )
    assert not root.received
    release.set()
    await work
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_wait_for_local_agent_list_modes_and_reciprocal_waits(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    first = root.spawn_subagent("first", MailboxAgent, enable_metrics=False)
    second = root.spawn_subagent("second", MailboxAgent, enable_metrics=False)
    third = root.spawn_subagent("third", MailboxAgent, enable_metrics=False)
    wait_for = hierarchy_tool(root, "wait_for")

    # Model calls are normally marked as current messages by the inbox loop.
    # These sentinel messages let this focused tool test model active work.
    first._current_agent_message = AgentMessage(
        content="first task", sender="user", recipient=first.agent_path
    )
    second._current_agent_message = AgentMessage(
        content="second task", sender="user", recipient=second.agent_path
    )
    root._current_agent_message = AgentMessage(
        content="root task", sender="user", recipient=root.agent_path
    )
    any_wait = asyncio.create_task(
        wait_for.ainvoke({"target": ["first", "second"], "timeout_seconds": 1})
    )
    for _ in range(100):
        if root.is_waiting:
            break
        await asyncio.sleep(0)
    assert root.is_waiting
    first._current_agent_message = None
    assert await any_wait == "Agent root/first finished its work."

    first._current_agent_message = AgentMessage(
        content="first task", sender="user", recipient=first.agent_path
    )
    all_wait = asyncio.create_task(
        wait_for.ainvoke({
            "target": ["first", "second"],
            "timeout_seconds": 1,
            "wait_all": True,
        })
    )
    for _ in range(100):
        if root.is_waiting:
            break
        await asyncio.sleep(0)
    assert root.is_waiting
    first._current_agent_message = None
    await asyncio.sleep(0)
    assert not all_wait.done()
    second._current_agent_message = None
    assert await all_wait == (
        "All agents finished their work: root/first, root/second."
    )

    # Root waits for two active children. A direct dependency may wait back on
    # root without deadlocking, but an unrelated third agent must not treat
    # root as ready merely because it is waiting on someone else.
    first._current_agent_message = AgentMessage(
        content="first task", sender="user", recipient=first.agent_path
    )
    second._current_agent_message = AgentMessage(
        content="second task", sender="user", recipient=second.agent_path
    )
    root_wait = asyncio.create_task(
        wait_for.ainvoke({
            "target": ["first", "second"],
            "timeout_seconds": 1,
            "wait_all": True,
        })
    )
    for _ in range(100):
        if root.is_waiting:
            break
        await asyncio.sleep(0)
    assert root.is_waiting
    unrelated_wait = asyncio.create_task(
        hierarchy_tool(third, "wait_for").ainvoke({
            "target": "root",
            "timeout_seconds": 1,
        })
    )
    await asyncio.sleep(0)
    assert not unrelated_wait.done()
    assert await hierarchy_tool(first, "wait_for").ainvoke({
        "target": "root",
        "timeout_seconds": 1,
    }) == (
        "Agent root is ready for this wait because it is waiting on you "
        "directly."
    )
    first._current_agent_message = None
    await asyncio.sleep(0)
    assert not root_wait.done()
    second._current_agent_message = None
    assert await root_wait == (
        "All agents finished their work: root/first, root/second."
    )
    root._current_agent_message = None
    assert await unrelated_wait == "Agent root finished its work."

    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_wait_for_local_indirect_cycle_and_timeout_cleanup(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    middle = root.spawn_subagent("middle", MailboxAgent, enable_metrics=False)
    end = root.spawn_subagent("end", MailboxAgent, enable_metrics=False)
    root._current_agent_message = AgentMessage(
        content="root task", sender="user", recipient=root.agent_path
    )
    middle._current_agent_message = AgentMessage(
        content="middle task", sender="user", recipient=middle.agent_path
    )
    end._current_agent_message = AgentMessage(
        content="end task", sender="user", recipient=end.agent_path
    )

    root_wait = asyncio.create_task(
        hierarchy_tool(root, "wait_for").ainvoke({
            "target": "middle",
            "timeout_seconds": 1,
        })
    )
    for _ in range(100):
        if root.is_waiting:
            break
        await asyncio.sleep(0)
    assert root.is_waiting
    middle_wait = asyncio.create_task(
        hierarchy_tool(middle, "wait_for").ainvoke({
            "target": "end",
            "timeout_seconds": 1,
        })
    )
    for _ in range(100):
        if middle.is_waiting:
            break
        await asyncio.sleep(0)
    assert middle.is_waiting
    assert await hierarchy_tool(end, "wait_for").ainvoke({
        "target": "root",
        "timeout_seconds": 1,
    }) == (
        "Agent root is ready for this wait because it is waiting on you "
        "indirectly."
    )

    root_wait.cancel()
    middle_wait.cancel()
    await asyncio.gather(root_wait, middle_wait, return_exceptions=True)
    assert not root.is_waiting
    assert not middle.is_waiting

    timed_out = await hierarchy_tool(root, "wait_for").ainvoke({
        "target": "middle",
        "timeout_seconds": 0,
    })
    assert timed_out == "Agent root/middle did not finish within 0 seconds."
    assert not root.is_waiting
    stale_reverse_wait = asyncio.create_task(
        hierarchy_tool(middle, "wait_for").ainvoke({
            "target": "root",
            "timeout_seconds": 1,
        })
    )
    await asyncio.sleep(0)
    assert not stale_reverse_wait.done()
    stale_reverse_wait.cancel()
    await asyncio.gather(stale_reverse_wait, return_exceptions=True)

    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_wait_for_invalid_local_path_is_recoverable(chat_model, tmp_path):
    root = make_agent(chat_model, tmp_path)
    child = root.spawn_subagent("child", MailboxAgent, enable_metrics=False)

    with pytest.raises(ToolException, match="Path escapes hierarchy root"):
        await hierarchy_tool(root, "wait_for").ainvoke({
            "target": "../missing",
        })
    with pytest.raises(ToolException, match="Unknown agent path"):
        await hierarchy_tool(root, "wait_for").ainvoke({
            "target": ["child", "missing"],
        })
    assert not root.is_waiting

    # Waiting on an ancestor is allowed: a waiting ancestor is ready, which
    # prevents reciprocal waits from deadlocking.
    assert await hierarchy_tool(child, "wait_for").ainvoke({
        "target": "root"
    }) == ("Agent root has already finished its work.")

    await root.stop_message_loop(descendants=True)


def test_subagent_count_limit_is_hierarchy_wide(chat_model, tmp_path):
    root = make_agent(
        chat_model,
        tmp_path,
        subagent_max_agents=2,
    )
    first = root.spawn_subagent("first", MailboxAgent, enable_metrics=False)
    first.spawn_subagent("nested", MailboxAgent, enable_metrics=False)

    with pytest.raises(RuntimeError, match="max_agents=2"):
        root.spawn_subagent("one-too-many", MailboxAgent, enable_metrics=False)


def test_subagent_depth_limit_is_hierarchy_wide(chat_model, tmp_path):
    root = make_agent(
        chat_model,
        tmp_path,
        subagent_max_depth=2,
    )
    child = root.spawn_subagent("child", MailboxAgent, enable_metrics=False)
    grandchild = child.spawn_subagent(
        "grandchild", MailboxAgent, enable_metrics=False
    )

    with pytest.raises(RuntimeError, match="max_depth=2"):
        grandchild.spawn_subagent(
            "too-deep", MailboxAgent, enable_metrics=False
        )


@pytest.mark.parametrize(
    "limit",
    [
        {"subagent_max_agents": 0},
        {"subagent_max_depth": 0},
    ],
)
def test_zero_subagent_limit_disables_spawning(chat_model, tmp_path, limit):
    root = make_agent(chat_model, tmp_path, **limit)

    with pytest.raises(RuntimeError, match="Subagents are disabled"):
        root.spawn_subagent("disabled", MailboxAgent, enable_metrics=False)


def test_none_removes_subagent_count_and_depth_limits(chat_model, tmp_path):
    root = make_agent(
        chat_model,
        tmp_path,
        subagent_max_agents=None,
        subagent_max_depth=None,
    )
    current = root

    for index in range(5):
        current = current.spawn_subagent(
            f"level-{index}",
            MailboxAgent,
            enable_metrics=False,
        )

    assert len(list(root.walk_agents())) == 6
    assert current.agent_path.endswith("level-4")


def test_subagents_inherit_configured_models(
    chat_model, embedding_model, tmp_path
):
    alternate_model = chat_model.model_copy()
    root = make_agent(
        chat_model,
        tmp_path,
        subagent_llm=alternate_model,
        subagent_embedding=embedding_model,
    )

    child = root.spawn_subagent("child", MailboxAgent, enable_metrics=False)
    grandchild = child.spawn_subagent(
        "grandchild", MailboxAgent, enable_metrics=False
    )

    def innermost(model):
        seen = set()
        while id(model) not in seen and hasattr(model, "bound"):
            seen.add(id(model))
            model = model.bound
        return model

    assert innermost(root.llm) is chat_model
    assert innermost(child.llm) is alternate_model
    assert innermost(grandchild.llm) is alternate_model
    assert child.rag_tool_embedding is embedding_model
    assert grandchild.rag_tool_embedding is embedding_model


@pytest.mark.asyncio
async def test_hierarchy_tool_propagates_callbacks_to_child(
    chat_model, tmp_path
):
    class HierarchyCallback:
        def __init__(self):
            self.lifecycle = []

        async def on_agent_message_start(self, path, *, message_id):
            self.lifecycle.append(("start", path, message_id))

        async def on_agent_message_end(self, path, *, succeeded, message_id):
            self.lifecycle.append(("end", path, message_id, succeeded))

    root = make_agent(chat_model, tmp_path)
    reviewer = root.spawn_subagent(
        "reviewer",
        MailboxAgent,
        enable_metrics=False,
    )
    callback = HierarchyCallback()
    root._current_agent_message = AgentMessage(
        content="delegate",
        sender="user",
        recipient="root",
        config={"callbacks": [callback]},
    )
    send_message = {
        hierarchy_tool.name: hierarchy_tool
        for hierarchy_tool in build_agent_tools(root)
    }["send_message"]

    try:
        response = await send_message.ainvoke({
            "target": "root/reviewer",
            "message": "review this",
        })
        await reviewer.wait_for_messages()
    finally:
        root._current_agent_message = None

    assert response == "Message queued for root/reviewer."
    assert reviewer.received[0].config["callbacks"] == [callback]
    assert callback.lifecycle[0][:2] == ("start", "root/reviewer")
    assert callback.lifecycle[1] == (
        "end",
        "root/reviewer",
        callback.lifecycle[0][2],
        True,
    )
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_hierarchy_tool_rejects_invalid_batch_before_delivery(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    analyst = root.spawn_subagent("analyst", MailboxAgent, enable_metrics=False)
    reviewer = root.spawn_subagent(
        "reviewer", MailboxAgent, enable_metrics=False
    )
    send_message = {
        hierarchy_tool.name: hierarchy_tool
        for hierarchy_tool in build_agent_tools(root)
    }["send_message"]

    with pytest.raises(ToolException, match="Unknown agent path"):
        await send_message.ainvoke({
            "target": ["analyst", "missing", "reviewer"],
            "message": "Compare the proposal.",
        })

    assert analyst.received == []
    assert reviewer.received == []
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_hierarchy_tool_broadcasts_valid_targets_in_input_order(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    analyst = root.spawn_subagent("analyst", MailboxAgent, enable_metrics=False)
    reviewer = root.spawn_subagent(
        "reviewer", MailboxAgent, enable_metrics=False
    )
    send_message = {
        hierarchy_tool.name: hierarchy_tool
        for hierarchy_tool in build_agent_tools(root)
    }["send_message"]

    response = await send_message.ainvoke({
        "target": ["reviewer", "analyst"],
        "message": "Compare the proposal.",
    })
    await asyncio.gather(
        reviewer.wait_for_messages(), analyst.wait_for_messages()
    )

    assert response.index("reviewer:") < response.index("analyst:")
    assert "Message queued for root/reviewer." in response
    assert "Message queued for root/analyst." in response
    assert reviewer.received[0].content == "Compare the proposal."
    assert analyst.received[0].content == "Compare the proposal."
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_stop_agent_removes_a_descendant_subtree(chat_model, tmp_path):
    root = make_agent(chat_model, tmp_path)
    reviewer = root.spawn_subagent(
        "reviewer", MailboxAgent, enable_metrics=False
    )
    critic = reviewer.spawn_subagent(
        "critic", MailboxAgent, enable_metrics=False
    )
    analyst = root.spawn_subagent("analyst", MailboxAgent, enable_metrics=False)

    result = await hierarchy_tool(root, "stop_agent").ainvoke({
        "target": "reviewer",
    })

    assert result == "Stopped agent root/reviewer and its subagents."
    assert list(root.subagents) == ["analyst"]
    assert reviewer.parent_agent is None
    assert critic.parent_agent is reviewer
    assert analyst.parent_agent is root
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_stop_agent_rejects_non_descendant_paths(chat_model, tmp_path):
    root = make_agent(chat_model, tmp_path)
    reviewer = root.spawn_subagent(
        "reviewer", MailboxAgent, enable_metrics=False
    )
    root.spawn_subagent("analyst", MailboxAgent, enable_metrics=False)

    with pytest.raises(ToolException, match="only stop one of its descendants"):
        await hierarchy_tool(root, "stop_agent").ainvoke({"target": "root"})
    with pytest.raises(ToolException, match="only stop one of its descendants"):
        await hierarchy_tool(reviewer, "stop_agent").ainvoke({
            "target": "root/analyst",
        })

    assert set(root.subagents) == {"reviewer", "analyst"}
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_stopped_local_subtree_rejects_stale_references(
    chat_model, tmp_path
):
    root = make_agent(chat_model, tmp_path)
    reviewer = root.spawn_subagent(
        "reviewer", MailboxAgent, enable_metrics=False
    )
    critic = reviewer.spawn_subagent(
        "critic", MailboxAgent, enable_metrics=False
    )

    root.stop_subagent("reviewer")

    with pytest.raises(RuntimeError, match="root/reviewer/critic.*is stopping"):
        await critic.post_message("Do not accept this.")
    with pytest.raises(RuntimeError, match="root/reviewer.*is stopping"):
        reviewer.spawn_subagent("late", MailboxAgent, enable_metrics=False)
    with pytest.raises(
        ToolException, match="root/reviewer/critic.*is stopping"
    ):
        await hierarchy_tool(reviewer, "send_message").ainvoke({
            "target": "critic",
            "message": "Do not accept this either.",
        })

    await asyncio.sleep(0)
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_spawn_agent_returns_while_initial_task_is_still_running(
    chat_model, tmp_path
):
    root = BlockingMailboxAgent(
        llm=chat_model,
        workspace=tmp_path,
        enable_metrics=False,
    )
    spawn_agent = hierarchy_tool(root, "spawn_agent")

    response = await asyncio.wait_for(
        spawn_agent.ainvoke({
            "name": "worker",
            "message": "Keep working until released.",
        }),
        timeout=1,
    )
    worker = root.resolve_agent("root/worker")
    assert isinstance(worker, BlockingMailboxAgent)
    await asyncio.wait_for(worker.started.wait(), timeout=1)

    assert response == (
        "Spawned agent root/worker\n\nMessage queued for root/worker."
    )
    assert worker.is_working
    assert not worker.release.is_set()

    worker.release.set()
    await worker.wait_for_messages()
    await root.stop_message_loop(descendants=True)


@pytest.mark.asyncio
async def test_async_factory_can_spawn_subagent(chat_model, tmp_path):
    root = make_agent(chat_model, tmp_path)

    async def factory(**kwargs):
        await asyncio.sleep(0)
        kwargs["enable_metrics"] = False
        return MailboxAgent(**kwargs)

    child = await root.aspawn_subagent("async-child", factory)

    assert child.parent_agent is root
    assert root.resolve_agent(child.agent_path) is child


@pytest.mark.asyncio
async def test_request_is_rejected_while_message_loop_is_stopping(
    chat_model, tmp_path
):
    class SlowAgent(MailboxAgent):
        def __init__(self, *args, **kwargs):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            super().__init__(*args, **kwargs)

        async def ahandle_message(self, message: AgentMessage) -> Any:
            self.started.set()
            await self.release.wait()
            return message.content

    agent = SlowAgent(
        llm=chat_model,
        workspace=tmp_path,
        enable_metrics=False,
    )
    request = asyncio.create_task(agent.request("in progress"))
    await agent.started.wait()
    stopping = asyncio.create_task(agent.stop_message_loop())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="inbox is stopping"):
        await agent.request("too late")

    agent.release.set()
    assert await request == "in progress"
    await stopping
