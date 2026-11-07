import asyncio
from typing import Any

import pytest
from langchain.tools import ToolException

from ursa.agents.base import BaseAgent
from ursa.messaging import AgentMessage
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


@pytest.fixture
async def mailbox(chat_model, tmp_path):
    agents = []

    def create(agent_type=MailboxAgent, **kwargs):
        agent = agent_type(llm=chat_model, workspace=tmp_path, **kwargs)
        agents.append(agent)
        return agent

    yield create
    for agent in agents:
        await agent.stop_message_loop(descendants=True, cancel=True)


def hierarchy_tool(agent: BaseAgent, name: str):
    return {tool.name: tool for tool in build_agent_tools(agent)}[name]


async def test_user_can_post_and_request_messages(mailbox):
    agent = mailbox()

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


async def test_agents_spawn_hierarchies_and_message_siblings(mailbox):
    root = mailbox()
    analyst = root.spawn_subagent(
        "analyst",
    )
    reviewer = root.spawn_subagent(
        "reviewer",
    )
    specialist = analyst.spawn_subagent(
        "specialist",
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


async def test_wait_for_local_agent_completion_timeout_and_message_wake(
    mailbox,
):
    root = mailbox()
    started = asyncio.Event()
    release = asyncio.Event()
    worker = root.spawn_subagent(
        "worker",
        BlockingMailboxAgent,
        started=started,
        release=release,
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


async def test_wait_for_local_path_cannot_escape_root(mailbox):
    root = mailbox()

    with pytest.raises(ToolException, match="Path escapes hierarchy root"):
        await hierarchy_tool(root, "wait_for").ainvoke({"target": "../missing"})
    assert not root.is_waiting


@pytest.mark.parametrize("limit", ["max_agents", "max_depth"])
def test_subagent_limit_is_hierarchy_wide(mailbox, limit):
    root = mailbox(**{f"subagent_{limit}": 2})
    child = root.spawn_subagent("child")
    grandchild = child.spawn_subagent("grandchild")
    parent = root if limit == "max_agents" else grandchild

    with pytest.raises(RuntimeError, match=f"{limit}=2"):
        parent.spawn_subagent("overflow")


@pytest.mark.parametrize(
    "limit",
    [
        {"subagent_max_agents": 0},
        {"subagent_max_depth": 0},
    ],
)
def test_zero_subagent_limit_disables_spawning(mailbox, limit):
    root = mailbox(**limit)

    with pytest.raises(RuntimeError, match="Subagents are disabled"):
        root.spawn_subagent("disabled")


def test_none_removes_subagent_count_and_depth_limits(mailbox):
    root = mailbox(
        subagent_max_agents=None,
        subagent_max_depth=None,
    )
    current = root

    for index in range(5):
        current = current.spawn_subagent(
            f"level-{index}",
            MailboxAgent,
        )

    assert len(list(root.walk_agents())) == 6
    assert current.agent_path.endswith("level-4")


def test_subagents_inherit_configured_models(
    mailbox, chat_model, embedding_model
):
    alternate_model = chat_model.model_copy()
    root = mailbox(
        subagent_llm=alternate_model,
        subagent_embedding=embedding_model,
    )

    child = root.spawn_subagent("child")
    grandchild = child.spawn_subagent("grandchild")

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


async def test_hierarchy_tool_propagates_callbacks_to_child(mailbox):
    class HierarchyCallback:
        def __init__(self):
            self.lifecycle = []

        async def on_agent_message_start(self, path, *, message_id):
            self.lifecycle.append(("start", path, message_id))

        async def on_agent_message_end(self, path, *, succeeded, message_id):
            self.lifecycle.append(("end", path, message_id, succeeded))

    root = mailbox()
    reviewer = root.spawn_subagent(
        "reviewer",
    )
    callback = HierarchyCallback()
    root._current_agent_message = AgentMessage(
        content="delegate",
        sender="user",
        recipient="root",
        config={"callbacks": [callback]},
    )
    send_message = hierarchy_tool(root, "send_message")

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


async def test_send_message_everyone_errors_when_the_sender_is_alone(mailbox):
    root = mailbox()

    with pytest.raises(ToolException, match="@everyone.*has no recipients"):
        await hierarchy_tool(root, "send_message").ainvoke({
            "target": "@everyone",
            "message": "Nobody is here.",
        })


async def test_managed_send_message_group_errors_are_recoverable_and_results_are_concrete(
    mailbox,
):
    class ManagedBridge:
        async def send_message(self, target: str, message: str) -> str:
            if target == "@reports":
                raise RuntimeError(
                    "Message target group '@reports' has no recipients"
                )
            return f"Message queued for {target}."

        async def send_messages(
            self, targets: list[str], message: str
        ) -> list[dict[str, str]]:
            assert targets == ["@everyone"]
            return [
                {"result": f"Message queued for root/p{index:03}."}
                for index in range(80)
            ]

    root = mailbox()
    root._process_runtime = ManagedBridge()
    send_message = hierarchy_tool(root, "send_message")

    with pytest.raises(ToolException, match="has no recipients"):
        await send_message.ainvoke({
            "target": "@reports",
            "message": "Nothing should receive this.",
        })

    assert (
        await send_message.ainvoke({
            "target": "root/reviewer",
            "message": "Continue normally.",
        })
        == "Message queued for root/reviewer."
    )
    response = await send_message.ainvoke({
        "target": ["@everyone"],
        "message": "Coordinate.",
    })
    assert response.count("Message queued for root/") == 80
    assert response.startswith("Message queued for root/p000.\n\n")
    assert "{'result':" not in response
    assert "root/p000" in response
    assert "root/p079" in response


@pytest.mark.parametrize("target", ["reviewer", "reviewer/critic"])
async def test_stop_agent_removes_a_descendant_subtree(mailbox, target):
    root = mailbox()
    reviewer = root.spawn_subagent("reviewer")
    critic = reviewer.spawn_subagent("critic")
    analyst = root.spawn_subagent("analyst")

    result = await hierarchy_tool(root, "stop_agent").ainvoke({
        "target": target,
    })

    assert result == f"Stopped agent root/{target} and its subagents."
    if target == "reviewer":
        assert list(root.subagents) == ["analyst"]
        assert reviewer.parent_agent is None
        assert critic.parent_agent is reviewer
    else:
        assert list(root.subagents) == ["reviewer", "analyst"]
        assert reviewer.parent_agent is root
        assert not reviewer.subagents
        assert critic.parent_agent is None
    assert analyst.parent_agent is root


async def test_stop_agent_rejects_non_descendant_paths(mailbox):
    root = mailbox()
    reviewer = root.spawn_subagent("reviewer")
    root.spawn_subagent("analyst")

    with pytest.raises(ToolException, match="only stop one of its descendants"):
        await hierarchy_tool(root, "stop_agent").ainvoke({"target": "root"})
    with pytest.raises(ToolException, match="only stop one of its descendants"):
        await hierarchy_tool(reviewer, "stop_agent").ainvoke({
            "target": "root/analyst",
        })

    assert set(root.subagents) == {"reviewer", "analyst"}


async def test_stopped_local_subtree_rejects_stale_references(mailbox):
    root = mailbox()
    reviewer = root.spawn_subagent("reviewer")
    critic = reviewer.spawn_subagent("critic")

    root.stop_subagent("reviewer")

    with pytest.raises(RuntimeError, match="root/reviewer/critic.*is stopping"):
        await critic.post_message("Do not accept this.")
    with pytest.raises(RuntimeError, match="root/reviewer.*is stopping"):
        reviewer.spawn_subagent("late")
    with pytest.raises(
        ToolException, match="root/reviewer/critic.*is stopping"
    ):
        await hierarchy_tool(reviewer, "send_message").ainvoke({
            "target": "critic",
            "message": "Do not accept this either.",
        })

    await asyncio.sleep(0)


async def test_spawn_agent_returns_while_initial_task_is_still_running(mailbox):
    root = mailbox(BlockingMailboxAgent)
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


async def test_async_factory_can_spawn_subagent(mailbox):
    root = mailbox()

    async def factory(**kwargs):
        await asyncio.sleep(0)
        return MailboxAgent(**kwargs)

    child = await root.aspawn_subagent("async-child", factory)

    assert child.parent_agent is root
    assert root.resolve_agent(child.agent_path) is child


async def test_request_is_rejected_while_message_loop_is_stopping(mailbox):
    agent = mailbox(BlockingMailboxAgent)
    request = asyncio.create_task(agent.request("in progress"))
    await agent.started.wait()
    stopping = asyncio.create_task(agent.stop_message_loop())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="inbox is stopping"):
        await agent.request("too late")

    agent.release.set()
    assert await request == {"content": "in progress"}
    await stopping


async def test_direct_invocations_serialize_but_allow_owner_reentry(
    mailbox, monkeypatch
):
    agent = mailbox()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def invoke(inputs, **_config):
        content = inputs["messages"][-1].content
        calls.append(content)
        assert agent.is_working
        if content == "outer":
            assert await agent.ainvoke("inner") == "inner"
            entered.set()
            await release.wait()
        return content

    monkeypatch.setattr(agent, "_ainvoke", invoke)
    outer = asyncio.create_task(agent.ainvoke("outer"))
    competing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        competing = asyncio.create_task(agent.ainvoke("competing"))
        await asyncio.sleep(0)
        assert calls == ["outer", "inner"]
        assert not competing.done()
        release.set()
        assert await asyncio.wait_for(
            asyncio.gather(outer, competing), timeout=2
        ) == ["outer", "competing"]
        assert calls == ["outer", "inner", "competing"]
        assert agent._async_invocation_owner is None
        assert not agent.is_working
    finally:
        for task in (outer, competing):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (outer, competing) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize("limit", ["subagent_max_agents", "subagent_max_depth"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_messaging_lifecycle_rejects_invalid_limits(mailbox, limit, value):
    with pytest.raises(
        ValueError, match=f"{limit} must be a non-negative integer"
    ):
        mailbox(**{limit: value})
