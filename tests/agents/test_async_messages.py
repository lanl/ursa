import asyncio
from contextlib import suppress
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ursa.agents.base import (
    DEFAULT_INTERRUPT_MESSAGE,
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


def make_agent(chat_model, tmp_path, **kwargs) -> MailboxAgent:
    return MailboxAgent(llm=chat_model, workspace=tmp_path, **kwargs)


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
    finally:
        root._current_agent_message = None

    assert response
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
async def test_interrupt_cancels_active_handler_and_prioritizes_new_message(
    chat_model, tmp_path
):
    class InterruptibleAgent(MailboxAgent):
        def __init__(self, *args, **kwargs):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            super().__init__(*args, **kwargs)

        async def ahandle_message(self, message: AgentMessage) -> Any:
            self.received.append(message)
            if message.content == "original":
                self.started.set()
                await self.release.wait()
            return message

    agent = InterruptibleAgent(
        llm=chat_model,
        workspace=tmp_path,
        enable_metrics=False,
    )
    started: list[str] = []

    async def record_start(message: AgentMessage) -> None:
        started.append(str(message.content))

    original = asyncio.create_task(agent.request("original"))
    await agent.started.wait()
    queued = asyncio.create_task(
        agent.request("already queued", on_start=record_start)
    )
    await asyncio.sleep(0)

    replacement = await agent.interrupt("new direction", on_start=record_start)
    queued_result = await queued

    assert isinstance(replacement, AgentMessage)
    assert replacement.content == "new direction"
    assert replacement.system_message == DEFAULT_INTERRUPT_MESSAGE
    assert queued_result.content == "already queued"
    assert [message.content for message in agent.received] == [
        "original",
        "new direction",
        "already queued",
    ]
    assert started == ["new direction", "already queued"]
    with suppress(asyncio.CancelledError):
        await original
    assert original.cancelled()
    await agent.stop_message_loop()


def test_interrupt_instruction_is_prepended_to_existing_system_message():
    original = {
        "messages": [
            SystemMessage(content="Original agent instructions"),
            HumanMessage(content="Old task"),
            HumanMessage(content="New direction"),
        ]
    }

    updated = BaseAgent._prepend_system_message(
        original, DEFAULT_INTERRUPT_MESSAGE
    )

    assert original["messages"][0].content == "Original agent instructions"
    assert len(updated["messages"]) == 3
    assert isinstance(updated["messages"][0], SystemMessage)
    assert updated["messages"][0].content.startswith(DEFAULT_INTERRUPT_MESSAGE)
    assert "Original agent instructions" in updated["messages"][0].content


@pytest.mark.asyncio
async def test_default_interrupt_handler_sends_system_message_to_graph(
    chat_model, tmp_path
):
    class CaptureAgent(BaseAgent):
        def __init__(self, *args, **kwargs):
            self.seen = None
            super().__init__(*args, enable_metrics=False, **kwargs)

        def capture(self, state):
            self.seen = state
            return state

        def _build_graph(self):
            self.add_node(self.capture, "capture")
            self.graph.set_entry_point("capture")
            self.graph.set_finish_point("capture")

    agent = CaptureAgent(llm=chat_model, workspace=tmp_path)

    result = await agent.interrupt("take a new direction")

    assert agent.seen is not None
    assert isinstance(agent.seen["messages"][0], SystemMessage)
    assert agent.seen["messages"][0].content == DEFAULT_INTERRUPT_MESSAGE
    assert isinstance(agent.seen["messages"][1], HumanMessage)
    assert result["messages"][1].content == "take a new direction"
    await agent.stop_message_loop()


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
