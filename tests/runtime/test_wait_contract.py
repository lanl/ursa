"""The same wait semantics must hold for local and process-hosted agents."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain.tools import ToolException

from tests.agents.test_async_messages import MailboxAgent, hierarchy_tool
from tests.runtime.fixtures import runtime_request, set_hierarchy
from ursa.messaging import AgentMessage


@pytest.fixture(params=["local", "managed"])
async def team(request, runtime_factory, chat_model, tmp_path):
    paths = ["root", "root/first", "root/second", "root/third"]
    tasks = []
    root = MailboxAgent(llm=chat_model, workspace=tmp_path)
    for path in paths[1:]:
        root.spawn_subagent(path.split("/")[1])
    runtime, _ = runtime_factory()
    set_hierarchy(runtime, {path: path for path in paths})
    runtime._active_root_id = "root"
    managed = request.param == "managed"

    def working(path, active=True):
        if managed:
            runtime._mark_working(path, 1 if active else -1)
        else:
            root.resolve_agent(path)._current_agent_message = (
                AgentMessage(content="working", sender="user", recipient=path)
                if active
                else None
            )

    def waiting(path="root"):
        if managed:
            states = {row.path: row.state for row in runtime.hierarchy_status()}
            return states[path] == "Waiting"
        return root.resolve_agent(path).is_waiting

    async def wait(sender, target, *, timeout=1, wait_all=False):
        options = dict(
            target=target, timeout_seconds=timeout, wait_all=wait_all
        )
        if managed:
            return await runtime_request(runtime, sender, "wait_for", **options)
        return await hierarchy_tool(
            root.resolve_agent(sender), "wait_for"
        ).ainvoke(options)

    async def begin(sender, target, **options):
        task = asyncio.create_task(wait(sender, target, **options))
        tasks.append(task)
        async with asyncio.timeout(1):
            while not waiting(sender) and not task.done():
                await asyncio.sleep(0)
        assert waiting(sender)
        return task

    async def wake():
        if managed:
            await runtime_request(runtime, "root", "wake_waiters")
        else:
            root._wake_message_waiters()

    async def send():
        if managed:
            assert (
                await runtime_request(
                    runtime,
                    "root/third",
                    "invoke",
                    target="root",
                    message="new direction",
                )
                == "Message queued for root."
            )
            assert not runtime._background_agent_tasks
        else:
            await root.post_message("new direction", sender="root/third")
            assert not root.received

    yield SimpleNamespace(
        working=working,
        waiting=waiting,
        wait=wait,
        begin=begin,
        wake=wake,
        send=send,
        route_error=KeyError if managed else ToolException,
    )
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await root.stop_message_loop(cancel=True)


@pytest.mark.parametrize("wait_all", [False, True])
async def test_list_wait_finishes_on_any_or_all_targets(team, wait_all):
    team.working("root/first")
    team.working("root/second")
    waiting = await team.begin("root", ["first", "second"], wait_all=wait_all)
    team.working("root/first", False)
    await asyncio.sleep(0)
    if wait_all:
        assert not waiting.done()
        team.working("root/second", False)
        expected = "All agents finished their work: root/first, root/second."
    else:
        expected = "Agent root/first finished its work."
    assert await waiting == expected
    assert not team.waiting()


@pytest.mark.parametrize(
    ("target", "wait_all", "expected"),
    [
        ("first", False, "Agent root/first did not finish within 0 seconds."),
        (
            ["first", "second"],
            False,
            "No agents finished within 0 seconds: root/first, root/second.",
        ),
        (
            ["first", "second"],
            True,
            "Not all agents finished within 0 seconds: root/first, root/second.",
        ),
    ],
)
async def test_timeout_removes_wait_dependencies(
    team, target, wait_all, expected
):
    for path in ("root", "root/first", "root/second"):
        team.working(path)
    assert (
        await team.wait("root", target, timeout=0, wait_all=wait_all)
        == expected
    )
    assert not team.waiting()
    reverse = await team.begin("root/first", "root")
    assert not reverse.done()


@pytest.mark.parametrize("indirect", [False, True])
async def test_cycles_are_ready_only_to_their_dependencies(team, indirect):
    for path in ("root", "root/first", "root/second"):
        team.working(path)
    target = "first" if indirect else ["first", "second"]
    root_wait = await team.begin("root", target, wait_all=not indirect)
    waits = [root_wait]
    if indirect:
        waits.append(await team.begin("root/first", "second"))
    unrelated = await team.begin("root/third", "root")
    assert not unrelated.done()
    sender = "root/second" if indirect else "root/first"
    relation = "indirectly" if indirect else "directly"
    assert await team.wait(sender, "root") == (
        f"Agent root is ready for this wait because it is waiting on you {relation}."
    )
    for task in waits:
        task.cancel()
    await asyncio.gather(*waits, return_exceptions=True)
    assert not team.waiting()
    assert not team.waiting("root/first")
    reverse = await team.begin("root/first", "root")
    assert not reverse.done()
    team.working("root", False)
    assert await unrelated == "Agent root finished its work."
    assert await reverse == "Agent root finished its work."


async def test_wake_and_message_delivery_end_the_wait_without_replay(team):
    team.working("root/first")
    waiting = await team.begin("root", "first")
    await team.wake()
    assert await waiting == "Stopped waiting because a new message arrived."
    waiting = await team.begin("root", "first")
    await team.send()
    assert await waiting == (
        "Received a message while waiting.\n"
        "Sender: root/third\nRecipient: root\n"
        "Exact message content:\nnew direction"
    )
    assert not team.waiting()


async def test_invalid_list_rejects_wait_before_registering_dependencies(team):
    with pytest.raises(team.route_error, match="Unknown agent path"):
        await team.wait("root", ["first", "missing"])
    assert not team.waiting()
    # A ready ancestor is a valid target after the rejected batch.
    assert await team.wait("root/first", "root") == (
        "Agent root has already finished its work."
    )
