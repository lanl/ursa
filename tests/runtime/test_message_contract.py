"""Message routing has the same contract in local and managed hierarchies."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain.tools import ToolException

from tests.agents.test_async_messages import MailboxAgent, hierarchy_tool
from tests.runtime.fixtures import runtime_request, set_hierarchy
from ursa.agents.base import AgentMessage


@pytest.fixture(params=["local", "managed"])
async def routing(request, runtime_factory, chat_model, tmp_path, monkeypatch):
    root = MailboxAgent(llm=chat_model, workspace=tmp_path)
    analyst = root.spawn_subagent("analyst")
    analyst.spawn_subagent("specialist")
    root.spawn_subagent("reviewer")
    members = {agent.agent_path: agent for agent in root.walk_agents()}
    runtime, manager = runtime_factory()
    identities = {path: f"id:{path}" for path in members}
    set_hierarchy(
        runtime, {identity: path for path, identity in identities.items()}
    )
    managed = request.param == "managed"
    received, logged = [], []

    async def invoke(identity, message, *, config, **kwargs):
        metadata = config["metadata"]
        recipient = metadata["ursa_message_recipient"]
        assert identities[recipient] == identity
        received.append((metadata["ursa_message_sender"], recipient, message))
        return "reply"

    monkeypatch.setattr(manager, "invoke", invoke)

    class Callback:
        async def on_hierarchy_message(self, sender, targets, message):
            logged.extend((sender, target, message) for target in targets)

    callback = Callback()

    async def send(sender, target, message="Share your progress."):
        if managed:
            multiple = isinstance(target, list)
            result = await runtime_request(
                runtime,
                identities[sender],
                "invoke_many" if multiple else "invoke",
                callbacks=(callback,),
                message=message,
                **({"targets": target} if multiple else {"target": target}),
            )
            return (
                "\n\n".join(row["result"] for row in result)
                if multiple
                else result
            )
        agent = members[sender]
        agent._current_agent_message = AgentMessage(
            content="coordinate",
            sender="user",
            recipient=sender,
            config={"callbacks": [callback]},
        )
        try:
            return await hierarchy_tool(agent, "send_message").ainvoke({
                "target": target,
                "message": message,
            })
        finally:
            agent._current_agent_message = None

    async def drain():
        if managed:
            while runtime._background_agent_tasks:
                await asyncio.gather(*tuple(runtime._background_agent_tasks))
        else:
            await asyncio.gather(
                *(agent.wait_for_messages() for agent in members.values())
            )
            received[:] = [
                (message.sender, message.recipient, message.content)
                for agent in members.values()
                for message in agent.received
            ]

    def expand(sender, target):
        if managed:
            return [
                path
                for _identity, path in runtime._resolve_message_targets(
                    identities[sender], [target]
                )
            ]
        return [
            agent.agent_path
            for agent in members[sender].resolve_message_targets(target)
        ]

    def stopping(path):
        if managed:
            runtime._stopping_subtrees.add((identities["root"], path))
        else:
            members[path]._hierarchy_stopping = True

    yield SimpleNamespace(
        send=send,
        drain=drain,
        expand=expand,
        stopping=stopping,
        received=received,
        logged=logged,
    )
    await root.stop_message_loop(descendants=True, cancel=True)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("@reports", ["root/analyst/specialist"]),
        ("@REPORTS", ["root/analyst/specialist"]),
        ("@peers", ["root/reviewer"]),
        ("@supervisor", ["root"]),
        ("@everyone", ["root", "root/analyst/specialist", "root/reviewer"]),
    ],
)
async def test_groups_expand_in_hierarchy_order_without_the_sender(
    routing, target, expected
):
    assert routing.expand("root/analyst", target) == expected


async def test_mixed_group_and_explicit_routes_deduplicate_in_input_order(
    routing,
):
    recipients = ["root/reviewer", "root/analyst/specialist", "root"]
    response = await routing.send(
        "root/analyst",
        [
            "@peers",
            "root/reviewer",
            "@reports",
            "@supervisor",
            "root/analyst",
        ],
    )
    await routing.drain()
    positions = []
    for path in recipients:
        acknowledgement = f"Message queued for {path}."
        assert response.count(acknowledgement) == 1
        positions.append(response.index(acknowledgement))
    assert positions == sorted(positions)
    expected = [
        ("root/analyst", path, "Share your progress.") for path in recipients
    ]
    assert routing.logged == expected
    assert sorted(routing.received) == sorted(expected)


@pytest.mark.parametrize(
    ("sender", "target", "error"),
    [
        ("root", "@supervisor", "has no recipients"),
        ("root", "@peers", "has no recipients"),
        ("root/reviewer", "@reports", "has no recipients"),
        ("root", "@unknown", "Unknown message target group"),
        ("root", "@parent", "Unknown message target group"),
        ("root", "@siblings", "Unknown message target group"),
        ("root", "@children", "Unknown message target group"),
        ("root", "missing", "Unknown agent path"),
    ],
)
async def test_invalid_route_rejects_the_whole_batch_before_delivery(
    routing, sender, target, error
):
    with pytest.raises((ToolException, ValueError, KeyError), match=error):
        await routing.send(sender, ["root/analyst", target, "root/reviewer"])
    await routing.drain()
    assert routing.logged == routing.received == []


async def test_stopping_recipient_rejects_the_whole_group(routing):
    routing.stopping("root/reviewer")
    with pytest.raises((ToolException, ValueError), match="is stopping"):
        await routing.send("root/analyst", "@everyone")
    await routing.drain()
    assert routing.logged == routing.received == []
