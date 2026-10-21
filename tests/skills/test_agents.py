import asyncio

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.agents.utils import (
    RecordingChatModel,
    ScriptedRecordingChatModel,
    assert_requests_provider_valid,
)
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent
from ursa.agents.deep_review_agent import DeepReviewAgent
from ursa.agents.execution_agent import ExecutionAgent
from ursa.skills import SkillScope
from ursa.skills.runtime import current_run, skill_run

BODY = "SKILL_BODY_SENTINEL"


def new_agent(tmp_path, llm=None, **kwargs):
    return ChatAgent(
        llm=llm or RecordingChatModel(),
        workspace=tmp_path,
        enable_metrics=False,
        **kwargs,
    )


def contents(messages):
    return "\n".join(message.text for message in messages)


def test_explicit_skill_is_transient_and_catalog_is_lazy(tmp_path, make_skill):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model)
    result = agent.invoke("Write a report")
    assert "lab-report" in contents(model.calls[-1])
    assert BODY not in contents(model.calls[-1])
    result = agent.invoke("$lab-report Write a report")
    assert BODY in contents(model.calls[-1])
    assert BODY not in contents(result["messages"])
    agent.invoke({
        "messages": [*result["messages"], HumanMessage("What time is it?")]
    })
    assert BODY not in contents(model.calls[-1])
    assert current_run(agent) is None
    assert_requests_provider_valid(model.calls)


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
async def test_contextual_tool_loading(mode, tmp_path, make_skill):
    make_skill(tmp_path)
    model = ScriptedRecordingChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "load-1",
                        "name": "load_skill",
                        "args": {"name": "lab-report"},
                    }
                ],
            ),
            AIMessage(content="finished"),
        ]
    )
    agent = new_agent(tmp_path, model)
    if mode == "async":
        result = await agent.ainvoke("Write a laboratory report")
    elif mode == "stream":
        result = list(agent.stream("Write a laboratory report"))[-1]
    else:
        result = agent.invoke("Write a laboratory report")
    assert BODY not in contents(model.calls[0])
    assert BODY in contents(model.calls[1])
    assert BODY not in str(result)
    assert_requests_provider_valid(model.calls)


def test_unknown_skill_fails_before_model_and_cleans_up(tmp_path):
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model)
    with pytest.raises(ValueError, match="Unknown skill"):
        agent.invoke("$missing do something")
    assert model.calls == []
    assert current_run(agent) is None
    agent.invoke("ordinary task")
    assert len(model.calls) == 1


def test_compaction_preserves_active_skill_without_checkpointing_body(
    tmp_path, make_skill
):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = new_agent(
        tmp_path, model, tokens_before_summarize=1, messages_to_keep=0
    )
    result = agent.invoke({
        "messages": [
            SystemMessage("Original role"),
            HumanMessage("old request " * 50),
            AIMessage("old response " * 50),
            HumanMessage("$lab-report report " * 50),
        ]
    })
    assert len(model.calls) >= 2
    assert BODY in contents(model.calls[-1])
    assert "Original role" in contents(model.calls[-1])
    assert BODY not in contents(result["messages"])
    assert_requests_provider_valid(model.calls)


def test_checkpoint_reuse_does_not_reactivate_skill(tmp_path, make_skill):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model, checkpointer=InMemorySaver())
    agent.invoke("$lab-report first task")
    agent.invoke("Unrelated next task")
    assert BODY not in contents(model.calls[-1])
    checkpoint = agent.compiled_graph.get_state(agent.build_config())
    assert BODY not in contents(checkpoint.values["messages"])


async def test_parallel_requests_and_nested_runs_are_isolated(
    tmp_path, make_skill
):
    make_skill(tmp_path, "first", body="FIRST_BODY")
    make_skill(tmp_path, "second", body="SECOND_BODY")
    owner = object()
    scope = SkillScope(workspace=tmp_path)

    async def run(name):
        with skill_run(owner, scope, f"${name}", None):
            await asyncio.sleep(0)
            assert list(current_run(owner).active) == [name]
            with skill_run(owner, scope, "plain request", None):
                assert not current_run(owner).active
            assert list(current_run(owner).active) == [name]
        assert current_run(owner) is None

    await asyncio.gather(run("first"), run("second"))


def test_resources_are_relative_and_request_scoped(tmp_path, make_skill):
    folder = make_skill(tmp_path)
    (folder / "references").mkdir()
    (folder / "references" / "guide.md").write_text("RESOURCE_SENTINEL")
    agent = new_agent(tmp_path)
    with agent._skill_run("$lab-report task", None):
        tool = agent.tools["read_skill_resource"]
        assert "Loaded resource" in tool.invoke({
            "name": "lab-report",
            "path": "references/guide.md",
        })
        assert "RESOURCE_SENTINEL" in current_run(agent).prompt()
        assert "within the skill directory" in tool.invoke({
            "name": "lab-report",
            "path": "../../outside",
        })
    with agent._skill_run("new task", None):
        assert not current_run(agent).resources


def test_current_request_override_ignores_flattened_history(
    tmp_path, make_skill
):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model)
    agent.invoke(
        "User: $lab-report old task\nUser: new task",
        config={"configurable": {"ursa_skill_request": "new task"}},
    )
    assert BODY not in contents(model.calls[-1])


@pytest.mark.parametrize("agent_type", [ExecutionAgent, DeepReviewAgent])
def test_supported_specialized_tool_agents(tmp_path, make_skill, agent_type):
    make_skill(tmp_path)
    model = RecordingChatModel(
        structured_factory=lambda schema: schema(
            is_complete=True, reason="done"
        )
    )
    kwargs = {"max_iterations": 1} if agent_type is DeepReviewAgent else {}
    agent = agent_type(
        llm=model, workspace=tmp_path, enable_metrics=False, **kwargs
    )
    agent.invoke("$lab-report Write a report")
    assert BODY in contents(model.calls[0])
    assert BODY in contents(model.calls[-1])
    assert_requests_provider_valid(model.calls)


def test_non_tool_agent_is_unchanged(tmp_path, make_skill):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = BasicChatAgent(llm=model, workspace=tmp_path, enable_metrics=False)
    agent.invoke("$lab-report Write a report")
    assert "available_skills" not in contents(model.calls[-1])
    assert BODY not in contents(model.calls[-1])


def test_activation_event_is_emitted_once(tmp_path, make_skill):
    make_skill(tmp_path)
    events = []

    class Handler(BaseCallbackHandler):
        def on_custom_event(self, name, data, **kwargs):
            events.append(data)

    agent = new_agent(tmp_path)
    agent.invoke(
        "$lab-report $lab-report report", config={"callbacks": [Handler()]}
    )
    assert (
        len([
            event
            for event in events
            if event.get("stage") == "skill_activation"
        ])
        == 1
    )


@pytest.mark.parametrize(
    "message",
    [
        ("user", "$lab-report report"),
        {"role": "user", "content": "$lab-report report"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "$lab-report report"}],
        },
    ],
)
def test_langchain_input_message_formats(tmp_path, make_skill, message):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model)
    agent.invoke({"messages": [message]})
    assert BODY in contents(model.calls[-1])


def test_multiple_explicit_skills_and_catalog_refresh(tmp_path, make_skill):
    make_skill(tmp_path)
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model)
    assert len(agent.list_skills().skills) == 1
    make_skill(tmp_path, "second-skill", body="SECOND_BODY")
    agent.invoke("$lab-report $second-skill $lab-report report")
    outgoing = contents(model.calls[-1])
    assert outgoing.count(BODY) == 1
    assert outgoing.count("SECOND_BODY") == 1
    assert len(agent.list_skills().skills) == 2


def test_tool_resource_loading_through_graph(tmp_path, make_skill):
    folder = make_skill(tmp_path)
    (folder / "guide.md").write_text("GRAPH_RESOURCE")
    model = ScriptedRecordingChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "read-1",
                        "name": "read_skill_resource",
                        "args": {"name": "lab-report", "path": "guide.md"},
                    }
                ],
            ),
            AIMessage(content="finished"),
        ]
    )
    agent = new_agent(tmp_path, model)
    result = agent.invoke("$lab-report report")
    assert "GRAPH_RESOURCE" in contents(model.calls[-1])
    assert "GRAPH_RESOURCE" not in contents(result["messages"])
    assert_requests_provider_valid(model.calls)


def test_interleaved_streams_do_not_leak_context(tmp_path, make_skill):
    make_skill(tmp_path, "first", body="FIRST_BODY")
    make_skill(tmp_path, "second", body="SECOND_BODY")
    model = RecordingChatModel()
    agent = new_agent(tmp_path, model)
    first = agent.stream("$first task")
    second = agent.stream("$second task")
    try:
        next(first)
        assert current_run(agent) is None
        assert "FIRST_BODY" in contents(model.calls[-1])
        next(second)
        assert current_run(agent) is None
        assert "SECOND_BODY" in contents(model.calls[-1])
        assert "FIRST_BODY" not in contents(model.calls[-1])
    finally:
        second.close()
        first.close()
    assert current_run(agent) is None
