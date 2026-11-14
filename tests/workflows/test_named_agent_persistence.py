"""Named workflow members can be resumed using the TUI/dashboard thread."""

import asyncio
from itertools import repeat
from typing import TypedDict

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from tests.composite_helpers import CompositeFakeModel, request_text
from ursa import security
from ursa.agents import ChatAgent, ExecutionAgent
from ursa.environments import AgentSymposiumEnvironment, AgentTeamEnvironment
from ursa.workflows import (
    Hypothesis,
    HypothesisOrchestratorWorkflow,
    HypothesisSymposiumWorkflow,
)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")


def model():
    return CompositeFakeModel(
        messages=(AIMessage(content="evidence recorded") for _ in repeat(None))
    )


def assert_canonical_checkpoints(agent):
    rows = agent.checkpointer.conn.execute(
        "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints"
    ).fetchall()
    assert rows == [("ursa", "")]


class ModelEvents(BaseCallbackHandler):
    def __init__(self):
        self.metadata = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.metadata.append(kwargs.get("metadata", {}))


class FixedHypothesesWorkflow(HypothesisSymposiumWorkflow):
    def build_hypotheses(self, state, config):
        return [Hypothesis(1, "A testable cause")]

    async def update_hypothesis_space(self, state, config):
        return {"summary": state["synthesis"]}


async def test_persistent_symposium_resumes_named_agents_across_parent_threads(
    tmp_path,
):
    llm = model()
    seed = ChatAgent(
        llm,
        agent_name="analyst",
        thread_id="ursa",
        workspace=tmp_path / "seed",
        enable_metrics=False,
    )
    await seed.ainvoke("Remember the original measurement")
    seed.close()
    environment = AgentSymposiumEnvironment(
        llm,
        name="research",
        workspace=tmp_path / "symposium",
        members=[
            {
                "name": name,
                "role": name,
                "agent": "ChatAgent",
                "config": {
                    "thread_id": "ignored-member-thread",
                    "enable_metrics": False,
                },
            }
            for name in ("analyst", "skeptic")
        ],
        organizer={
            "name": "organizer",
            "role": "organizer",
            "agent": "ChatAgent",
            "config": {"enable_metrics": False},
        },
    )
    agents = [*environment.members.values(), environment.organizer]
    assert all(agent.thread_id == "ursa" for agent in agents)
    assert len({agent.den for agent in agents}) == 3
    saver = InMemorySaver()
    workflows = [
        FixedHypothesesWorkflow(
            llm,
            symposium=environment,
            checkpointer=saver,
            thread_id=thread,
            workspace=tmp_path / thread,
            visualize=False,
            enable_metrics=False,
        )
        for thread in ("parent-a", "parent-b")
    ]
    events = ModelEvents()
    try:
        for index, workflow in enumerate(workflows):
            result = await workflow.ainvoke(
                f"investigation {index}",
                config={
                    "callbacks": [events],
                    "metadata": {"experiment": "persistence"},
                },
            )
            assert "FAILED" not in str(result["symposium_result"])
        assert events.metadata
        assert all(
            event["experiment"] == "persistence" for event in events.metadata
        )
        assert all(event["thread_id"] == "ursa" for event in events.metadata)
        for workflow, query in zip(
            workflows, ("investigation 0", "investigation 1")
        ):
            assert (
                workflow._graph.get_state(workflow._base_config(None)).values[
                    "query"
                ]
                == query
            )
        for agent in agents:
            assert_canonical_checkpoints(agent)
    finally:
        for agent in [
            *agents,
            *(workflow.hypothesizer for workflow in workflows),
        ]:
            await agent.aclose()
            agent.close()

    # Recreate the agent exactly as a later named-agent session would.
    resumed = ChatAgent(
        llm,
        agent_name="analyst",
        thread_id="ursa",
        workspace=tmp_path / "later-session",
        enable_metrics=False,
    )
    try:
        await resumed.ainvoke("Continue the research")
        prompt = request_text(llm.plain_requests[-1])
        assert "Remember the original measurement" in prompt
        assert "investigation 0" in prompt
        assert "investigation 1" in prompt
        assert "Continue the research" in prompt
        assert_canonical_checkpoints(resumed)
    finally:
        resumed.close()


class DelegationState(TypedDict):
    label: str
    task: str
    result: str


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("forward_config", [False, True])
async def test_investigators_use_own_files_and_resume_on_ursa(
    tmp_path, asynchronous, forward_config
):
    llm = model()
    workflow = HypothesisOrchestratorWorkflow(
        llm,
        workspace=tmp_path / "investigators",
        thread_id="parent-thread",
        visualize=False,
        trace=False,
        enable_metrics=False,
    )
    registry = workflow._build_registry()

    def spawn_tool(config):
        workflow._run_config = config if forward_config else None
        return workflow._make_spawn_tools(registry)[0]

    def delegate(state: DelegationState, config: RunnableConfig):
        result = spawn_tool(config).invoke(
            {"hypothesis_label": state["label"], "task": state["task"]},
            config=config,
        )
        return {"result": result}

    async def adelegate(state: DelegationState, config: RunnableConfig):
        result = await spawn_tool(config).ainvoke(
            {"hypothesis_label": state["label"], "task": state["task"]},
            config=config,
        )
        return {"result": result}

    graph = StateGraph(DelegationState)
    graph.add_node("delegate", RunnableLambda(delegate, afunc=adelegate))
    graph.add_edge(START, "delegate")
    graph.add_edge("delegate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    try:
        for label in ("H1", "H2"):
            for phase in ("investigation", "review"):
                inputs = {"label": label, "task": f"{label} private {phase}"}
                config = {"configurable": {"thread_id": f"parent-{phase}"}}
                if asynchronous:
                    await compiled.ainvoke(inputs, config)
                else:
                    compiled.invoke(inputs, config)
        agents = [registry.get_or_create(label) for label in ("H1", "H2")]
        assert agents[0].den != agents[1].den
        for agent in agents:
            assert agent.thread_id == "ursa"
            assert_canonical_checkpoints(agent)
    finally:
        for agent in [workflow.hypothesizer, *registry._agents.values()]:
            await agent.aclose()
            agent.close()

    resumed = ExecutionAgent(
        llm,
        agent_name=registry.identities["h1"],
        thread_id="ursa",
        workspace=tmp_path / "later-session",
        enable_metrics=False,
    )
    try:
        resumed.invoke("Continue H1")
        prompt = request_text(llm.plain_requests[-1])
        assert "H1 private investigation" in prompt
        assert "H1 private review" in prompt
        assert "H2 private" not in prompt
        assert "Continue H1" in prompt
        assert_canonical_checkpoints(resumed)
    finally:
        resumed.close()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_named_team_delegation_uses_canonical_thread(
    tmp_path, asynchronous
):
    llm = model()
    team = AgentTeamEnvironment(
        llm,
        name="team",
        workspace=tmp_path / "team",
        pi={
            "name": "pi",
            "role": "lead",
            "agent": "ChatAgent",
            "config": {"enable_metrics": False},
        },
        members=[
            {
                "name": "analyst",
                "role": "analysis",
                "agent": "ChatAgent",
                "config": {"enable_metrics": False},
            }
        ],
    )
    from ursa.environments.base import (
        bind_current_environment_config,
        reset_current_environment_config,
    )

    token = bind_current_environment_config({
        "configurable": {"thread_id": "parent-thread"}
    })
    try:
        delegation = team._make_delegation_tool(team.config.members[0])
        if asynchronous:
            await delegation.ainvoke({"task": "Analyze the data"})
        else:
            await asyncio.to_thread(
                delegation.invoke, {"task": "Analyze the data"}
            )
        assert team.pi.thread_id == team.members["analyst"].thread_id == "ursa"
        assert_canonical_checkpoints(team.members["analyst"])
    finally:
        reset_current_environment_config(token)
        for agent in [team.pi, *team.members.values()]:
            await agent.aclose()
            agent.close()
