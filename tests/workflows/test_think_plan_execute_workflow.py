from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ursa.agents.planning_agent import Plan, PlanStep
from ursa.workflows import planning_execution_workflow as pew
from ursa.workflows import think_plan_execute as tpe


class StubPlanner:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.prompts: list[str] = []
        self.invoke_kwargs: list[dict] = []

    def format_query(self, prompt: str):
        self.prompts.append(prompt)
        return {"messages": [HumanMessage(content=prompt)]}

    async def _ainvoke(self, _state, **kwargs):
        self.invoke_kwargs.append(kwargs)
        return {"plan": self.plan}


class StubExecutor:
    def __init__(self):
        self.prompts: list[str] = []
        self.step_counter = 0
        self.invoke_kwargs: list[dict] = []

    def format_query(self, prompt: str):
        self.prompts.append(prompt)
        return {"messages": [HumanMessage(content=prompt)]}

    async def _ainvoke(self, _state, **kwargs):
        self.invoke_kwargs.append(kwargs)
        self.step_counter += 1
        return {
            "messages": [AIMessage(content=f"step-{self.step_counter}-done")]
        }

    def format_result(self, state):
        return state["messages"][-1].text


class StubHypothesizer:
    def __init__(self, hypothesis: str):
        self.hypothesis = hypothesis
        self.prompts: list[str] = []
        self.invoke_kwargs: list[dict] = []

    def format_query(self, prompt: str):
        self.prompts.append(prompt)
        return {"question": prompt}

    async def _ainvoke(self, _state, **kwargs):
        self.invoke_kwargs.append(kwargs)
        return {"solution": self.hypothesis}

    def format_result(self, state):
        return state["solution"]


def _plan_with_two_steps() -> Plan:
    return Plan(
        steps=[
            PlanStep(
                name="Collect facts",
                description="Gather task constraints.",
                requires_code=False,
                expected_outputs=["constraints list"],
                success_criteria=["constraints identified"],
            ),
            PlanStep(
                name="Produce answer",
                description="Deliver the final result.",
                requires_code=False,
                expected_outputs=["final answer"],
                success_criteria=["answer is complete"],
            ),
        ]
    )


class FakeCompiledGraph:
    def __init__(self, nodes: dict[str, object]):
        self.nodes = nodes


class FakeStateGraph:
    def __init__(self, state_schema=None):
        self.state_schema = state_schema
        self.nodes: dict[str, object] = {}
        self.conditional_edges: dict[str, object] = {}
        self.edges: list[tuple[str, str]] = []

    def add_node(self, name: str, fn):
        self.nodes[name] = fn

    def add_conditional_edges(self, source: str, fn):
        self.conditional_edges[source] = fn

    def add_edge(self, source: str, target: str):
        self.edges.append((source, target))

    def compile(self, checkpointer=None):
        _ = checkpointer
        return FakeCompiledGraph(nodes=self.nodes)


@pytest.mark.asyncio
async def test_plan_execute_workflow_hands_off_between_planner_and_executor(
    monkeypatch,
):
    monkeypatch.setattr(pew, "StateGraph", FakeStateGraph)
    planner = StubPlanner(plan=_plan_with_two_steps())
    executor = StubExecutor()
    workflow = pew.plan_execute_workflow(planner=planner, executor=executor)

    create_plan = workflow.nodes["create_plan"]
    execute_step = workflow.nodes["execute_step"]
    state = await create_plan(
        {"task": "Solve the task"},
        None,
        {"configurable": {"thread_id": "t-1"}},
    )
    first_step = await execute_step(
        state, None, {"configurable": {"thread_id": "t-1"}}
    )
    second_step = await execute_step(
        {**state, **first_step},
        None,
        {"configurable": {"thread_id": "t-1"}},
    )
    assert len(planner.prompts) == 1
    assert planner.prompts[0] == "Solve the task"
    assert (
        planner.invoke_kwargs[0]["configurable"]["checkpoint_ns"] == "planner"
    )
    assert len(executor.prompts) == 2
    assert "Previous-step summary" not in executor.prompts[0]
    assert "Previous-step summary:\nstep-1-done" in executor.prompts[1]
    assert (
        executor.invoke_kwargs[0]["configurable"]["checkpoint_ns"] == "executor"
    )
    assert second_step["messages"][-1].text == "step-2-done"


@pytest.mark.asyncio
async def test_think_plan_execute_workflow_threads_hypothesis_into_planning(
    monkeypatch,
):
    monkeypatch.setattr(tpe, "StateGraph", FakeStateGraph)
    hypothesizer = StubHypothesizer(hypothesis="Try decomposition first.")
    planner = StubPlanner(plan=_plan_with_two_steps())
    executor = StubExecutor()
    workflow = tpe.think_plan_execute_workflow(
        hypothesizer=hypothesizer,
        planner=planner,
        executor=executor,
    )

    create_hypothesis = workflow.nodes["create_hypothesis"]
    create_plan = workflow.nodes["create_plan"]
    execute_step = workflow.nodes["execute_step"]
    hypothesis_state = await create_hypothesis(
        {"task": "Handle this problem"},
        None,
        {"configurable": {"thread_id": "t-2"}},
    )
    planning_state = await create_plan(
        hypothesis_state,
        None,
        {"configurable": {"thread_id": "t-2"}},
    )
    current_state = {**hypothesis_state, **planning_state}
    first_step = await execute_step(
        current_state,
        None,
        {"configurable": {"thread_id": "t-2"}},
    )
    final_state = await execute_step(
        {**current_state, **first_step},
        None,
        {"configurable": {"thread_id": "t-2"}},
    )
    assert hypothesizer.prompts == ["Handle this problem"]
    assert (
        hypothesizer.invoke_kwargs[0]["configurable"]["checkpoint_ns"]
        == "hypothesizer"
    )
    assert len(planner.prompts) == 1
    assert "Working hypothesis:\nTry decomposition first." in planner.prompts[0]
    assert len(executor.prompts) == 2
    assert (
        executor.invoke_kwargs[0]["configurable"]["checkpoint_ns"] == "executor"
    )
    assert (
        "Hypothesis to test or refine:\nTry decomposition first."
        in executor.prompts[0]
    )
    assert final_state["messages"][-1].text == "step-2-done"


@pytest.mark.asyncio
async def test_plan_execute_workflow_uses_last_message_as_task_fallback(
    monkeypatch,
):
    monkeypatch.setattr(pew, "StateGraph", FakeStateGraph)
    planner = StubPlanner(plan=_plan_with_two_steps())
    executor = StubExecutor()
    workflow = pew.plan_execute_workflow(planner=planner, executor=executor)
    create_plan = workflow.nodes["create_plan"]

    await create_plan(
        {"messages": [HumanMessage(content="Task from message")]}, None
    )
    assert planner.prompts[0] == "Task from message"


@pytest.mark.asyncio
async def test_plan_execute_workflow_requires_task_or_message(monkeypatch):
    monkeypatch.setattr(pew, "StateGraph", FakeStateGraph)
    planner = StubPlanner(plan=_plan_with_two_steps())
    executor = StubExecutor()
    workflow = pew.plan_execute_workflow(planner=planner, executor=executor)
    create_plan = workflow.nodes["create_plan"]

    with pytest.raises(ValueError, match="requires a task or message input"):
        await create_plan({}, None)
