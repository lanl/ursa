from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from tests.composite_helpers import (
    CompositeFakeModel,
    request_text,
)
from ursa.agents.planning_agent import Plan, PlanStep
from ursa.agents.planning_execution_agent import PlanningExecutionAgent
from ursa.util import Checkpointer
from ursa.workflows.planning_execution_workflow import (
    PlanningExecutorWorkflow,
)


def test_composite_uses_native_subgraphs_and_hands_off_step_context(tmp_path):
    model = CompositeFakeModel()
    agent = PlanningExecutionAgent(
        llm=model,
        workspace=tmp_path,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    state = agent.invoke("Solve this task")

    assert agent.format_result(state) == "step-2-summary"
    assert state["step_idx"] == 2
    assert state["step_results"] == ["step-1-summary", "step-2-summary"]
    assert [name for name, _graph in agent.compiled_graph.get_subgraphs()] == [
        "planner",
        "executor",
    ]

    executor_requests = [
        request_text(messages)
        for messages in model.plain_requests
        if isinstance(messages[-1], HumanMessage)
        and str(messages[-1].content).startswith(
            "You are contributing to the larger solution"
        )
    ]
    assert len(executor_requests) == 2
    assert "Previous-step summary" not in executor_requests[0]
    assert "Previous-step summary:\nstep-1-summary" in executor_requests[1]
    # The explicit summary is the only cross-step transcript context.
    assert "step-1-work" not in executor_requests[1]

    planner_request = next(
        request_text(messages)
        for schema, messages in model.structured_requests
        if schema == "Plan"
    )
    assert "Task:\nSolve this task" in planner_request
    agent.close()


def test_parent_checkpointer_owns_isolated_native_subgraph_namespaces(
    tmp_path: Path,
):
    checkpointer = Checkpointer.from_workspace(tmp_path)
    agent = PlanningExecutionAgent(
        llm=CompositeFakeModel(),
        workspace=tmp_path,
        checkpointer=checkpointer,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    agent.invoke(
        "Solve this task",
        config={"configurable": {"thread_id": "persistent-thread"}},
    )

    namespaces = checkpointer.conn.execute(
        "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
        ("persistent-thread",),
    ).fetchall()
    assert {row[0] for row in namespaces} == {"", "planner", "executor"}
    assert agent.checkpointer is checkpointer
    agent.close()


@pytest.mark.asyncio
async def test_composite_ainvoke_persists_the_same_native_namespaces(tmp_path):
    checkpointer = await Checkpointer.async_from_workspace(tmp_path)
    agent = PlanningExecutionAgent(
        llm=CompositeFakeModel(),
        workspace=tmp_path,
        checkpointer=checkpointer,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    state = await agent.ainvoke(
        "Solve this asynchronously",
        config={"configurable": {"thread_id": "async-thread"}},
    )

    assert agent.format_result(state) == "step-2-summary"
    assert state["step_idx"] == 2
    cursor = await checkpointer.conn.execute(
        "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
        ("async-thread",),
    )
    assert {row[0] for row in await cursor.fetchall()} == {
        "",
        "planner",
        "executor",
    }
    await checkpointer.conn.close()


def test_parent_workspace_context_reaches_native_executor_subgraph(tmp_path):
    workspace = tmp_path / "workspace"
    source = tmp_path / "source"
    source.mkdir()
    agent = PlanningExecutionAgent(
        llm=CompositeFakeModel(),
        workspace=workspace,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    state = agent.invoke({
        "task": "Use the linked source",
        "messages": [HumanMessage(content="Use the linked source")],
        "symlinkdir": {"source": str(source), "dest": "inputs/source"},
    })

    link = workspace / "inputs" / "source"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()
    assert state["symlinkdir"]["is_linked"] is True
    agent.close()


def test_composite_rejects_an_empty_plan_with_a_clear_error(tmp_path):
    model = CompositeFakeModel(plan=Plan(steps=[]))
    agent = PlanningExecutionAgent(
        llm=model,
        workspace=tmp_path,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    with pytest.raises(ValueError, match="produced an empty plan"):
        agent.invoke("Solve this task")
    agent.close()


def test_current_invocation_task_replaces_persisted_thread_state(tmp_path):
    checkpointer = Checkpointer.from_workspace(tmp_path)
    first_model = CompositeFakeModel()
    agent = PlanningExecutionAgent(
        llm=first_model,
        workspace=tmp_path,
        checkpointer=checkpointer,
        max_reflection_steps=0,
        enable_metrics=False,
    )
    config = {"configurable": {"thread_id": "same-thread"}}
    agent.invoke("First task", config=config)

    # Supply enough responses for a second execution while retaining one parent
    # identity and thread. The bound graph nodes share this underlying model.
    first_model.messages = CompositeFakeModel().messages
    state = agent.invoke("Second task", config=config)

    assert state["task"] == "Second task"
    plan_requests = [
        request_text(messages)
        for schema, messages in first_model.structured_requests
        if schema == "Plan"
    ]
    assert "Task:\nSecond task" in plan_requests[-1]
    assert "First task" not in plan_requests[-1]
    agent.close()


def test_interrupted_tool_resumes_without_repeating_non_idempotent_work(
    tmp_path,
):
    side_effects: list[str] = []

    @tool("commit_once")
    def commit_once() -> str:
        """Perform one non-idempotent operation after human approval."""
        approval = interrupt({"question": "Approve the operation?"})
        side_effects.append(str(approval))
        return "operation committed"

    plan = Plan(
        steps=[
            PlanStep(
                name="Commit",
                description="Perform the approved operation.",
                requires_code=False,
                expected_outputs=["commit confirmation"],
                success_criteria=["operation committed"],
            )
        ]
    )
    model = CompositeFakeModel(
        plan=plan,
        messages=iter([
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_once",
                        "args": {},
                        "id": "commit-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="operation completed"),
            AIMessage(content="one-step-summary"),
        ]),
    )
    checkpointer = Checkpointer.from_workspace(tmp_path)
    agent = PlanningExecutionAgent(
        llm=model,
        workspace=tmp_path,
        checkpointer=checkpointer,
        extra_tools=[commit_once],
        max_reflection_steps=0,
        enable_metrics=False,
    )
    config = {"configurable": {"thread_id": "interrupted-thread"}}

    interrupted_state = agent.invoke("Commit once", config=config)
    assert interrupted_state["__interrupt__"]
    assert side_effects == []

    resumed_state = agent.invoke(Command(resume="approved"), config=config)

    assert agent.format_result(resumed_state) == "one-step-summary"
    assert side_effects == ["approved"]
    assert resumed_state["step_results"] == ["one-step-summary"]
    agent.close()


def test_compatibility_name_is_the_composite_and_rejects_child_injection(
    tmp_path,
):
    assert PlanningExecutorWorkflow is PlanningExecutionAgent
    with pytest.raises(TypeError):
        PlanningExecutorWorkflow(
            llm=CompositeFakeModel(),
            planner=object(),
            executor=object(),
            workspace=tmp_path,
        )
