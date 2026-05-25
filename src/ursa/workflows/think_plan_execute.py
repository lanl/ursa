from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.runtime import Runtime

from ursa.agents.execution_agent import ExecutionAgent
from ursa.agents.hypothesizer_agent import HypothesizerAgent, HypothesizerState
from ursa.agents.planning_agent import PlanningAgent
from ursa.workflows.planning_execution_workflow import (
    PlanExecuteState,
    _extract_task,
    add_plan_nodes,
)
from ursa.workflows.utils import nested_agent_kwargs as _nested_agent_kwargs


def think_plan_execute_workflow(
    hypothesizer: HypothesizerAgent,
    planner: PlanningAgent,
    executor: ExecutionAgent,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    async def create_hypothesis(
        state: PlanExecuteState,
        runtime: Runtime[Any],
        config: RunnableConfig | None = None,
    ):
        del runtime
        task = _extract_task(state)
        hypothesis_state = hypothesizer.format_query(task)
        result: HypothesizerState = await hypothesizer._ainvoke(
            hypothesis_state,
            **_nested_agent_kwargs(config, checkpoint_ns="hypothesizer"),
        )
        return {
            "task": task,
            "hypothesis": hypothesizer.format_result(result),
        }

    builder = StateGraph(state_schema=PlanExecuteState)
    builder.add_node("create_hypothesis", create_hypothesis)
    add_plan_nodes(builder, planner=planner, executor=executor)
    builder.add_edge(START, "create_hypothesis")
    builder.add_edge("create_hypothesis", "create_plan")
    return builder.compile(checkpointer=checkpointer)
