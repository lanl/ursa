import asyncio
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.runtime import Runtime
from rich import get_console

from ursa.agents.execution_agent import ExecutionAgent, ExecutionState
from ursa.agents.planning_agent import Plan, PlanningAgent, PlanningState
from ursa.util.plan_renderer import render_plan_steps_rich
from ursa.workflows.base_workflow import BaseWorkflow
from ursa.workflows.utils import message_text, nested_agent_kwargs

console = get_console()


class PlanExecuteState(TypedDict, total=False):
    task: str
    hypothesis: str
    plan: Plan
    step_idx: int
    messages: Annotated[list, add_messages]


def _extract_task(state: Mapping[str, Any]) -> str:
    task = str(state.get("task", "") or "").strip()
    if task:
        return task

    for message in reversed(state.get("messages", [])):
        text = message_text(message).strip()
        if text:
            return text

    raise ValueError(
        "planning execution workflow requires a task or message input"
    )


def _last_step_summary(state: PlanExecuteState) -> str | None:
    if state.get("step_idx", 0) <= 0:
        return None
    messages = state.get("messages", [])
    if not messages:
        return None
    summary = message_text(messages[-1]).strip()
    return summary or None


def _result_message(result: Mapping[str, Any], fallback: str) -> BaseMessage:
    messages = result.get("messages")
    if (
        isinstance(messages, list)
        and messages
        and isinstance(messages[-1], BaseMessage)
    ):
        return messages[-1]
    return AIMessage(content=fallback)


def _planning_prompt(task: str, hypothesis: str | None = None) -> str:
    if not hypothesis:
        return task
    return (
        f"{task}\n\n"
        f"Working hypothesis:\n{hypothesis}\n\n"
        "Create a concrete execution plan that uses this hypothesis to solve the task."
    )


def _step_prompt(state: PlanExecuteState) -> str:
    step_idx = state.get("step_idx", 0)
    plan_step = state["plan"].steps[step_idx]
    prompt_parts = [
        f"You are contributing to the larger solution:\n{state['task']}"
    ]

    if hypothesis := str(state.get("hypothesis", "") or "").strip():
        prompt_parts.append(f"Hypothesis to test or refine:\n{hypothesis}")

    if last_step_summary := _last_step_summary(state):
        prompt_parts.append(f"Previous-step summary:\n{last_step_summary}")

    prompt_parts.append(f"Current step:\n{plan_step}")
    prompt_parts.append(
        "Execute this step and report results for the executor of the next step. "
        "Do not use placeholders. "
        "Run commands to execute code generated for the step if applicable. "
        "Only address the current step. Stay in your lane."
    )
    return "\n\n".join(prompt_parts)


def _should_continue(state: PlanExecuteState) -> str:
    if state.get("step_idx", 0) >= len(state["plan"].steps):
        return END
    return "execute_step"


def build_plan_nodes(
    planner: PlanningAgent,
    executor: ExecutionAgent,
):
    async def create_plan(
        state: PlanExecuteState,
        runtime: Runtime[Any],
        config: RunnableConfig | None = None,
    ):
        del runtime
        task = _extract_task(state)
        plan_prompt = _planning_prompt(task, state.get("hypothesis"))
        planning_state = planner.format_query(plan_prompt)
        plan_result: PlanningState = await planner._ainvoke(
            planning_state,
            **nested_agent_kwargs(config, checkpoint_ns="planner"),
        )
        return {"task": task, "plan": plan_result["plan"], "step_idx": 0}

    async def execute_step(
        state: PlanExecuteState,
        runtime: Runtime[Any],
        config: RunnableConfig | None = None,
    ):
        del runtime
        step_prompt = executor.format_query(_step_prompt(state))
        result: ExecutionState = await executor._ainvoke(
            step_prompt,
            **nested_agent_kwargs(config, checkpoint_ns="executor"),
        )
        result_text = executor.format_result(result)
        return {
            "messages": [_result_message(result, result_text)],
            "step_idx": state.get("step_idx", 0) + 1,
        }

    return create_plan, execute_step


def add_plan_nodes(
    builder: StateGraph,
    planner: PlanningAgent,
    executor: ExecutionAgent,
) -> None:
    create_plan, execute_step = build_plan_nodes(
        planner=planner,
        executor=executor,
    )
    builder.add_node("create_plan", create_plan)
    builder.add_node("execute_step", execute_step)
    builder.add_conditional_edges("create_plan", _should_continue)
    builder.add_conditional_edges("execute_step", _should_continue)


def plan_execute_workflow(
    planner: PlanningAgent,
    executor: ExecutionAgent,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    builder = StateGraph(state_schema=PlanExecuteState)
    add_plan_nodes(builder, planner=planner, executor=executor)
    builder.add_edge(START, "create_plan")
    return builder.compile(checkpointer=checkpointer)


class PlanningExecutorWorkflow(BaseWorkflow):
    """
    The Planning-Executor workflow is a workflow that composes two agents in a for-loop:
        - The planning agent takes the user input, develops a step-by-step plan as a list
        - The list is passed, entry by entry to an execution agent to carry out the plan.
    """

    def __init__(self, planner, executor, workspace=None, **kwargs):
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.workspace = workspace

        # FIXME: DOES NOT CURRENTLY WORK IN WEB INTERFACE WITH
        # SQL checkpointing
        # MOVING TO IN MEMORY CHECKPOINTING FOR NOW
        # Setup checkpointing
        # db_path = Path(workspace) / "checkpoint.db"
        # db_path.parent.mkdir(parents=True, exist_ok=True)
        # conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # checkpointer = SqliteSaver(conn)

        self.planner.checkpointer = InMemorySaver()
        self.executor.checkpointer = InMemorySaver()
        self._workflow = plan_execute_workflow(
            planner=self.planner,
            executor=self.executor,
        )

    async def ainvoke(self, input, config):
        if isinstance(input, str):
            input = {"task": input}
        await self._workflow.ainvoke(input, config=config)

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        *,
        config: RunnableConfig | None = None,
        **_kwargs,
    ):
        with console.status(
            "[bold deep_pink1]Planning overarching steps . . .",
            spinner="point",
            spinner_style="deep_pink1",
        ):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = asyncio.run(
                    self._workflow.ainvoke(inputs, config=config)
                )
            else:
                raise RuntimeError(
                    "PlanningExecutorWorkflow.invoke() cannot be used from an "
                    "async context because it relies on async workflow nodes. "
                    "Use the graph workflow directly via `.ainvoke(...)`."
                )

        render_plan_steps_rich(result["plan"].steps)
        return message_text(result["messages"][-1])


def main():
    from uuid import uuid4

    from langchain.chat_models import init_chat_model

    from ursa.agents import ExecutionAgent, PlanningAgent
    from ursa.observability.timing import render_session_summary

    tid = "run-" + uuid4().hex[:8]

    # Define a simple problem
    index_to_find = 35

    problem = (
        f"Create a single python script to compute the Fibonacci \n"
        f"number at position {index_to_find} in the sequence.\n\n"
        # f"Compute the answer through more than one distinct technique, \n"
        # f"benchmark and compare the approaches then explain which one is the best."
    )

    # Setup Planning Agent
    planner_model = init_chat_model(model="openai:o4-mini")
    planner = PlanningAgent(
        llm=planner_model, enable_metrics=True, thread_id=tid
    )

    # Setup Execution Agent
    executor_model = init_chat_model(model="openai:o4-mini")
    executor = ExecutionAgent(
        llm=executor_model, enable_metrics=True, thread_id=tid
    )

    # Initialize workflow
    workflow = PlanningExecutorWorkflow(planner=planner, executor=executor)

    # Run problem through the workflow
    workflow(problem)

    # Print agent telemetry data
    render_session_summary(tid)


if __name__ == "__main__":
    main()
