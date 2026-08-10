"""Persistent composite agent for planning and stepwise execution.

The parent agent owns one LangGraph runtime, checkpointer, store, thread, and LLM.
Planning and execution are native checkpointed subgraphs rather than separately
compiled/invoked child agents.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langgraph.types import Command, Overwrite

from ursa.agents.base import AgentContext
from ursa.agents.execution_agent import ExecutionAgent, ReviewAssessment
from ursa.agents.planning_agent import Plan, PlanningGraphMixin


def message_text(message: Any) -> str:
    """Return plain text content from a message-like value."""
    if isinstance(message, BaseMessage):
        return message.text
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content or "")


class PlanExecuteState(TypedDict, total=False):
    """Shared parent state and the channels used by its native subgraphs."""

    task: str
    hypothesis: str
    plan: Plan
    step_idx: int
    step_results: list[str]

    # Planner channels
    reflection_steps: int

    # ``review`` is reused sequentially by the planner and executor subgraphs.
    review: str | ReviewAssessment

    # Executor channels
    messages: Annotated[list[AnyMessage], add_messages]
    symlinkdir: dict[str, Any]
    current_user_request: str


class PlanningExecutionAgent(PlanningGraphMixin, ExecutionAgent):
    """Plan a task and execute each plan step in one persistent graph.

    The agent uses one LLM for planning, reflection, tool-driven execution,
    review, and recap.  The planner and executor are native LangGraph subgraphs
    compiled with ``checkpointer=True`` so they inherit the parent agent's
    checkpointer and receive LangGraph-managed nested checkpoint namespaces.

    ``max_reflection_steps`` configures the planner. All remaining keyword
    arguments are the normal :class:`ExecutionAgent` / :class:`BaseAgent`
    options, including workspace, tools, persistence, thread, telemetry, and
    retention settings.
    """

    state_type = PlanExecuteState

    def _normalize_inputs(self, inputs: Any) -> dict[str, Any] | Command:
        normalized_input = super()._normalize_inputs(inputs)
        if isinstance(normalized_input, Command):
            return normalized_input
        normalized = dict(normalized_input)
        task = str(normalized.get("task", "") or "").strip()
        if not task:
            for message in reversed(normalized.get("messages", [])):
                text = message_text(message).strip()
                if text:
                    task = text
                    break
        if not task:
            raise ValueError(
                "PlanningExecutionAgent requires a task or message input."
            )
        # A scalar task channel ensures a new invocation on a persistent thread
        # supersedes the previous invocation's task.
        normalized["task"] = task
        # Symlink requests are invocation-scoped. An explicit empty mapping
        # prevents a prior persistent-thread request from leaking forward.
        normalized.setdefault("symlinkdir", {})
        return normalized

    def format_query(
        self, prompt: str, state: PlanExecuteState | None = None
    ) -> PlanExecuteState:
        query: dict[str, Any] = dict(state or {})
        query.update({
            "task": prompt,
            "messages": [HumanMessage(content=prompt)],
        })
        return cast(PlanExecuteState, query)

    def format_result(self, state: PlanExecuteState) -> str:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("Planning execution completed without a response.")
        return message_text(messages[-1])

    def _prepare_planning(self, state: PlanExecuteState) -> PlanExecuteState:
        task = str(state.get("task", "") or "").strip()
        if not task:
            raise ValueError(
                "PlanningExecutionAgent requires a task or message input."
            )
        self.events().emit("Planning task", stage="planning")
        return cast(
            PlanExecuteState,
            {
                "task": task,
                "hypothesis": "",
                "step_idx": 0,
                "step_results": [],
                "review": "",
                "reflection_steps": self.max_reflection_steps,
                # Do not leak an earlier persistent run or executor transcript
                # into the planner. The planner's own full state remains in its
                # nested checkpoint namespace.
                "messages": Overwrite([HumanMessage(content=task)]),
                "symlinkdir": dict(state.get("symlinkdir", {})),
            },
        )

    @staticmethod
    def _last_step_summary(state: PlanExecuteState) -> str | None:
        results = state.get("step_results", [])
        if not results:
            return None
        summary = str(results[-1]).strip()
        return summary or None

    def _step_prompt(self, state: PlanExecuteState) -> str:
        plan = state.get("plan")
        if plan is None:
            raise ValueError("The planner completed without producing a plan.")
        if not plan.steps:
            raise ValueError(
                "The planner produced an empty plan; at least one step is required."
            )

        step_idx = state.get("step_idx", 0)
        if step_idx < 0 or step_idx >= len(plan.steps):
            raise ValueError(f"Invalid plan step index {step_idx}.")
        plan_step = plan.steps[step_idx]

        prompt_parts = [
            f"You are contributing to the larger solution:\n{state['task']}"
        ]
        if hypothesis := str(state.get("hypothesis", "") or "").strip():
            prompt_parts.append(f"Hypothesis to test or refine:\n{hypothesis}")
        if previous := self._last_step_summary(state):
            prompt_parts.append(f"Previous-step summary:\n{previous}")
        prompt_parts.extend([
            f"Current step:\n{plan_step}",
            (
                "Execute this step and report concrete results for the executor "
                "of the next step. Do not use placeholders. Run commands to "
                "execute generated code when applicable. Only address the "
                "current step."
            ),
        ])
        return "\n\n".join(prompt_parts)

    def _prepare_step(self, state: PlanExecuteState) -> PlanExecuteState:
        prompt = self._step_prompt(state)
        step_idx = state.get("step_idx", 0)
        self.events().emit(
            "Executing plan step",
            stage="execute_step",
            step=step_idx + 1,
            total=len(state["plan"].steps),
        )
        return cast(
            PlanExecuteState,
            {
                # Each executor subgraph invocation receives a clean transcript.
                # Cross-step context is explicit through step_results.
                "messages": Overwrite([HumanMessage(content=prompt)]),
                "current_user_request": prompt,
                "symlinkdir": state.get("symlinkdir", {}),
            },
        )

    def _record_step(self, state: PlanExecuteState) -> PlanExecuteState:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError(
                "The executor completed a step without a response."
            )
        summary = message_text(messages[-1]).strip()
        if not summary:
            raise ValueError("The executor returned an empty step summary.")
        return cast(
            PlanExecuteState,
            {
                "step_results": [*state.get("step_results", []), summary],
                "step_idx": state.get("step_idx", 0) + 1,
            },
        )

    def _after_step(self, state: PlanExecuteState) -> str:
        plan = state.get("plan")
        if plan is None:
            raise ValueError("The planner completed without producing a plan.")
        if state.get("step_idx", 0) >= len(plan.steps):
            return self._after_plan_completion()
        return "prepare_step"

    def _after_plan_completion(self) -> str:
        """Return the route used after the final execution step."""
        return END

    def _completion_routes(self) -> dict[str, str]:
        """Declare post-execution routes added by a specialized composite."""
        return {END: END}

    def _connect_pre_planner_nodes(self) -> None:
        """Connect preparation to planning; subclasses may insert nodes."""
        self.graph.add_edge("prepare_planning", "planner")

    def _connect_post_execution_nodes(self) -> None:
        """Add specialized nodes reached after all plan steps complete."""

    def _build_graph(self) -> None:
        # The unbound model remains available for planner/review/recap calls.
        self.tool_llm = self.llm.model_copy().bind_tools(self.tools.values())

        planner_builder = StateGraph(
            self.state_type, context_schema=AgentContext
        )
        self._populate_planning_graph(planner_builder)
        planner_subgraph = planner_builder.compile(
            checkpointer=True, name="planner"
        )

        executor_builder = StateGraph(
            self.state_type, context_schema=AgentContext
        )
        self._populate_execution_graph(executor_builder)
        executor_subgraph = executor_builder.compile(
            checkpointer=True, name="executor"
        )

        self.add_node(self._prepare_planning, "prepare_planning")
        self.add_node(planner_subgraph, "planner")
        self.add_node(self._prepare_step, "prepare_step")
        self.add_node(executor_subgraph, "executor")
        self.add_node(self._record_step, "record_step")

        self.graph.add_edge(START, "prepare_planning")
        self._connect_pre_planner_nodes()
        self.graph.add_edge("planner", "prepare_step")
        self.graph.add_edge("prepare_step", "executor")
        self.graph.add_edge("executor", "record_step")
        self._connect_post_execution_nodes()
        self.graph.add_conditional_edges(
            "record_step",
            self._wrap_cond(
                self._after_step, "after_step", "planning_execution"
            ),
            {
                "prepare_step": "prepare_step",
                **self._completion_routes(),
            },
        )


# Compatibility name: this is now a BaseAgent, not a BaseWorkflow facade.
PlanningExecutorWorkflow = PlanningExecutionAgent
