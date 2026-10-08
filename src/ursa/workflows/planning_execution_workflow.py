# ruff: noqa: TID251
"""Persistent workflow for planning and stepwise execution.

The parent owns the LangGraph runtime, checkpointer, store, and thread. Separate
``PlanningAgent`` and ``ExecutionAgent`` instances are adapted into checkpointed
child nodes with narrow state contracts.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Mapping, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph.message import add_messages
from langgraph.types import Command, Overwrite
from rich import get_console
from rich.panel import Panel

from ursa.agents.base import BaseAgent
from ursa.agents.execution_agent import ExecutionAgent, ExecutionState
from ursa.agents.planning_agent import (
    Plan,
    PlanningAgent,
    PlanningState,
)
from ursa.util.plan_renderer import render_plan_steps_rich
from ursa.workflows.base_workflow import BaseWorkflow

console = get_console()


def message_text(message: Any) -> str:
    """Return plain text content from a message-like value."""
    if isinstance(message, BaseMessage):
        return message.text
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content or "")


class PlanExecuteState(TypedDict, total=False):
    """Parent orchestration state for planning and stepwise execution."""

    task: str
    plan: Plan
    step_idx: int
    step_results: list[str]

    messages: Annotated[list[AnyMessage], add_messages]
    symlinkdir: dict[str, Any]
    current_user_request: str


class PlanningExecutionAgent(BaseAgent[PlanExecuteState]):
    """Plan a task and execute each plan step in one persistent graph.

    The agent uses one underlying LLM for planning, reflection, tool-driven
    execution, review, and recap. Both child agents are compiled with
    ``checkpointer=True`` so they inherit the parent agent's
    checkpointer and receive LangGraph-managed nested checkpoint namespaces.

    ``max_reflection_steps`` configures the planner. All remaining keyword
    arguments are the normal :class:`ExecutionAgent` / :class:`BaseAgent`
    options, including workspace, tools, persistence, thread, telemetry, and
    retention settings.
    """

    state_type = PlanExecuteState

    def __init__(
        self,
        llm: BaseChatModel,
        max_reflection_steps: int = 1,
        log_state: bool = False,
        extra_tools: list[BaseTool] | None = None,
        tokens_before_summarize: int = 50000,
        messages_to_keep: int = 20,
        use_web: bool = False,
        safe_codes: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.max_reflection_steps = max_reflection_steps
        self._child_llm_source = llm
        super().__init__(
            llm,
            tokens_before_summarize=tokens_before_summarize,
            messages_to_keep=messages_to_keep,
            **kwargs,
        )
        child_kwargs = {
            "workspace": self.workspace,
            "group": self.group,
            "thread_id": self.thread_id,
            "enable_metrics": False,
            "rag_tools": self.rag_tools,
            "rag_tool_group": self.rag_tool_group,
            "rag_tool_embedding": self.rag_tool_embedding,
            "rag_tool_return_k": self.rag_tool_return_k,
            "max_single_tool_message_tokens": (
                self.max_single_tool_message_tokens
            ),
        }
        self.planner_agent = PlanningAgent(
            llm,
            max_reflection_steps=max_reflection_steps,
            **child_kwargs,
        )
        self.execution_agent = ExecutionAgent(
            llm,
            log_state=log_state,
            extra_tools=extra_tools,
            tokens_before_summarize=tokens_before_summarize,
            messages_to_keep=messages_to_keep,
            use_web=use_web,
            safe_codes=safe_codes,
            **child_kwargs,
        )

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

    def format_result(self, result: PlanExecuteState) -> str:
        messages = result.get("messages", [])
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
                "step_idx": 0,
                "step_results": [],
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

    def _planner_input(self, state: PlanExecuteState) -> PlanningState:
        """Expose only the planner contract to the child agent."""
        return cast(
            PlanningState,
            {
                "task": state["task"],
                "review": "",
                # The child state has an add_messages reducer and a persistent
                # namespace. Replace its transcript for every new plan.
                "messages": Overwrite(list(state.get("messages", []))),
                "reflection_steps": self.max_reflection_steps,
            },
        )

    @staticmethod
    def _planner_output(state: PlanningState) -> PlanExecuteState:
        """Return only the completed plan to the parent graph."""
        plan = state.get("plan")
        if plan is None:
            raise ValueError("The planner completed without producing a plan.")
        return cast(PlanExecuteState, {"plan": plan})

    @staticmethod
    def _executor_input(state: PlanExecuteState) -> ExecutionState:
        """Expose one isolated plan-step request to the executor child."""
        messages = list(state.get("messages", []))
        if not messages:
            raise ValueError("Execution requires a prepared step message.")
        return cast(
            ExecutionState,
            {
                "messages": Overwrite(messages),
                "symlinkdir": dict(state.get("symlinkdir", {})),
                "current_user_request": state.get("current_user_request", ""),
            },
        )

    @staticmethod
    def _executor_output(state: ExecutionState) -> PlanExecuteState:
        """Return the executor transcript and workspace update to the parent."""
        messages = list(state.get("messages", []))
        if not messages:
            raise ValueError("The executor completed without a response.")
        return cast(
            PlanExecuteState,
            {
                "messages": Overwrite(messages),
                "symlinkdir": dict(state.get("symlinkdir", {})),
            },
        )

    def _build_graph(self) -> None:
        self.add_node(self._prepare_planning, "prepare_planning")
        self.add_agent_node(
            "planner",
            self.planner_agent,
            input_fn=self._planner_input,
            output_fn=self._planner_output,
        )
        self.add_node(self._prepare_step, "prepare_step")
        self.add_agent_node(
            "executor",
            self.execution_agent,
            input_fn=self._executor_input,
            output_fn=self._executor_output,
        )
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


class PlanningExecutorWorkflow(BaseWorkflow):
    """Deprecated two-agent planning/execution loop.

    This preserves the original ``planner=`` / ``executor=`` constructor and
    string return value. New code should use :class:`PlanningExecutionAgent`,
    which composes its child agents into a persistent LangGraph.
    """

    def __init__(
        self,
        planner: Any,
        executor: Any,
        workspace: Any = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "PlanningExecutorWorkflow is deprecated; use "
            "PlanningExecutionAgent(llm=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.workspace = workspace

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        *,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        task = str(inputs.get("task", "") or "").strip()
        if not task:
            raise ValueError("PlanningExecutorWorkflow requires a task.")
        invoke_config = dict(config or {})

        with console.status(
            "[bold deep_pink1]Planning overarching steps . . .",
            spinner="point",
            spinner_style="deep_pink1",
        ):
            planner_prompt = (
                f"Break this down into one step per technique:\n{task}"
            )
            planning_output = self.planner.invoke(
                planner_prompt,
                config=invoke_config,
            )
            render_plan_steps_rich(planning_output["plan"].steps)

        last_step_summary = "No previous step."
        for i, step in enumerate(planning_output["plan"].steps):
            step_prompt = (
                f"You are contributing to the larger solution:\n"
                f"{task}\n\n"
                f"Previous-step summary:\n"
                f"{last_step_summary}\n\n"
                f"Current step:\n"
                f"{step}\n\n"
                "Execute this step and report results for the executor of the next step."
                "Do not use placeholders."
                "Run commands to execute code generated for the step if applicable."
                "Only address the current step. Stay in your lane."
            )
            console.print(
                Panel(
                    step_prompt,
                    title=f"[bold orange3 on black]Solving Step {i + 1}",
                    border_style="orange3 on black",
                    style="orange3 on black",
                )
            )
            result = self.executor.invoke(
                step_prompt,
                config=invoke_config,
            )
            last_step_summary = message_text(result["messages"][-1])
            console.print(
                Panel(
                    last_step_summary,
                    title=f"Step {i + 1} Final Response",
                    border_style="orange3 on black",
                    style="orange3 on black",
                )
            )
        return last_step_summary
