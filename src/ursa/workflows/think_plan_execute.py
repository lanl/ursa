"""Think-plan-execute composite using URSA's hypothesis-space behavior."""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain.chat_models import BaseChatModel
from langgraph.constants import END
from langgraph.graph.state import StateGraph
from langgraph.types import Command

from ursa.agents.base import AgentContext
from ursa.agents.hypothesizer_agent import (
    HypothesizerGraphMixin,
    HypothesizerState,
)
from ursa.agents.planning_execution_agent import (
    PlanExecuteState,
    PlanningExecutionAgent,
)


class ThinkPlanExecuteState(PlanExecuteState, HypothesizerState, total=False):
    """Parent state shared with the planner, executor, and hypothesizer."""

    hypothesis_phase: Literal["initial", "results"]


class ThinkPlanningExecutionAgent(
    HypothesizerGraphMixin, PlanningExecutionAgent
):
    """Build, investigate, and revise a durable hypothesis space.

    One parent runtime embeds URSA's hypothesizer, planner, and executor as
    native checkpointed subgraphs. The hypothesizer first initializes or updates
    its durable hypothesis space from the current user request. The planner then
    creates a plan guided by that space, the executor performs each step, and
    the resulting step summaries are sent through the hypothesizer again as new
    evidence.
    """

    state_type = ThinkPlanExecuteState

    # HypothesizerGraphMixin also supports use as a standalone agent. In this
    # composite, public input and follow-up formatting must retain the parent
    # planning/execution contract.
    def _normalize_inputs(self, inputs: Any) -> dict[str, Any] | Command:
        return PlanningExecutionAgent._normalize_inputs(self, inputs)

    def format_query(
        self,
        prompt: str,
        state: ThinkPlanExecuteState | None = None,
    ) -> ThinkPlanExecuteState:
        return cast(
            ThinkPlanExecuteState,
            PlanningExecutionAgent.format_query(self, prompt, state),
        )

    def format_result(self, state: ThinkPlanExecuteState) -> str:
        artifact = self._response_text(
            state.get("hypothesis_space_markdown", "")
        )
        if artifact:
            return artifact
        return PlanningExecutionAgent.format_result(self, state)

    def _build_graph(self) -> None:
        PlanningExecutionAgent._build_graph(self)

    def _prepare_initial_hypothesis(
        self, state: ThinkPlanExecuteState
    ) -> ThinkPlanExecuteState:
        task = state["task"]
        prior_artifact = self._response_text(
            state.get("hypothesis_space_markdown", "")
        )
        # A follow-up is new evidence for the original question rather than an
        # unrelated replacement. A fresh thread starts a fresh question while
        # the experience artifact can still supply durable background.
        query = (
            str(state.get("query", "") or "").strip()
            if prior_artifact
            else task
        )
        query = query or task
        return cast(
            ThinkPlanExecuteState,
            {
                "query": query,
                "new_information": task,
                "context": (
                    "Treat this as a follow-up request and update the existing "
                    "hypothesis space before planning."
                    if prior_artifact
                    else "Initialize a hypothesis space before planning."
                ),
                "experience_filename": state.get(
                    "experience_filename", self.experience_filename
                ),
                "revision_history": list(state.get("revision_history", [])),
                "hypothesis_phase": "initial",
            },
        )

    def _prepare_results_hypothesis_update(
        self, state: ThinkPlanExecuteState
    ) -> ThinkPlanExecuteState:
        results = [
            str(result).strip()
            for result in state.get("step_results", [])
            if str(result).strip()
        ]
        if not results:
            raise ValueError(
                "Think-plan-execute completed without results to send back "
                "to the hypothesizer."
            )
        summary = "\n".join(
            f"- Step {index}: {result}"
            for index, result in enumerate(results, start=1)
        )
        return cast(
            ThinkPlanExecuteState,
            {
                "query": state.get("query", state["task"]),
                "new_information": (
                    "Execution produced the following evidence and results:\n"
                    f"{summary}"
                ),
                "context": f"Current user request:\n{state['task']}",
                "experience_filename": state.get(
                    "experience_filename", self.experience_filename
                ),
                "revision_history": list(state.get("revision_history", [])),
                "hypothesis_phase": "results",
            },
        )

    def _adopt_hypothesis_space(
        self, state: ThinkPlanExecuteState
    ) -> ThinkPlanExecuteState:
        artifact = self._response_text(
            state.get("hypothesis_space_markdown", "")
        )
        if not artifact:
            raise ValueError(
                "The hypothesizer completed without producing a hypothesis "
                "space."
            )
        return cast(ThinkPlanExecuteState, {"hypothesis": artifact})

    @staticmethod
    def _after_hypothesis_space(
        state: ThinkPlanExecuteState,
    ) -> Literal["planner", "END"]:
        phase = state.get("hypothesis_phase")
        if phase == "initial":
            return "planner"
        if phase == "results":
            return "END"
        raise ValueError(f"Unknown hypothesis update phase: {phase!r}.")

    def _connect_pre_planner_nodes(self) -> None:
        hypothesizer_builder = StateGraph(
            self.state_type, context_schema=AgentContext
        )
        self._populate_hypothesizer_graph(hypothesizer_builder)
        hypothesizer_subgraph = hypothesizer_builder.compile(
            checkpointer=True, name="hypothesizer"
        )

        self.add_node(
            self._prepare_initial_hypothesis, "prepare_initial_hypothesis"
        )
        self.add_node(hypothesizer_subgraph, "hypothesizer")
        self.add_node(self._adopt_hypothesis_space, "adopt_hypothesis_space")
        self.add_node(
            self._prepare_results_hypothesis_update,
            "prepare_results_hypothesis_update",
        )

        self.graph.add_edge("prepare_planning", "prepare_initial_hypothesis")
        self.graph.add_edge("prepare_initial_hypothesis", "hypothesizer")
        self.graph.add_edge("hypothesizer", "adopt_hypothesis_space")
        self.graph.add_conditional_edges(
            "adopt_hypothesis_space",
            self._wrap_cond(
                self._after_hypothesis_space,
                "after_hypothesis_space",
                "think_planning_execution",
            ),
            {"planner": "planner", "END": END},
        )

    def _after_plan_completion(self) -> str:
        return "prepare_results_hypothesis_update"

    def _completion_routes(self) -> dict[str, str]:
        return {
            "prepare_results_hypothesis_update": (
                "prepare_results_hypothesis_update"
            )
        }

    def _connect_post_execution_nodes(self) -> None:
        self.graph.add_edge("prepare_results_hypothesis_update", "hypothesizer")


def think_plan_execute_workflow(
    llm: BaseChatModel, **agent_kwargs: Any
) -> ThinkPlanningExecutionAgent:
    """Create a persistent hypothesis-plan-execute-update parent agent."""
    return ThinkPlanningExecutionAgent(llm=llm, **agent_kwargs)


__all__ = [
    "ThinkPlanExecuteState",
    "ThinkPlanningExecutionAgent",
    "think_plan_execute_workflow",
]
