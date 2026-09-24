"""Think-plan-execute composite using URSA's hypothesis-space behavior."""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain.chat_models import BaseChatModel
from langgraph.constants import END
from langgraph.types import Command

from ursa.agents.hypothesizer_agent import (
    DEFAULT_HYPOTHESIS_EXPERIENCE,
    HypothesizerAgent,
    HypothesizerState,
)
from ursa.agents.planning_agent import PlanningState
from ursa.workflows.planning_execution_workflow import (
    PlanExecuteState,
    PlanningExecutionAgent,
)


class ThinkPlanExecuteState(PlanExecuteState, HypothesizerState, total=False):
    """Parent state shared with the planner, executor, and hypothesizer."""

    hypothesis: str
    hypothesis_phase: Literal["initial", "results"]


class ThinkPlanningExecutionAgent(PlanningExecutionAgent):
    """Build, investigate, and revise a durable hypothesis space.

    One parent runtime adapts URSA's hypothesizer, planner, and executor agents
    into checkpointed child nodes. The hypothesizer first initializes or
    updates its durable hypothesis space from the current user request. The
    planner then creates a plan guided by that space, the executor performs each
    step, and the resulting summaries return to the hypothesizer as new evidence.
    """

    state_type = ThinkPlanExecuteState

    def __init__(
        self,
        llm: BaseChatModel,
        experience_filename: str = DEFAULT_HYPOTHESIS_EXPERIENCE,
        **kwargs: Any,
    ) -> None:
        # The graph is built during BaseAgent initialization, so this adapter
        # setting must be available before the parent constructor runs.
        self.experience_filename = (
            HypothesizerAgent._validate_experience_filename(experience_filename)
        )
        super().__init__(llm, **kwargs)

    @staticmethod
    def _hypothesis_text(value: Any) -> str:
        return HypothesizerAgent._response_text(value)

    # Public input and follow-up formatting retain the parent workflow contract;
    # adapters isolate the standalone child-agent contracts inside the graph.
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
        artifact = self._hypothesis_text(
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
        prior_artifact = self._hypothesis_text(
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
        artifact = self._hypothesis_text(
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

    @staticmethod
    def _hypothesizer_input(
        state: ThinkPlanExecuteState,
    ) -> HypothesizerState:
        """Expose only the hypothesis-maintenance contract to the child."""
        return cast(
            HypothesizerState,
            {
                key: state[key]
                for key in HypothesizerState.__annotations__
                if key in state
            },
        )

    @staticmethod
    def _hypothesizer_output(
        state: HypothesizerState,
    ) -> ThinkPlanExecuteState:
        """Map the child artifact state back into the composite state."""
        return cast(ThinkPlanExecuteState, dict(state))

    def _planner_input(self, state: ThinkPlanExecuteState) -> PlanningState:
        planner_state = super()._planner_input(state)
        planner_state["hypothesis"] = state.get("hypothesis", "")
        return planner_state

    def _step_prompt(self, state: ThinkPlanExecuteState) -> str:
        prompt = super()._step_prompt(state)
        hypothesis = str(state.get("hypothesis", "") or "").strip()
        if not hypothesis:
            return prompt
        return f"Hypothesis to test or refine:\n{hypothesis}\n\n{prompt}"

    def _connect_pre_planner_nodes(self) -> None:
        if not hasattr(self, "hypothesizer_agent"):
            self.hypothesizer_agent = HypothesizerAgent(
                self._child_llm_source,
                # The child does not own persistence. Point its artifact
                # directory at the parent's den while LangGraph owns
                # checkpoint persistence.
                workspace=self.den,
                group=self.group,
                thread_id=self.thread_id,
                experience_filename=self.experience_filename,
                enable_metrics=False,
            )

        self.add_node(
            self._prepare_initial_hypothesis, "prepare_initial_hypothesis"
        )
        self.add_agent_node(
            "hypothesizer",
            self.hypothesizer_agent,
            input_fn=self._hypothesizer_input,
            output_fn=self._hypothesizer_output,
        )
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
