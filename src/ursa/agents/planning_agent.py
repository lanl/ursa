from textwrap import dedent
from typing import Annotated, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ursa.prompt_library.planning_prompts import (
    planner_prompt,
    reflection_prompt,
)
from ursa.util.structured_output import invoke_structured

from .base import BaseAgent


# plan schema
class PlanStep(BaseModel):
    name: str = Field(description="Short, specific step title")
    description: str = Field(description="Detailed description of the step")
    requires_code: bool = Field(
        description="True if this step needs code to be written/run"
    )
    expected_outputs: list[str] = Field(
        description="Concrete artifacts or results produced by this step"
    )
    success_criteria: list[str] = Field(
        description="Measurable checks that indicate the step succeeded"
    )


class Plan(BaseModel):
    steps: list[PlanStep] = Field(
        description="Ordered list of steps to solve the problem"
    )

    def __str__(self):
        plan = []
        for id, step in enumerate(self.steps):
            expected_outputs = [
                f"- {output}" for output in step.expected_outputs
            ]
            expected_outputs = "\n".join(expected_outputs)
            success_criteria = [
                f"- {criterion}" for criterion in step.success_criteria
            ]
            success_criteria = "\n".join(success_criteria)

            step_str = f"""
            ## {id} -- {step.name}
            Requires Code: {step.requires_code}

            {step.description}

            """
            step_str = dedent(step_str)

            step_str += "### Expected Outputs\n"
            for output in step.expected_outputs:
                step_str += f"- {output}\n"

            step_str += "\n\n"
            step_str += "### Success Criteria\n"
            for criterion in step.success_criteria:
                step_str += f"- {criterion}\n"

            plan.append(step_str)

        return "\n".join(plan)


# planning state
class PlanningState(TypedDict, total=False):
    """State dictionary for planning agent"""

    task: str
    plan: Plan
    review: str
    messages: Annotated[list, add_messages]
    reflection_steps: int


class PlanningGraphMixin:
    """Reusable planning graph nodes for agents that embed a planner subgraph.

    The mixin deliberately does not own or compile a graph.  Its nodes use the
    containing agent's LLM, events, and node metadata so a composite agent can
    keep one runtime and persistence identity.
    """

    planner_prompt: str
    reflection_prompt: str
    max_reflection_steps: int

    def __init__(
        self,
        llm: BaseChatModel,
        max_reflection_steps: int = 1,
        **kwargs,
    ):
        # Set these before BaseAgent's post-init graph build. This also makes the
        # mixin safe in a composite-agent MRO where ExecutionAgent is next.
        self.planner_prompt = planner_prompt
        self.reflection_prompt = reflection_prompt
        self.max_reflection_steps = max_reflection_steps
        super().__init__(llm, **kwargs)

    def format_result(self, state: PlanningState) -> str:
        return str(state["plan"])

    def format_query(
        self, prompt: str, state: PlanningState | None = None
    ) -> PlanningState:
        """Start a planning run while retaining its persisted graph state."""
        query = dict(state or {})
        query.update({
            "task": prompt,
            "review": "",
            "messages": [
                *(state or {}).get("messages", []),
                HumanMessage(content=prompt),
            ],
            "reflection_steps": self.max_reflection_steps,
        })
        return cast(PlanningState, query)

    def generation_node(
        self,
        state: PlanningState,
        config: RunnableConfig | None = None,
    ) -> PlanningState:
        """
        Plan generation with structured output. Produces a JSON string in messages
        and a parsed list of steps in state["plan_steps"].
        """
        events = self.events(config)
        events.emit("Drafting plan", stage="generate")
        task = state.get("task") or _latest_human_message(state)
        request = f"Task:\n{task}"
        hypothesis = str(state.get("hypothesis", "") or "").strip()
        if hypothesis:
            request += f"\n\nWorking hypothesis:\n{hypothesis}"
        if review := state.get("review"):
            request += (
                f"\n\nCurrent plan:\n{state['plan'].model_dump_json(indent=2)}"
                f"\n\nReviewer feedback:\n{review}"
                "\n\nProduce a revised plan that addresses the feedback."
            )
        messages = [
            SystemMessage(content=self.planner_prompt),
            HumanMessage(content=request),
        ]

        plan = cast(
            Plan,
            invoke_structured(
                self.llm,
                Plan,
                messages,
                config=self.nested_config(config, tags=["planner", "generate"]),
                context="planning generation",
                repair=3,
            ),
        )
        events.emit(
            "Drafted plan",
            stage="generate_result",
            steps=[step.model_dump() for step in plan.steps],
        )

        return {
            "task": task,
            "plan": plan,
            "messages": [AIMessage(content=plan.model_dump_json())],
            "reflection_steps": state.get(
                "reflection_steps", self.max_reflection_steps
            ),
        }

    def reflection_node(
        self,
        state: PlanningState,
        config: RunnableConfig | None = None,
    ) -> PlanningState:
        events = self.events(config)
        events.emit("Reviewing plan", stage="reflect")
        messages = [
            SystemMessage(content=self.reflection_prompt),
            HumanMessage(
                content=(
                    f"Original task:\n{state['task']}"
                    "\n\nCandidate plan:\n"
                    f"{state['plan'].model_dump_json(indent=2)}"
                    "\n\nReview this candidate plan."
                )
            ),
        ]
        res = StrOutputParser().invoke(
            self.llm.invoke(
                messages,
                self.nested_config(config, tags=["planner", "reflect"]),
            )
        )

        if not res.strip():
            # Some providers can return an empty reflection message; treat that as
            # "no objections" so we do not regenerate from an empty human turn.
            res = "[APPROVED]"

        approved = "[APPROVED]" in res
        events.emit(
            "Plan approved" if approved else "Plan needs another pass",
            stage="reflect_result",
            approved=approved,
            reason=res,
        )
        return {
            "plan": state["plan"],
            "review": res,
            "messages": [HumanMessage(content=res)],
            "reflection_steps": state["reflection_steps"] - 1,
        }

    def _populate_planning_graph(self, builder) -> None:
        """Add the planner nodes and edges to ``builder`` without compiling it."""
        builder.add_node(
            "generate",
            self._wrap_node(self.generation_node, "generate", "planner"),
        )
        builder.add_node(
            "reflect",
            self._wrap_node(self.reflection_node, "reflect", "planner"),
        )
        builder.set_entry_point("generate")
        builder.add_conditional_edges(
            "generate",
            self._wrap_cond(_should_reflect, "should_reflect", "planner"),
            {"reflect": "reflect", "END": END},
        )
        builder.add_conditional_edges(
            "reflect",
            self._wrap_cond(_should_regenerate, "should_regenerate", "planner"),
            {"generate": "generate", "END": END},
        )

    def _build_graph(self) -> None:
        self._populate_planning_graph(self.graph)


class PlanningAgent(PlanningGraphMixin, BaseAgent[PlanningState]):
    """Standalone graph-backed planning agent."""

    state_type = PlanningState


def _should_reflect(state: PlanningState):
    # Hit the reflection cap?
    if state["reflection_steps"] > 0:
        return "reflect"
    return "END"


def _should_regenerate(state: PlanningState):
    # Approved?
    if "[APPROVED]" in state.get("review", ""):
        return "END"

    return "generate"


def _latest_human_message(state: PlanningState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.text
    raise ValueError("Planning requires a task or human message.")
