from textwrap import dedent
from typing import Annotated, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ursa.prompt_library.planning_prompts import (
    planner_prompt,
    reflection_prompt,
)

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

    plan: Plan
    messages: Annotated[list, add_messages]
    reflection_steps: int


class PlanningAgent(BaseAgent[PlanningState]):
    agent_state = PlanningState

    def __init__(
        self,
        llm: BaseChatModel,
        max_reflection_steps: int = 1,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.planner_prompt = planner_prompt
        self.reflection_prompt = reflection_prompt
        self.max_reflection_steps = max_reflection_steps

    def format_result(self, state: PlanningState) -> str:
        return str(state["plan"])

    def generation_node(self, state: PlanningState) -> PlanningState:
        """
        Plan generation with structured output. Produces a JSON string in messages
        and a parsed list of steps in state["plan_steps"].
        """

        print("PlanningAgent: generating . . .")
        messages = cast(list, state.get("messages"))
        if isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=self.planner_prompt)
        else:
            messages = [SystemMessage(content=self.planner_prompt)] + messages

        structured_llm = self.llm.with_structured_output(Plan)
        plan = cast(Plan, structured_llm.invoke(messages))

        return {
            "plan": plan,
            "messages": [AIMessage(content=plan.model_dump_json())],
            "reflection_steps": state.get(
                "reflection_steps", self.max_reflection_steps
            ),
        }

    def reflection_node(self, state: PlanningState) -> PlanningState:
        print("PlanningAgent: reflecting . . .")

        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [state["messages"][0]] + [
            cls_map[msg.type](content=msg.content)
            for msg in state["messages"][1:]
        ]
        translated = [SystemMessage(content=reflection_prompt)] + translated
        res = self.llm.invoke(
            translated,
            self.build_config(tags=["planner", "reflect"]),
        )
        return {
            "plan": state["plan"],
            "messages": [HumanMessage(content=res.content)],
            "reflection_steps": state["reflection_steps"] - 1,
        }

    def _build_graph(self):
        self.add_node(self.generation_node, "generate")
        self.add_node(self.reflection_node, "reflect")
        self.graph.set_entry_point("generate")
        self.graph.add_conditional_edges(
            "generate",
            self._wrap_cond(
                _should_reflect, "should_reflect", "planning_agent"
            ),
            {"reflect": "reflect", "END": END},
        )
        self.graph.add_conditional_edges(
            "reflect",
            self._wrap_cond(
                _should_regenerate, "should_regenerate", "planning_agent"
            ),
            {"generate": "generate", "END": END},
        )


def _should_reflect(state: PlanningState):
    # Hit the reflection cap?
    if state["reflection_steps"] > 0:
        return "reflect"

    print("PlanningAgent: Reached reflection limit")
    return "END"


def _should_regenerate(state: PlanningState):
    reviewMaxLength = 0  # 0 = no limit, else some character limit like 300 (only used for console printing)

    # Latest reviewer output (if present)
    last_content = (
        state["messages"][-1].content if state.get("messages") else ""
    )

    # Approved?
    if "[APPROVED]" in last_content:
        print("PlanningAgent: Plan APPROVED")
        return "END"

    # Not approved — print a concise reason before another cycle
    reason = " ".join(last_content.strip().split())  # collapse whitespace
    if reviewMaxLength > 0 and len(reason) > reviewMaxLength:
        reason = reason[:reviewMaxLength] + ". . ."
    print(
        f"PlanningAgent: not approved — iterating again. Reviewer notes: {reason}"
    )
    return "generate"
