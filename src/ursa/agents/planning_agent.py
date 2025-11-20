from typing import Annotated, Any, Iterator, List, Mapping, Optional

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..prompt_library.planning_prompts import (
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
    expected_outputs: List[str] = Field(
        description="Concrete artifacts or results produced by this step"
    )
    success_criteria: List[str] = Field(
        description="Measurable checks that indicate the step succeeded"
    )


class Plan(BaseModel):
    steps: List[PlanStep] = Field(
        description="Ordered list of steps to solve the problem"
    )


# planning state
class PlanningState(TypedDict):
    """Here is the planning state"""

    messages: Annotated[list, add_messages]
    plan_steps: Optional[List[PlanStep]] = Field(
        description="Ordered steps in the solution plan"
    )
    reflection_steps: Optional[int] = Field(
        default=3, description="Number of reflection steps"
    )


class PlanningAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.planner_prompt = planner_prompt
        self.reflection_prompt = reflection_prompt
        self._action = self._build_graph()

    def generation_node(self, state: PlanningState) -> PlanningState:
        """
        Plan generation with structured output. Produces a JSON string in messages
        and a parsed list of steps in state["plan_steps"].
        """

        print("PlanningAgent: generating . . .")

        messages = state["messages"]
        if isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=self.planner_prompt)
        else:
            messages = [SystemMessage(content=self.planner_prompt)] + messages

        structured_llm = self.llm.with_structured_output(Plan)
        plan_obj: Plan = structured_llm.invoke(
            messages, self.build_config(tags=["planner"])
        )

        try:
            json_text = plan_obj.model_dump_json(indent=2)

        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize Plan object with Pydantic v2: {e}"
            )

        return {
            "messages": [AIMessage(content=json_text)],
            "plan_steps": plan_obj.steps,
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
            "messages": [HumanMessage(content=res.content)],
            "reflection_steps": state["reflection_steps"] - 1,
        }

    def _build_graph(self):
        graph = StateGraph(PlanningState)
        self.add_node(graph, self.generation_node, "generate")
        self.add_node(graph, self.reflection_node, "reflect")
        graph.set_entry_point("generate")
        graph.add_conditional_edges(
            "generate",
            self._wrap_cond(
                _should_reflect, "should_reflect", "planning_agent"
            ),
            {"reflect": "reflect", "END": END},
        )
        graph.add_conditional_edges(
            "reflect",
            self._wrap_cond(
                _should_regenerate, "should_regenerate", "planning_agent"
            ),
            {"generate": "generate", "END": END},
        )
        return graph.compile(checkpointer=self.checkpointer)

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 1000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["planner"]
        )
        inputs.setdefault("reflection_steps", 1)
        return self._action.invoke(inputs, config)

    def _stream(
        self,
        inputs: Mapping[str, Any],
        *,
        config: dict | None = None,
        recursion_limit: int = 1000,
        **_,
    ) -> Iterator[dict]:
        # If you have defaults, merge them here:
        default = self.build_config(
            recursion_limit=recursion_limit, tags=["planner"]
        )
        if config:
            merged = {**default, **config}
            if "configurable" in config:
                merged["configurable"] = {
                    **default.get("configurable", {}),
                    **config["configurable"],
                }
        else:
            merged = default

        inputs.setdefault("reflection_steps", 1)
        # Delegate to the compiled graph's stream
        yield from self._action.stream(inputs, merged)


def _should_reflect(state: PlanningState):
    # Hit the reflection cap?
    steps = state.get("reflection_steps")
    if steps == 0:
        print("PlanningAgent: Reached reflection limit")
        return "END"
    else:
        return "reflect"


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
