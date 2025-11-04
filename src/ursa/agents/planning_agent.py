# from langgraph.checkpoint.memory  import MemorySaver
# from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Annotated, Any, Dict, Iterator, List, Mapping, Optional

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import Field
from typing_extensions import TypedDict

from ..prompt_library.planning_prompts import (
    formalize_prompt,
    planner_prompt,
    reflection_prompt,
)
from ..util.parse import extract_json
from .base import BaseAgent


class PlanningState(TypedDict):
    messages: Annotated[list, add_messages]
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Ordered steps in the solution plan"
    )
    reflection_steps: Optional[int] = Field(
        default=3, description="Number of reflection steps"
    )


class PlanningAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.planner_prompt = planner_prompt
        self.formalize_prompt = formalize_prompt
        self.reflection_prompt = reflection_prompt
        self._action = self._build_graph()

    def generation_node(self, state: PlanningState) -> PlanningState:
        print("PlanningAgent: generating . . .")
        messages = state["messages"]
        if isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=self.planner_prompt)
        else:
            messages = [SystemMessage(content=self.planner_prompt)] + messages
        return {
            "messages": [
                self.llm.invoke(
                    messages,
                    self.build_config(tags=["planner", "generate"]),
                )
            ]
        }

    def formalize_node(self, state: PlanningState) -> PlanningState:
        print("PlanningAgent: formalizing . . .")
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [state["messages"][0]] + [
            cls_map[msg.type](content=msg.content)
            for msg in state["messages"][1:]
        ]
        translated = [SystemMessage(content=self.formalize_prompt)] + translated
        for _ in range(10):
            try:
                res = self.llm.invoke(
                    translated,
                    self.build_config(tags=["planner", "formalize"]),
                )
                json_out = extract_json(res.content)
                break
            except ValueError:
                translated.append(
                    HumanMessage(
                        content="Your response was not valid JSON. Try again."
                    )
                )
        return {
            "messages": [HumanMessage(content=res.content)],
            "plan_steps": json_out,
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
        return {"messages": [HumanMessage(content=res.content)]}

    def _build_graph(self):
        graph = StateGraph(PlanningState)
        self.add_node(graph, self.generation_node, "generate")
        self.add_node(graph, self.reflection_node, "reflect")
        self.add_node(graph, self.formalize_node, "formalize")

        # Edges
        graph.set_entry_point("generate")
        graph.add_edge("generate", "reflect")
        graph.set_finish_point("formalize")

        # Time the router logic too
        graph.add_conditional_edges(
            "reflect",
            self._wrap_cond(should_continue, "should_continue", "planner"),
            {"generate": "generate", "formalize": "formalize"},
        )

        # memory      = MemorySaver()
        # self.action = self.graph.compile(checkpointer=memory)
        return graph.compile(checkpointer=self.checkpointer)
        # self.action.get_graph().draw_mermaid_png(output_file_path="planning_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 1000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
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

        # Delegate to the compiled graph's stream
        yield from self._action.stream(inputs, merged)

    # prevent bypass
    @property
    def action(self):
        raise AttributeError(
            "Use .stream(...) or .invoke(...); direct .action access is unsupported."
        )


config = {"configurable": {"thread_id": "1"}}


def should_continue(state: PlanningState):
    reviewMaxLength = 0  # 0 = no limit, else some character limit like 300

    # Latest reviewer output (if present)
    last_content = (
        state["messages"][-1].content if state.get("messages") else ""
    )

    max_reflections = state.get("reflection_steps", 3)

    # Hit the reflection cap?
    if len(state["messages"]) > (max_reflections + 3):
        print(
            f"PlanningAgent: reached reflection limit ({max_reflections}); formalizing . . ."
        )
        return "formalize"

    # Approved?
    if "[APPROVED]" in last_content:
        print("PlanningAgent: [APPROVED] — formalizing . . .")
        return "formalize"

    # Not approved — print a concise reason before another cycle
    reason = " ".join(last_content.strip().split())  # collapse whitespace
    if reviewMaxLength > 0 and len(reason) > reviewMaxLength:
        reason = reason[:reviewMaxLength] + ". . ."
    print(
        f"PlanningAgent: not approved — iterating again. Reviewer notes: {reason}"
    )

    return "generate"


def main():
    planning_agent = PlanningAgent()

    for event in planning_agent.stream(
        {
            "messages": [
                HumanMessage(
                    content="Find a city with at least 10 vowels in its name."
                )
            ],
        },
    ):
        print("-" * 30)
        print(event.keys())
        print(event[list(event.keys())[0]]["messages"][-1].content)
        print("-" * 30)


if __name__ == "__main__":
    main()
