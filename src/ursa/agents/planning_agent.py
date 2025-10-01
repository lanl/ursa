# from langgraph.checkpoint.memory  import MemorySaver
# from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
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
        self, llm: str | BaseChatModel = "openai/gpt-4o-mini", **kwargs
    ):
        super().__init__(llm, **kwargs)
        self.planner_prompt = planner_prompt
        self.formalize_prompt = formalize_prompt
        self.reflection_prompt = reflection_prompt
        self._initialize_agent()

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
                    messages, {"configurable": {"thread_id": self.thread_id}}
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
                    translated, {"configurable": {"thread_id": self.thread_id}}
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
            translated, {"configurable": {"thread_id": self.thread_id}}
        )
        return {"messages": [HumanMessage(content=res.content)]}

    def _initialize_agent(self):
        self.graph = StateGraph(PlanningState)
        self.graph.add_node("generate", self.generation_node)
        self.graph.add_node("reflect", self.reflection_node)
        self.graph.add_node("formalize", self.formalize_node)

        self.graph.add_edge(START, "generate")
        self.graph.add_edge("generate", "reflect")
        self.graph.add_edge("formalize", END)

        self.graph.add_conditional_edges(
            "reflect",
            should_continue,
            {"generate": "generate", "formalize": "formalize"},
        )

        # memory      = MemorySaver()
        # self.action = self.graph.compile(checkpointer=memory)
        self.action = self.graph.compile(checkpointer=self.checkpointer)
        # self.action.get_graph().draw_mermaid_png(output_file_path="planning_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

    def run(self, prompt, recursion_limit=100):
        initial_state = {"messages": [HumanMessage(content=prompt)]}
        return self.action.invoke(
            initial_state,
            {
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": self.thread_id},
            },
        )


config = {"configurable": {"thread_id": "1"}}


def should_continue(state: PlanningState):
    if len(state["messages"]) > (state.get("reflection_steps", 3) + 3):
        return "formalize"
    if "[APPROVED]" in state["messages"][-1].content:
        return "formalize"
    return "generate"


def main():
    planning_agent = PlanningAgent()
    for event in planning_agent.action.stream(
        {
            "messages": [
                HumanMessage(
                    content="Find a city with as least 10 vowels in its name."  # "Write an essay on ideal high-entropy alloys for spacecraft."
                )
            ],
        },
        config,
    ):
        print("-" * 30)
        print(event.keys())
        print(event[list(event.keys())[0]]["messages"][-1].content)
        print("-" * 30)


if __name__ == "__main__":
    main()
