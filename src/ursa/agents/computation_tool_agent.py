import os
from typing import Annotated, Literal

import randomname
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.language_models import BaseChatModel

from typing_extensions import TypedDict

from ursa.agents import (
    BaseAgent,
    ExecutionAgent,
    PlanningAgent,
)

from ..prompt_library.computation_tool_prompts import code_schema_prompt

model = ChatLiteLLM(
    model="openai/o3",
    max_tokens=50000,
)

class ComputationToolState(TypedDict):
    messages: Annotated[list, add_messages]

class ComputationToolAgent(BaseAgent):
    def __init__(
        self,
        llm: str | BaseChatModel = "openai/gpt-4o-mini",
        log_state: bool = False,
        tool_description: str = "",
        problem_description: str = "",
        tool_schema: str = code_schema_prompt,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.log_state = log_state
        self.tool_description = tool_description
        self.problem_description = problem_description
        self.tool_schema = tool_schema

        self._initialize_agent()

    # Define the function that calls the model
    def query_planner(self):
        
        #Planning portion
        planner_prompt = (
            f"Break this down into one step per technique:\n{self.problem_description}"
            f"Here is the schema used to describe the computational model:\n{self.tool_schema}"
            f"Here is the description of what to run using this schema:\n{self.tool_description}"
        )

        planner = PlanningAgent(llm=model)
        planner_config = {
            "recursion_limit": 999_999,
            "configurable": { "thread_id": planner.thread_id }
        }

        planning_output = planner.action.invoke({
            "messages": [HumanMessage(content=planner_prompt)]
        }, 
        planner_config,
        )
  
    def _initialize_agent(self):
        self.graph = StateGraph(ComputationToolState)

        self.graph.add_node("planning_agent", self.query_planner)
 #       self.graph.add_node("execution_agent", self.query_executor)

        # Set the entrypoint as `planning_agent`
        # This means that this node is the first one called
        self.graph.add_edge(START, "planning_agent")
#        self.graph.add_edge("planning_agent", "execution_agent")
        self.graph.add_edge("planning_agent", END)

        self.action = self.graph.compile(checkpointer=self.checkpointer)

    def run(self, problem_description, tool_description):
        print(problem_description)
#        inputs = {"messages": [HumanMessage(content=problem_description)]}
 #       return self.action.invoke(
  #          inputs,
   #     )

