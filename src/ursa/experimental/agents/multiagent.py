from pathlib import Path
from typing import Optional

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.util import Checkpointer

system_prompt = """\
You are an agent with multiple subagents and tools.

These agents are available to you:

* execution_agent
  * Use this agent whenever you are asked to write/edit code or run arbitrary
    commands from the command line.

* planning_agent
  * Use this agent whenever you are asked to plan out tasks.

Note that if the user asks you to plan and then execute a task, you are 
to iterate through each part (step or bullet point) of a task and then
carry out the execution agent. Here is an example query:

Please make a plan to print the first 10 natural numbers in python, then execute
the code.

For this query, a generated plan might look like this:

```
The user wants to compute the first 10 natural numbers in python. This is the plan.

* step 1: write code
* step 2: check that code is correct
```

In this case you should call the execution agent for step 1; and then call the
execution agent for step 2. If more steps are in the plan, keep calling the
execution agent.
"""


# NOTE: Is the solution to have a tool that breaks up the string plan, and then
# execute each section of the plan?
@tool
def execute_plan(plan: str):
    """Execute plan item by item."""
    ...


class Ursa:
    def __init__(
        self,
        llm: BaseChatModel,
        extra_tools: list = [],
        workspace: Path = Path("ursa-workspace"),
        checkpointer: Optional[BaseCheckpointSaver] = None,
        thread_id: str = "ursa",
        max_reflection_steps: int = 1,
        system_prompt: str = system_prompt,
    ):
        self.llm = llm
        self.extra_tools = extra_tools
        self.workspace = workspace
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.system_prompt = system_prompt
        self.max_reflection_steps = max_reflection_steps
        self.checkpointer = checkpointer or Checkpointer.from_workspace(
            workspace
        )

    def make_planning_tool(self):
        planning_agent = PlanningAgent(
            self.llm,
            max_reflection_steps=self.max_reflection_steps,
            thread_id=self.thread_id,
        )

        @tool(
            "planning_agent",
            description="Create plans for arbitrary tasks",
        )
        def call_agent(query: str):
            result = planning_agent.invoke({
                "messages": [HumanMessage(query)],
            })
            return result["messages"][-1].content

        return call_agent

    def make_execution_tool(self):
        execution_agent = ExecutionAgent(self.llm, thread_id=self.thread_id)

        @tool(
            "execution_agent",
            description="Read and edit scripts/code, and execute arbitrary commands on command line.",
        )
        def call_agent(query: str):
            result = execution_agent.invoke({
                "messages": [HumanMessage(query)],
                "workspace": str(self.workspace),
            })
            return result["messages"][-1].content

        return call_agent

    def create(self, **kwargs):
        """Create agent.

        kwargs: for `create_agent`
        """
        self.subagents = [
            self.make_execution_tool(),
            self.make_planning_tool(),
        ]
        self.tools = self.subagents + self.extra_tools
        return create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            **kwargs,
        )
