import json
from pathlib import Path
from typing import Optional

import yaml
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.util import Checkpointer

system_prompt = """\
You are a data scientist with multiple tools.

These tools are available to you:

* planning_agent
  * Use this tool whenever you are asked to plan out tasks.
  * In each step of your plan, if code needs to be generated, please explicitly
    state in the step that code needs to be written and executed.

* execution_agent
  * Use this tool **whenever** you are asked to write/edit code or run
    arbitrary commands from the command line.

* execute_plan_tool
  * Use this tool if you are asked to execute a plan that starts with <PLAN>
    and ends with </PLAN>.
  * Do not use this tool if the <PLAN></PLAN> tags are not present in the
    instruction!

Note that this project is managed by `uv. So, if you need to execute python
code, you MUST run `uv run path/to/file.py`. DO NOT run `python
/path/to/file.py` or `python3 /path/to/file.py`.
"""


def tag(tag_name: str, content: str):
    """Wrap content in XML tag"""
    return f"\n<{tag_name}>\n{content}\n</{tag_name}>\n\n"


# NOTE: Resources
# https://docs.langchain.com/oss/python/langchain/multi-agent#where-to-customize
def make_execute_plan_tool(llm: BaseChatModel, workspace: Path):
    execution_agent = ExecutionAgent(llm)._action

    @tool(
        "execute_plan_tool",
        description="Execute a plan from the planning agent tool.",
    )
    def execute_plan(plan: str):
        """Execute plan item by item."""

        print("EXECUTING PLAN")
        if plan.startswith("<PLAN>") and plan.endswith("</PLAN>"):
            summaries = []

            task_and_plan_steps = json.loads(
                plan.replace("<PLAN>", "").replace("</PLAN>", "").strip()
            )
            task = task_and_plan_steps[0]["task"]
            plan_steps = task_and_plan_steps[1:]
            for step in plan_steps:
                step_prompt = (
                    "You are contributing a solution of an overall plan. "
                    "The overall plan, last step's summary, and next step are provided below. "
                    "With the provided information, please carry out the next step. "
                    "IF you write any code, be sure to execute the code to make "
                    "sure it properly runs."
                )
                step_prompt += tag("OVERALL_PLAN", task)
                if len(summaries) > 0:
                    last_step_summary = summaries[-1]
                    step_prompt += tag(
                        "SUMMARY_OF_LAST_STEP", last_step_summary
                    )

                step_prompt += tag("NEXT_STEP", yaml.dump(step).strip())
                print(step_prompt)

                result = execution_agent.invoke({
                    "messages": [HumanMessage(step_prompt)],
                    "workspace": str(workspace),
                })
                last_step_summary = result["messages"][-1].content
                summaries.append(last_step_summary)
            return "Grand summary of plan execution:\n\n" + "\n\n".join(
                summaries
            )
        else:
            return (
                "Could not use `execute_plan` tool execute plan "
                "as plan does not start/end with <PLAN>/</PLAN>."
            )

    return execute_plan


def make_planning_tool(llm: BaseChatModel, max_reflection_steps: int):
    planning_agent = PlanningAgent(
        llm, max_reflection_steps=max_reflection_steps
    )._action

    @tool(
        "planning_agent",
        description="Create plans for arbitrary tasks",
    )
    def call_agent(query: str):
        result = planning_agent.invoke({
            "messages": [HumanMessage(query)],
            "reflection_steps": max_reflection_steps,
        })
        # return result["messages"][-1].content
        plan_steps = [{"task": query}] + result["plan_steps"]
        plan = f"<PLAN>\n{json.dumps(plan_steps)}\n</PLAN>"
        # print(json.dumps(plan_steps, indent=4))
        print(yaml.dump(plan_steps))
        return plan

    return call_agent


def make_execution_tool(llm: BaseChatModel, workspace: Path):
    execution_agent = ExecutionAgent(llm)._action

    @tool(
        "execution_agent",
        description="Read and edit scripts/code, and execute arbitrary commands on command line.",
    )
    def call_agent(query: str):
        result = execution_agent.invoke({
            "messages": [HumanMessage(query)],
            "workspace": str(workspace),
        })
        return result["messages"][-1].content

    return call_agent


class Ursa:
    def __init__(
        self,
        llm: BaseChatModel,
        extra_tools: Optional[list] = None,
        workspace: Path = Path("ursa-workspace"),
        checkpointer: Optional[BaseCheckpointSaver] = None,
        thread_id: str = "ursa",
        max_reflection_steps: int = 1,
        system_prompt: str = system_prompt,
    ):
        self.llm = llm
        self.extra_tools = extra_tools or []
        self.workspace = workspace
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.system_prompt = system_prompt
        self.max_reflection_steps = max_reflection_steps
        self.checkpointer = checkpointer or Checkpointer.from_workspace(
            workspace
        )

    def create(self, **kwargs):
        """Create agent.

        kwargs: for `create_agent`
        """
        self.subagents = [
            make_execution_tool(llm=self.llm, workspace=self.workspace),
            make_planning_tool(
                llm=self.llm, max_reflection_steps=self.max_reflection_steps
            ),
            make_execute_plan_tool(llm=self.llm, workspace=self.workspace),
        ]
        self.tools = self.subagents + self.extra_tools
        return create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            **kwargs,
        )
