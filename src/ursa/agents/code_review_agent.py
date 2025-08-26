import os
import subprocess
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from typing_extensions import TypedDict

from ..prompt_library.code_review_prompts import (
    get_code_review_prompt,
    get_plan_review_prompt,
)
from ..prompt_library.execution_prompts import summarize_prompt

# from langchain_core.runnables.graph import MermaidDrawMethod
from .base import BaseAgent

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

code_extensions = [
    ".py",
    ".R",
    ".jl",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".c++",
    ".C",
    ".f90",
    ".f95",
    ".f03",
]


class CodeReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    project_prompt: str
    code_files: list[str]
    edited_files: list[str]
    workspace: str
    iteration: int


class CodeReviewAgent(BaseAgent):
    def __init__(
        self, llm: str | BaseChatModel = "openai/gpt-4o-mini", **kwargs
    ):
        super().__init__(llm, **kwargs)
        print("### WORK IN PROGRESS ###")
        print(
            "CODE REVIEW AGENT NOT YET FULLY IMPLEMENTED AND TESTED. BE AWARE THAT IT WILL LIKELY NOT WORK AS INTENDED YET."
        )
        self.summarize_prompt = summarize_prompt
        self.tools = [run_cmd, write_file, read_file]
        self.tool_node = ToolNode(self.tools)
        self.llm = self.llm.bind_tools(self.tools)

        self._initialize_agent()

    # Define the function that calls the model
    def plan_review(self, state: CodeReviewState) -> CodeReviewState:
        new_state = state.copy()

        assert "workspace" in new_state.keys(), "No workspace set for review!"

        plan_review_prompt = get_plan_review_prompt(
            project_prompt=state["project_prompt"],
            file_list=state["code_files"],
        )
        new_state["messages"] = [
            SystemMessage(content=plan_review_prompt)
        ] + state["messages"]
        response = self.llm.invoke(
            new_state["messages"],
            {"configurable": {"thread_id": self.thread_id}},
        )
        return {"messages": [response]}

    # Define the function that calls the model
    def file_review(self, state: CodeReviewState) -> CodeReviewState:
        new_state = state.copy()
        code_review_prompt = get_code_review_prompt(
            project_prompt=state["project_prompt"],
            file_list=state["code_files"],
        )
        filename = state["code_files"][state["iteration"]]
        new_state["messages"][0] = SystemMessage(content=code_review_prompt)
        new_state["messages"].append(
            HumanMessage(content=f"Please review {filename}")
        )
        response = self.llm.invoke(
            new_state["messages"],
            {"configurable": {"thread_id": self.thread_id}},
        )
        return {"messages": [response]}

    # Define the function that calls the model
    def summarize(self, state: CodeReviewState) -> CodeReviewState:
        messages = [SystemMessage(content=summarize_prompt)] + state["messages"]
        response = self.llm.invoke(
            messages, {"configurable": {"thread_id": self.thread_id}}
        )
        return {"messages": [response.content]}

    def increment(self, state: CodeReviewState) -> CodeReviewState:
        new_state = state.copy()
        new_state["iteration"] += 1
        if new_state["iteration"] >= len(new_state["code_files"]):
            new_state["iteration"] = -1
        print(
            f"On to file {new_state['iteration'] + 1} out of {len(new_state['code_files'])}"
        )
        return new_state

    # Define the function that calls the model
    def safety_check(self, state: CodeReviewState) -> CodeReviewState:
        new_state = state.copy()
        if state["messages"][-1].tool_calls[0]["name"] == "run_cmd":
            query = state["messages"][-1].tool_calls[0]["args"]["query"]
            safety_check = self.llm.invoke(
                (
                    "Assume commands to run python and Julia are safe because "
                    "the files are from a trusted source. "
                    "Answer only either [YES] or [NO]. Is this command safe to run: "
                )
                + query,
                {"configurable": {"thread_id": self.thread_id}},
            )
            if "[NO]" in safety_check.content:
                print(f"{RED}{BOLD} [WARNING] {RESET}")
                print(
                    f"{RED}{BOLD} [WARNING] That command deemed unsafe and cannot be run: {RESET}",
                    query,
                    " --- ",
                    safety_check,
                )
                print(f"{RED}{BOLD} [WARNING] {RESET}")
                return {
                    "messages": [
                        "[UNSAFE] That command deemed unsafe and cannot be run: "
                        + query
                    ]
                }

            print(f"{GREEN}[PASSED] the safety check: {RESET}" + query)
        elif state["messages"][-1].tool_calls[0]["name"] == "write_code":
            fn = (
                state["messages"][-1]
                .tool_calls[0]["args"]
                .get("filename", None)
            )
            if "code_files" in new_state:
                if fn not in new_state["code_files"]:
                    new_state["code_files"].append(fn)
                    new_state["edited_files"].append(fn)
                else:
                    new_state["edited_files"].append(fn)
            else:
                new_state["code_files"] = [fn]

        return new_state

    def _initialize_agent(self):
        self.graph = StateGraph(CodeReviewState)

        self.graph.add_node("plan_review", self.plan_review)
        self.graph.add_node("file_review", self.file_review)
        self.graph.add_node("increment", self.increment)
        self.graph.add_node("action", self.tool_node)
        self.graph.add_node("summarize", self.summarize)
        self.graph.add_node("safety_check", self.safety_check)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.add_edge(START, "plan_review")

        self.graph.add_conditional_edges(
            "file_review",
            should_continue,
            {
                "action": "safety_check",
                "increment": "increment",
                "summarize": "summarize",
            },
        )

        self.graph.add_conditional_edges(
            "safety_check",
            command_safe,
            {
                "safe": "action",
                "unsafe": "file_review",
            },
        )

        self.graph.add_edge("plan_review", "file_review")
        self.graph.add_edge("action", "file_review")
        self.graph.add_edge("increment", "file_review")
        self.graph.add_edge("summarize", END)

        self.action = self.graph.compile(checkpointer=self.checkpointer)
        # self.action.get_graph().draw_mermaid_png(output_file_path="code_review_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

    def run(self, prompt, workspace):
        code_files = [
            x
            for x in os.listdir(workspace)
            if any([ext in x for ext in code_extensions])
        ]
        initial_state = {
            "messages": [],
            "project_prompt": prompt,
            "code_files": code_files,
            "edited_files": [],
            "iteration": 0,
            "workspace": workspace,
        }
        return self.action.invoke(
            initial_state, {"configurable": {"thread_id": self.thread_id}}
        )


@tool
def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Run command from commandline"""
    workspace_dir = state["workspace"]

    print("RUNNING: ", query)
    process = subprocess.Popen(
        query.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace_dir,
    )

    stdout, stderr = process.communicate(timeout=600)

    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"


@tool
def read_file(filename: str, state: Annotated[dict, InjectedState]):
    """
    Reads in a file with a given filename into a string

    Args:
        filename: string filename to read in
    """
    workspace_dir = state["workspace"]
    full_filename = os.path.join(workspace_dir, filename)

    print("[READING]: ", full_filename)
    with open(full_filename, "r") as file:
        file_contents = file.read()
    return file_contents


@tool
def write_file(code: str, filename: str, state: Annotated[dict, InjectedState]):
    """
    Writes text to a file in the given workspace as requested.

    Args:
        code: Text to write to a file
        filename: the filename to write to

    Returns:
        Execution results
    """
    workspace_dir = state["workspace"]

    print("[WRITING]: ", filename)
    try:
        # Extract code if wrapped in markdown code blocks
        if "```" in code:
            code_parts = code.split("```")
            if len(code_parts) >= 3:
                # Extract the actual code
                if "\n" in code_parts[1]:
                    code = "\n".join(code_parts[1].strip().split("\n")[1:])
                else:
                    code = code_parts[2].strip()

        # Write code to a file
        code_file = os.path.join(workspace_dir, filename)

        with open(code_file, "w") as f:
            f.write(code)
        print(f"Written code to file: {code_file}")

        return f"File {filename} written successfully."

    except Exception as e:
        print(f"Error generating code: {str(e)}")
        # Return minimal code that prints the error
        return f"Failed to write {filename} successfully."


# Define the function that determines whether to continue or not
def should_continue(
    state: CodeReviewState,
) -> Literal["summarize", "increment", "action"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        if state["iteration"] == -1:
            target_node = "summarize"
        else:
            target_node = "increment"
    # Otherwise if there is, we use the tool
    else:
        target_node = "action"
    return target_node


# Define the function that determines whether to continue or not
def command_safe(state: CodeReviewState) -> Literal["safe", "unsafe"]:
    messages = state["messages"]
    last_message = messages[-1]
    if "[UNSAFE]" in last_message.content:
        return "unsafe"
    else:
        return "safe"


def main():
    code_review_agent = CodeReviewAgent(llm="openai/o3-mini")
    initial_state = {
        "messages": [],
        "project_prompt": "Find a city with as least 10 vowels in its name.",
        "code_files": ["vowel_count.py"],
        "edited_files": [],
        "iteration": 0,
    }
    result = (
        code_review_agent.action.invoke(initial_state),
        {"configurable": {"thread_id": 42}},
    )
    for x in result["messages"]:
        print(x.content)
    return result


if __name__ == "__main__":
    main()
